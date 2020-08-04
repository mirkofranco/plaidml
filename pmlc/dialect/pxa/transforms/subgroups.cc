// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using mlir::BlockArgument;

struct SubgroupOptions {
  // Subgroup sizes to consider
  SmallVector<int64_t, 4> subgroup_sizes = {8, 16};
  // Maximum register memory for a single subgroup (for each subgroup size)
  SmallVector<int64_t, 4> max_mem = {2240, 4480};
  // Latency to global memeory
  int64_t mem_latency = 420;
  // Latency to L2 memory
  int64_t l2_cache_latency = 125;
  // L2 Cache width (in bytes)
  int64_t cache_width = 64;
  // L2 Cache size
  int64_t cache_size = 3 * 768 * 1024;
  // The threshold of computations/memory_accesses to be memory bound
  double mem_bounded_threshold = 14;
  // Limit of inner block operations during unrolling
  int64_t inner_stmts_limit = 1250;
};

typedef DenseMap<BlockArgument, unsigned> Tiling;

struct SubgroupPlan {
  size_t subgroup_size;
  SmallVector<int64_t, 6> tile_size;
  BlockArgument primary_subgroup;
  DenseSet<BlockArgument> load_subgroups;
};

/*
class SubgroupCostModel {
 public:
  //SubgroupCostModel(stripe::Block* block, const SubgroupPlan& plan, const
proto::SubgroupPass& options)
  //              : block_(block), plan_(plan), options_(options) {}



  // For a given operation and a given tile size, compute the total amount of
  // memory accesses excluding caching effects.  Basically, if the stride of a
  // given index is 0, we dont consider that index, otherwise the number of
  // accesses is just tht product of each tile size.
  size_t num_accesses(const Tiling& tile, Operation* op) const {
    const StrideInfo& si = op_strides.find(op)->second;
    size_t num = 1;
    for (const auto& kvp : tile) {
      if (si.strides.count(kvp.first)) {
        num *= kvp.second;
      }
    }
    return num;
  }

  // Given the tiling, and the strides of the output tensor,
  size_t num_work_items(Operation* out_op) const {
    const StrideInfo& out_strides = op_strides.find(out_op)->second;
    size_t num = 1;
    for (const auto& kvp : ranges) {
      // For cases which are not stride 0 relative the output
      if (out_strides.strides.count(kvp.first)) {
        // Begin with the range of the index
        size_t prod = kvp.second;
        // If it's subgrouped and not the primary thread, reduce
        if (plan.subgroup_tile.count(kvp.first) && plan.thread_idx != kvp.first)
{ prod /= plan.subgroup_tile.find(kvp.first)->second;
        }
        // If it's extra-tiles, reduce
        if (plan.extra_tile.count(kvp.first)) {
          prod /= plan.extra_tile.find(kvp.first)->second;
        }
        // Accumulate in
        num *= prod;
      }
    }
    return num;
  }


  // The primary parallel op we are doing analysis on
  AffineParallelOp ap_op;
  //.Precomputed range values for each index
  DenseMap<BlockArgument, int64_t> ranges;
  // Precomputed stride into for each operation
  DenseMap<Operation*, StrideInfo> op_strides;
  // The subgroup plan being evaluated
  SubgroupPlan plan;
};
*/

Operation *GetOriginalDef(Value val) {
  auto opRes = val.cast<mlir::OpResult>();
  while (true) {
    auto ap = mlir::dyn_cast<AffineParallelOp>(opRes.getOwner());
    if (!ap)
      break;
    auto ret = mlir::cast<AffineYieldOp>(ap.getBody()->getTerminator());
    auto src = ret.getOperand(opRes.getResultNumber());
    opRes = src.cast<mlir::OpResult>();
  }
  return opRes.getOwner();
}

void TileAccumulations(AffineParallelOp op) {
  // Find the originating reduce
  assert(op.getNumResults() == 1);
  auto srcDef = GetOriginalDef(op.getResult(0));
  auto red = mlir::cast<AffineReduceOp>(srcDef);
  // Get strides for output
  auto si = *computeStrideInfo(red);
  // Find all the accumulation indexes (stride 0 with respect to output) and
  // tile them into an inner block
  auto ranges = *op.getConstantRanges();
  SmallVector<int64_t, 6> accumTile;
  auto steps = op.steps().cast<ArrayAttr>().getValue();
  for (size_t i = 0; i < ranges.size(); i++) {
    auto arg = op.getIVs()[i];
    if (si.strides.count(arg)) {
      accumTile.push_back(steps[i].cast<IntegerAttr>().getInt());
    } else {
      accumTile.push_back(ranges[i]);
    }
    IVLOG(1, "accumTile[" << i << "] = " << accumTile[i]);
  }
  performTiling(op, accumTile);
}

// Cache the load specified relative to the block specified
LogicalResult CacheLoad(AffineParallelOp par, AffineLoadOp load) {
  auto maybeStrides =
      computeStrideInfo(load.getAffineMap(), load.getMapOperands());
  if (!maybeStrides) {
    return failure();
  }
  IVLOG(1, "Passed point 1");
  const auto &strides = *maybeStrides;

  SmallVector<StrideInfo, 4> outer;
  SmallVector<StrideInfo, 4> inner;
  SmallVector<int64_t, 4> innerSize;
  for (size_t i = 0; i < strides.size(); i++) {
    auto dimStride = strides[i];
    outer.push_back(dimStride.outer(par.getBody()));
    inner.push_back(dimStride.inner(par.getBody()));
    auto rangeInner = inner[i].range();
    if (!rangeInner.valid) {
      return failure();
    }
    if (rangeInner.stride != 1 && rangeInner.stride != 0) {
      // TODO: Handle this case
      return failure();
    }
    // inner.push_back(rangeInner);
    innerSize.push_back(rangeInner.count());
  }
  IVLOG(1, "Passed point 2");

  auto loc = load.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  auto type = MemRefType::get(innerSize, load.getMemRefType().getElementType());
  auto localBuf = builder.create<AllocOp>(loc, type);
  auto loadLoop = builder.create<AffineParallelOp>(
      loc, ArrayRef<Type>{type}, ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
      innerSize);
  SmallVector<StrideInfo, 4> globalStrides;
  SmallVector<StrideInfo, 4> localStrides;
  for (size_t i = 0; i < strides.size(); i++) {
    auto offset = StrideInfo(loadLoop.getIVs()[i]);
    globalStrides.push_back(outer[i] + offset);
    localStrides.push_back(offset);
  }
  auto globalMap = StridesToValueMap(par.getContext(), globalStrides);
  auto localMap = StridesToValueMap(par.getContext(), localStrides);
  auto innerMap = StridesToValueMap(par.getContext(), inner);
  auto loadBuilder = loadLoop.getBodyBuilder();
  auto loaded = loadBuilder.create<AffineLoadOp>(
      loc, load.getMemRef(), globalMap.getAffineMap(), globalMap.getOperands());
  auto stored = loadBuilder.create<AffineReduceOp>(
      loc, AggregationKind::assign, loaded, localBuf, localMap.getAffineMap(),
      localMap.getOperands());
  loadBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  OpBuilder newLoadBuilder(load);
  auto newLoad = newLoadBuilder.create<AffineLoadOp>(
      loc, localBuf, innerMap.getAffineMap(), innerMap.getOperands());
  load.replaceAllUsesWith(newLoad.result());
  load.erase();

  return success();
}

void SubgroupApply(AffineParallelOp op, SubgroupPlan plan) {
  auto cheapLoad = mlir::cast<AffineLoadOp>(op.getBody()->front());
  // Perform the primary innermost tiling
  performTiling(op, plan.tile_size);
  // auto inner = mlir::cast<AffineParallelOp>(op.getBody()->front());
  // Tile over accumulations
  TileAccumulations(op);
  auto accum = mlir::cast<AffineParallelOp>(op.getBody()->front());
  CacheLoad(accum, cheapLoad);
}

struct SubgroupsPass : public SubgroupsBase<SubgroupsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { doSubgroups(op); });
  }

  void doSubgroups(AffineParallelOp op) {
    if (op.getIVs().size() != 6) {
      return;
    }
    SubgroupPlan plan;
    plan.subgroup_size = 4;
    plan.tile_size = {1, 8, 16, 1, 1, 8};
    plan.primary_subgroup = op.getIVs()[2];
    plan.load_subgroups = {op.getIVs()[5]};

    SubgroupApply(op, plan);

    IVLOG(1, "Woot");
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

} // namespace pmlc::dialect::pxa
