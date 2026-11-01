from dataclasses import replace
from tinygrad.helpers import NOOPT, USE_TC, TC_SELECT, EMULATED_DTYPES, prod
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, ProgramInfo, AxisType, GroupOp, KernelInfo
from tinygrad.renderer import Estimates


def tensorops_matmul(ctx, sink:UOp):
  if EMULATED_DTYPES: return None
  if NOOPT or USE_TC != 1 or TC_SELECT != -1 or sink.arg.optimized or sink.arg.opts_to_apply is not None or sink.arg.beam: return None
  if len(sink.src) != 1 or (end:=sink.src[0]).op is not Ops.END or len(end.src) < 3: return None
  if (store:=end.src[0]).op is not Ops.STORE or len(store.src) != 2: return None
  out, value = store.src
  if out.op is not Ops.INDEX or len(out.src) != 2 or out.src[0].op is not Ops.PARAM: return None
  if out.dtype not in (dtypes.half, dtypes.float) or out.dtype != value.dtype: return None
  reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
  if len(reductions) != 1: return None
  red = reductions[0]
  if red.op is not Ops.REDUCE or red.arg != (Ops.ADD, 0) or len(red.src) != 2 or red.dtype != dtypes.float: return None
  if (cast:=red.src[0]).op is not Ops.CAST or (mul:=cast.src[0]).op is not Ops.MUL or mul.dtype != dtypes.half: return None
  if any(x.op is not Ops.INDEX or len(x.src) != 2 or x.src[0].op is not Ops.PARAM or x.dtype != dtypes.half for x in mul.src): return None
  if any(r.op is not Ops.RANGE or r.src[0].op is not Ops.CONST or r.axis_type not in (AxisType.WEAK, AxisType.GLOBAL)
         for r in (*end.src[1:], red.src[1])): return None
  k = red.src[1]
  if len(set((*end.src[1:], k))) != len(end.src): return None
  for a, b in (mul.src, mul.src[::-1]):
    groups = [[r for r in end.src[1:] if r in x.ranges and r not in y.ranges] for x, y in ((a, b), (b, a))]
    if not all(groups) or sum(map(len, groups)) != len(end.src)-1: continue
    # Flatten spatial/batch axes belonging to one operand; shared batch axes require a different GEMM.
    axes = [g[0].replace(src=(g[0].src[0].const_like(prod(r.src[0].val for r in g)),)) for g in groups]
    coords = {r: ((axis//prod(s.src[0].val for s in g[i+1:])) % r.src[0]).simplify()
              for g, axis in zip(groups, axes) for i, r in enumerate(g)}
    m, n = axes
    M, N, K = [r.src[0].val for r in (m, n, k)]
    # Keep small GEMMs on the existing path; require dense matrices with no offset or input-side computation.
    if M < 32 or min(N, K) < 64: continue
    oi = out.src[1].substitute(coords).simplify()
    if not any(oi is expr.simplify() for expr in (m*N+n, n*M+m)): continue
    if any(buf.shape != (size,) for buf, size in ((out.src[0], M*N), (a.src[0], M*K), (b.src[0], K*N))): continue
    layouts = [(a.src[1], (m*K+k, k*M+m)), (b.src[1], (k*N+n, n*K+k))]
    ta, tb = [next((i for i, expr in enumerate(exprs) if idx.substitute(coords).simplify() is expr.simplify()), None) for idx, exprs in layouts]
    if ta is None or tb is None or out.src[0] in (a.src[0], b.src[0]): continue
    row, col = [UOp(Ops.CUSTOMI, src=(UOp.const(0, dtypes.int32),), arg=(name, dtypes.int32)) for name in ('row', 'col')]
    epilogue = store.substitute({red: UOp(Ops.CUSTOMI, src=(UOp.const(0., dtypes.float),), arg=('acc[i]', dtypes.float)),
                                **{r: idx.simplify().substitute({m:row, n:col}) for r, idx in coords.items()}})
    if any(u.op not in GroupOp.ALU | {Ops.CAST, Ops.BITCAST, Ops.INDEX, Ops.PARAM, Ops.CONST, Ops.CUSTOMI, Ops.STORE}
           for u in epilogue.toposort()): continue
    return matmul_program(ctx, sink, epilogue, out.src[0], a.src[0], b.src[0], M, N, K, ta, tb)
  return None


def matmul_program(ctx, sink:UOp, epilogue:UOp, out:UOp, a:UOp, b:UOp, m:int, n:int, k:int, ta:int, tb:int):
  from tinygrad.codegen import full_rewrite_to_sink, linearize, line_rewrite, pm_linearize_cleanups, pm_alloc_to_buf
  params = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda x: x.arg.slot)
  if any(p.dtype not in ctx.supported_dtypes() for p in params): return None
  # Reuse normal scalar lowering for casts, bias, activations and indexing after the matrix product.
  epilogue = full_rewrite_to_sink(UOp.sink(epilogue, arg=KernelInfo()), ctx, optimize=False)
  uops = line_rewrite(linearize(epilogue), pm_linearize_cleanups+pm_alloc_to_buf)
  if any(u.op is Ops.CALL for u in uops): return None
  _, code, bufs = ctx._render([a, b]+[u for u in uops if u is not a and u is not b])
  names = {u: name for name, (u, _) in bufs}
  sink = sink.replace(arg=replace(sink.arg, name=f'tensorops_{m}_{n}_{k}',
                                 estimates=Estimates(ops=2*m*n*k, mem=sum(x.max_numel()*x.dtype.itemsize for x in params))))
  info = replace(ProgramInfo.from_sink(sink, ctx.target), global_size=((n+63)//64, (m+63)//64, 1), local_size=(128, 1, 1),
                 outs=(out.arg.slot,), ins=tuple(x.arg.slot for x in params if x is not out))
  # Preserve every ABI slot, including inputs removed while simplifying the epilogue.
  fields = ' '.join(f'{ctx.param_type(x)} arg{x.arg.slot};' for x in params)
  bindings = ' '.join(f'{ctx.param_type(x)} {name} = args.arg{x.arg.slot};' for x, name in names.items())
  body = '\n'.join(code)
  source = f'''#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Args {{ {fields} }};
kernel void {sink.arg.function_name}(constant Args& args [[buffer(0)]], uint3 gid [[threadgroup_position_in_grid]]) {{
  {bindings}
  auto a = tensor({names[a]}, dextents<int, 2>{{{m if ta else k}, {k if ta else m}}}, array<int, 2>{{1, {m if ta else k}}});
  auto b = tensor({names[b]}, dextents<int, 2>{{{k if tb else n}, {n if tb else k}}}, array<int, 2>{{1, {k if tb else n}}});
  constexpr auto desc = matmul2d_descriptor(64, 64, int(dynamic_extent), {str(bool(ta)).lower()}, {str(bool(tb)).lower()});
  matmul2d<desc, execution_simdgroups<4>> op;
  auto x = a.slice({"gid.y*64, 0" if ta else "0, gid.y*64"});
  auto y = b.slice({"0, gid.x*64" if tb else "gid.x*64, 0"});
  auto acc = op.get_destination_cooperative_tensor<decltype(x), decltype(y), float>();
  op.run(x, y, acc);
  for (ushort i = 0; i < acc.get_capacity(); i++) {{
    auto coord = acc.get_multidimensional_index(i);
    int row = gid.y*64 + coord[1], col = gid.x*64 + coord[0];
    if (acc.is_valid_element(i) && row < {m} && col < {n}) {{
{body}
    }}
  }}
}}'''
  return UOp(Ops.PROGRAM, arg=info, src=(sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source)))


pm_tensorops = PatternMatcher([(UPat(Ops.SINK, name='sink'), tensorops_matmul)])
