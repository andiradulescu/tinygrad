import itertools, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops
from tinygrad.renderer.cstyle import MetalRenderer


@unittest.skipUnless(Device.DEFAULT == 'METAL', 'Metal required')
class TestTensorOps(unittest.TestCase):
  def setUp(self):
    if Device['METAL'].renderer.target.arch != 'Apple10': self.skipTest('Apple10 required')

  def run_matmul(self, result, accelerated=True):
    linear, _ = result.linear_with_vars()
    programs = [to_program(call.src[0], Device['METAL'].renderer) for call in linear.src]
    self.assertEqual(any('matmul2d' in u.arg for p in programs for u in p.src if u.op is Ops.SOURCE), accelerated)
    run_linear(linear)
    return result.numpy()

  def test_matmul(self):
    rng = np.random.default_rng(42)
    a, b = [rng.normal(0, .1, shape).astype(np.float16) for shape in [(64, 128), (128, 256)]]
    x, y = Tensor(a).realize(), Tensor(b).realize()
    np.testing.assert_allclose(self.run_matmul(x @ y), (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), atol=.002, rtol=.002)

  def test_layouts_and_edges(self):
    rng = np.random.default_rng(12)
    for (m, k, n), ta, tb in itertools.product([(65, 67, 69), (288, 512, 2048), (288, 2048, 512)], [False, True], [False, True]):
      with self.subTest(shape=(m, k, n), transpose=(ta, tb)):
        a, b = [rng.normal(0, .1, shape).astype(np.float16) for shape in [(k, m) if ta else (m, k), (n, k) if tb else (k, n)]]
        x, y = Tensor(a).realize(), Tensor(b).realize()
        if ta: x, a = x.T, a.T
        if tb: y, b = y.T, b.T
        np.testing.assert_allclose(self.run_matmul(x @ y), (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), atol=.002, rtol=.002)

  def test_changed_inputs(self):
    a = Tensor.zeros(67, 65, dtype=dtypes.half).contiguous().realize()
    b = Tensor.ones(67, 69, dtype=dtypes.half).contiguous().realize()
    run = TinyJit(lambda a, b: (a.T @ b).realize())
    for value in [0, 1, 2, -1, 3]:
      a.assign(Tensor.full(a.shape, value, dtype=a.dtype)).realize()
      np.testing.assert_array_equal(run(a, b).numpy(), np.full((65, 69), 67*value, dtype=np.float16))

  def test_fallbacks(self):
    rng = np.random.default_rng(21)
    a, b = [rng.normal(0, .1, shape).astype(np.float16) for shape in [(65, 67), (67, 69)]]
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    for flags in [dict(TC=0), dict(NOOPT=1), dict(TC_SELECT=1)]:
      with self.subTest(flags=flags), Context(**flags):
        x, y = Tensor(a.copy()).realize(), Tensor(b.copy()).realize()
        np.testing.assert_allclose(self.run_matmul(x @ y, False), expected, atol=.002, rtol=.002)
    x, y = Tensor(a.copy()).realize(), Tensor(b.copy()).realize()
    np.testing.assert_allclose(self.run_matmul(((x @ y) + 1).relu()), np.maximum(expected+1, 0), atol=.002, rtol=.002)
    x, y = Tensor(a.copy()).realize(), Tensor(b.copy()).realize()
    np.testing.assert_allclose(self.run_matmul(x[:17] @ y, False), expected[:17], atol=.002, rtol=.002)

  def test_fp32_accumulation(self):
    # Exact cancellation with a residual that is lost when the accumulator rounds to FP16 each step.
    a = np.tile(np.array([2048., 1., -2048., 1.], dtype=np.float16), (65, 512))
    b = np.ones((2048, 69), dtype=np.float16)
    x, y = Tensor(a).realize(), Tensor(b).realize()
    np.testing.assert_array_equal(self.run_matmul(x @ y), np.full((65, 69), 1024., dtype=np.float16))
    np.testing.assert_array_equal(self.run_matmul(x.matmul(y, dtype=dtypes.float)), np.full((65, 69), 1024., dtype=np.float32))

  def test_bias_residual_activation(self):
    rng = np.random.default_rng(9)
    a, b = [rng.normal(0, .1, shape).astype(np.float16) for shape in [(65, 67), (67, 69)]]
    bias = rng.normal(0, .1, (69,)).astype(np.float32)
    residual = rng.normal(0, .1, (65, 69)).astype(np.float32)
    x, y, z, r = [Tensor(v).realize() for v in (a, b, bias, residual)]
    expected = a.astype(np.float32) @ b.astype(np.float32) + bias + residual
    np.testing.assert_allclose(self.run_matmul((x.matmul(y, dtype=dtypes.float)+z+r).silu()),
                               expected/(1+np.exp(-expected)), atol=2e-4, rtol=2e-4)

  def test_shared_input(self):
    a = Tensor.ones(64, 64, dtype=dtypes.half).contiguous().realize()
    np.testing.assert_array_equal(self.run_matmul(a @ a), np.full((64, 64), 64., dtype=np.float16))

  def test_spatial_axes_and_transposed_output(self):
    rng = np.random.default_rng(32)
    a = rng.normal(0, .1, (2, 32, 67)).astype(np.float16)
    b = rng.normal(0, .1, (67, 69)).astype(np.float16)
    bias = rng.normal(0, .1, (69,)).astype(np.float16)
    x, y, z = [Tensor(v).realize() for v in (a, b, bias)]
    expected = ((a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)+bias).transpose(2, 0, 1)
    np.testing.assert_allclose(self.run_matmul(((x @ y)+z).permute(2, 0, 1).contiguous()), expected, atol=.002, rtol=.002)

  def test_gradient(self):
    a = Tensor.ones(65, 67, dtype=dtypes.half).contiguous().realize()
    b = Tensor.ones(67, 69, dtype=dtypes.half).contiguous().realize()
    da, db = (a @ b).sum().gradient(a, b)
    np.testing.assert_array_equal(da.numpy(), np.full((65, 67), 69., dtype=np.float16))
    np.testing.assert_array_equal(db.numpy(), np.full((67, 69), 65., dtype=np.float16))

  def test_batched_weights_fallback(self):
    rng = np.random.default_rng(56)
    a, b = [rng.normal(0, .1, shape).astype(np.float16) for shape in [(2, 65, 67), (2, 67, 69)]]
    x, y = Tensor(a).realize(), Tensor(b).realize()
    np.testing.assert_allclose(self.run_matmul(x @ y, False), (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16), atol=.002, rtol=.002)

  def test_dtype_fallbacks(self):
    x = Tensor.ones(65, 67, dtype=dtypes.half).contiguous().realize()
    y = Tensor.ones(67, 69, dtype=dtypes.half).contiguous().realize()
    with Context(EMULATED_DTYPES='float16'):
      np.testing.assert_array_equal(self.run_matmul(x @ y, False), np.full((65, 69), 67., dtype=np.float16))
    bias = Tensor.ones(69, dtype=dtypes.fp8e4m3).contiguous().realize()
    np.testing.assert_array_equal(self.run_matmul(x.matmul(y, dtype=dtypes.float)+bias.float(), False), np.full((65, 69), 68.))

  def test_other_arch(self):
    a, b = Tensor.empty(64, 128, dtype=dtypes.half).realize(), Tensor.empty(128, 256, dtype=dtypes.half).realize()
    linear, _ = (a @ b).linear_with_vars()
    renderer = MetalRenderer(replace(Device['METAL'].renderer.target, arch='Apple9'))
    program = to_program(linear.src[0].src[0], renderer)
    self.assertNotIn('matmul2d', next(u.arg for u in program.src if u.op is Ops.SOURCE))


if __name__ == '__main__': unittest.main()
