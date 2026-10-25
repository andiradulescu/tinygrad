import ctypes, tempfile, unittest
from pathlib import Path
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.helpers import Context
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.uop.ops import UOp
from tinygrad.runtime.ops_cl import CLDevice, CLAllocator

@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestCLError(unittest.TestCase):
  @unittest.skip("allocates tons of memory")
  def test_oom(self):
    with self.assertRaises(RuntimeError) as err:
      allocator = CLAllocator(CLDevice())
      for i in range(1_000_000):
        allocator.alloc(1_000_000_000)
    assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"

  def test_invalid_kernel_name(self):
    device = Device[Device.DEFAULT]
    with self.assertRaises(RuntimeError) as err:
      device.function(b"__kernel void test(__global int* a) { a[0] = 1; }", "")
    assert str(err.exception) == "OpenCL Error -46: CL_INVALID_KERNEL_NAME"

  def test_unaligned_copy(self):
    data = list(range(65))
    unaligned = memoryview(bytearray(data))[1:]
    buffer = Buffer("CL", 64).allocate()
    buffer.copy_from(Buffer("PYTHON", 64, opaque=unaligned))
    result = memoryview(bytearray(len(data) - 1))
    result[:] = buffer.as_memoryview()
    assert unaligned == result, "Unaligned data copied in must be equal to data copied out."


@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestCLBufferView(unittest.TestCase):
  def test_aligned_view_alias(self):
    alignment_bits = ctypes.c_uint32()
    self.assertEqual(cl.clGetDeviceInfo(Device["CL"].cl_dev, cl.CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4, ctypes.byref(alignment_bits), None), 0)
    alignment = max(1, alignment_bits.value // 8)
    for lru in (0, 1):
      with self.subTest(lru=lru), Context(LRU=lru):
        base = Buffer("CL", 3*alignment, initial_value=bytes([3])*3*alignment)
        view = base.view(2*alignment, alignment).view(alignment, alignment).ensure_allocated()
        self.assertEqual(bytes(view.as_memoryview()), bytes([3])*alignment)
        view.copy_from(Buffer("PYTHON", alignment, initial_value=bytes([5])*alignment))
        tensor = Tensor(UOp.from_buffer(view, dtypes.uint8))
        tensor.assign(tensor + 1).realize()
        self.assertEqual(bytes(base.as_memoryview()), bytes([3])*2*alignment + bytes([6])*alignment)
        view.deallocate()
        self.assertEqual(bytes(base.view(alignment, 2*alignment).ensure_allocated().as_memoryview()), bytes([6])*alignment)
        base.deallocate()
        base.allocate()
        base.copy_from(Buffer("PYTHON", 3*alignment, initial_value=bytes([7])*3*alignment))
        self.assertEqual(bytes(view.ensure_allocated().as_memoryview()), bytes([7])*alignment)

  def test_jit_out_of_band_reload(self):
    from examples.openpilot.helpers import dump_pickle, load_pickle
    weights = Tensor.full((64,), 2).contiguous().realize()
    @TinyJit
    def run(x): return (x * weights + 1).realize()
    for value in (1, 2, 3):
      self.assertEqual(run(Tensor.full((64,), value).contiguous().realize()).tolist(), [2*value+1]*64)
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "jit.pkl"
      dump_pickle(run, path)
      loaded = load_pickle(path, out_of_band=True)
    for value in (4, 5):
      self.assertEqual(loaded(Tensor.full((64,), value).contiguous().realize()).tolist(), [2*value+1]*64)

  def test_misaligned_view(self):
    alignment_bits = ctypes.c_uint32()
    self.assertEqual(cl.clGetDeviceInfo(Device["CL"].cl_dev, cl.CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4, ctypes.byref(alignment_bits), None), 0)
    if alignment_bits.value <= 8: self.skipTest("device supports byte-aligned sub-buffers")
    base = Buffer("CL", 32).allocate()
    with self.assertRaisesRegex(RuntimeError, "CL_MISALIGNED_SUB_BUFFER_OFFSET"):
      base.view(16, 1).ensure_allocated()

  def test_host_view_alias(self):
    base = Buffer("CL", 32, options=BufferSpec(host=True)).allocate()
    base.host[:] = bytes(range(32))
    view = base.view(16, 8).ensure_allocated()
    self.assertEqual(bytes(view.as_memoryview()), bytes(range(8, 24)))
    view.host[:] = bytes([7])*16
    self.assertEqual(bytes(base.as_memoryview()), bytes(range(8)) + bytes([7])*16 + bytes(range(24, 32)))
