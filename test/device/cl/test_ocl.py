import ctypes, pickle, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.device import Buffer, BufferSpec
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
class TestCLJit(unittest.TestCase):
  def test_jit_reload(self):
    weights = Tensor.full((64,), 2).contiguous().realize()
    @TinyJit
    def run(x): return (x * weights + 1).realize()
    for value in (1, 2, 3):
      self.assertEqual(run(Tensor.full((64,), value).contiguous().realize()).tolist(), [2*value+1]*64)
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "jit.pkl"
      path.write_bytes(pickle.dumps(run))
      subprocess.run([sys.executable, "-c", """
import sys
from tinygrad import Tensor
import pickle
from pathlib import Path
run = pickle.loads(Path(sys.argv[1]).read_bytes())
for value in (4, 5):
  assert run(Tensor.full((64,), value).contiguous().realize()).tolist() == [2*value+1]*64
""", str(path)], check=True, timeout=Device["CL"].wait_timeout_ms / 1000)
      loaded = pickle.loads(path.read_bytes())
    for value in (4, 5):
      self.assertEqual(loaded(Tensor.full((64,), value).contiguous().realize()).tolist(), [2*value+1]*64)

@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestCLQueue(unittest.TestCase):
  def test_progress_without_synchronize(self):
    with Context(PROFILE=1, DEBUG=0, BEAM=0):
      device = Device["CL"]
      x = Tensor.ones(1024).contiguous().realize()
      device.synchronize()
      y = (x + 1).realize()
      (buffer, _), entry = list(device.prof_ents.items())[-1]
      event = ctypes.cast(buffer.host.view(fmt="Q")[entry.en_id - 1], cl.cl_event)
      status = ctypes.c_int32()
      deadline = time.monotonic() + device.wait_timeout_ms / 1000
      try:
        while True:
          self.assertEqual(cl.clGetEventInfo(event, cl.CL_EVENT_COMMAND_EXECUTION_STATUS, 4, ctypes.byref(status), None), 0)
          if status.value <= cl.CL_COMPLETE or time.monotonic() >= deadline: break
          time.sleep(.001)
        self.assertEqual(status.value, cl.CL_COMPLETE)
      finally:
        device.synchronize()
      self.assertEqual(y.tolist(), [2.]*1024)


@unittest.skipUnless(Device.DEFAULT == "CL", "Runs only on OpenCL")
class TestCLBufferView(unittest.TestCase):
  def test_zero_offset_view_alias(self):
    for lru in (0, 1):
      with self.subTest(lru=lru), Context(LRU=lru):
        base = Buffer("CL", 512).allocate()
        view = base.view(256, 0).view(128, 0).ensure_allocated()
        for value in range(16):
          base.copy_from(Buffer("PYTHON", 512, initial_value=bytes([value])*512))
          tensor = Tensor(UOp.from_buffer(view, dtypes.uint8))
          tensor.assign(tensor + 1).realize()
          self.assertEqual(bytes(base.as_memoryview()), bytes([value+1])*128 + bytes([value])*384)
          self.assertEqual(bytes(view.as_memoryview()), bytes([value+1])*128)
          parent = Tensor(UOp.from_buffer(base, dtypes.uint8))
          parent.assign(parent + 1).realize()
          view.copy_from(Buffer("PYTHON", 128, initial_value=bytes([9])*128))
          self.assertEqual(bytes(base.as_memoryview()), bytes([9])*128 + bytes([value+1])*384)
        view.deallocate()
        base.deallocate()
        base.allocate()
        base.copy_from(Buffer("PYTHON", 512, initial_value=bytes([7])*512))
        self.assertEqual(bytes(view.ensure_allocated().as_memoryview()), bytes([7])*128)

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
        tensor.assign(tensor + 1).realize()
        base.view(alignment, 0).ensure_allocated().copy_from(Buffer("PYTHON", alignment, initial_value=bytes([9])*alignment))
        self.assertEqual(bytes(base.as_memoryview()), bytes([9])*alignment + bytes([3])*alignment + bytes([7])*alignment)
        view.deallocate()
        self.assertEqual(bytes(base.view(alignment, 2*alignment).ensure_allocated().as_memoryview()), bytes([7])*alignment)
        base.deallocate()
        base.allocate()
        base.copy_from(Buffer("PYTHON", 3*alignment, initial_value=bytes([7])*3*alignment))
        self.assertEqual(bytes(view.ensure_allocated().as_memoryview()), bytes([7])*alignment)

  def test_overlapping_view_alias(self):
    alignment_bits = ctypes.c_uint32()
    self.assertEqual(cl.clGetDeviceInfo(Device["CL"].cl_dev, cl.CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4, ctypes.byref(alignment_bits), None), 0)
    alignment = max(1, alignment_bits.value // 8)
    base = Buffer("CL", 4*alignment, initial_value=bytes(4*alignment))
    first = base.view(2*alignment, alignment).ensure_allocated()
    second = base.view(2*alignment, 2*alignment).ensure_allocated()
    a, b = (Tensor(UOp.from_buffer(v, dtypes.uint8)) for v in (first, second))
    a.assign(5).realize()
    b.assign(b + 3).realize()
    self.assertEqual(bytes(base.as_memoryview()), bytes(alignment) + bytes([5])*alignment + bytes([8])*alignment + bytes([3])*alignment)
    first.copy_from(Buffer("PYTHON", 2*alignment, initial_value=bytes([9])*2*alignment))
    self.assertEqual(bytes(base.as_memoryview()), bytes(alignment) + bytes([9])*2*alignment + bytes([3])*alignment)
    self.assertEqual(bytes(second.as_memoryview()), bytes([9])*alignment + bytes([3])*alignment)

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
