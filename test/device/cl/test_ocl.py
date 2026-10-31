import ctypes, pickle, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import Context
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.device import Buffer
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
