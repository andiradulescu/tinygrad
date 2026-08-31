import ctypes, sys, unittest
from unittest.mock import Mock
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestQCOMCommandBuffer(unittest.TestCase):
  def test_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    gpu_mem = (ctypes.c_uint32 * 2)(0xaaaaaaaa, 0xaaaaaaaa)
    cpu_mem = (ctypes.c_uint32 * 2)()
    cmd_buf = HCQBuffer(ctypes.addressof(gpu_mem), 8, view=MMIOInterface(ctypes.addressof(cpu_mem), 8))
    dev = Mock(ctx=0, cmd_buf=cmd_buf)
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678)

    dev.cmd_buf_allocator.alloc.return_value = cmd_buf.va_addr + 4
    _, obj = queue._build_gpu_command(dev)
    self.assertEqual(obj.gpuaddr, cmd_buf.va_addr + 4)
    self.assertEqual(tuple(gpu_mem), (0xaaaaaaaa, 0xaaaaaaaa))
    self.assertEqual(tuple(cpu_mem), (0, 0x12345678))

    dev.allocator.alloc.return_value = cmd_buf.offset(size=4)
    queue.bind(dev)
    self.assertEqual(queue.obj.gpuaddr, cmd_buf.va_addr)
    self.assertEqual(tuple(gpu_mem), (0xaaaaaaaa, 0xaaaaaaaa))
    self.assertEqual(tuple(cpu_mem), (0x12345678, 0x12345678))
    queue._q[0] = 0x87654321
    self.assertEqual(tuple(cpu_mem), (0x87654321, 0x12345678))

if __name__ == '__main__':
  unittest.main()
