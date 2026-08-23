import ctypes, sys, unittest
from unittest.mock import Mock
from tinygrad.runtime.autogen import msm_drm
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface

def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number

class TestMSMDRMUAPI(unittest.TestCase):
  def test_layouts(self):
    layouts = {
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
    }
    for struct_type, (size, offsets) in layouts.items():
      self.assertEqual((ctypes.sizeof(struct_type), tuple(x[2] for x in struct_type._real_fields_)), (size, offsets))
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE), 0x40086409)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)

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
