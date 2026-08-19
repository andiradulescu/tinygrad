import ctypes, errno, mmap, sys, unittest
from unittest.mock import Mock, patch
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
    self.assertEqual(msm_drm.MSM_BO_NO_SHARE, 4)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE), 0x40086409)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestQCOMCommandBuffer(unittest.TestCase):
  def test_cpu_view(self):
    from tinygrad.runtime.ops_qcom import KGSLIface, QCOMComputeQueue
    gpu_mem = (ctypes.c_uint32 * 2)(0xaaaaaaaa, 0xaaaaaaaa)
    cpu_mem = (ctypes.c_uint32 * 2)()
    cmd_buf = HCQBuffer(ctypes.addressof(gpu_mem), 8, view=MMIOInterface(ctypes.addressof(cpu_mem), 8))
    iface, dev = object.__new__(KGSLIface), Mock(ctx=0, cmd_buf=cmd_buf)
    iface.ctx, dev.iface = 0, iface
    dev.cmd_buf_allocator.alloc.return_value = cmd_buf.va_addr + 4
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678)

    page = queue._build_gpu_command(dev)
    _, obj = iface.prepare_submit(page, 4)
    self.assertEqual(obj.gpuaddr, cmd_buf.va_addr + 4)
    self.assertEqual((tuple(gpu_mem), tuple(cpu_mem)), ((0xaaaaaaaa, 0xaaaaaaaa), (0, 0x12345678)))

    dev.allocator.alloc.return_value = cmd_buf.offset(size=4)
    queue.bind(dev)
    self.assertEqual(queue.hw_page.va_addr, cmd_buf.va_addr)
    queue._q[0] = 0x87654321
    self.assertEqual((tuple(gpu_mem), tuple(cpu_mem)), ((0xaaaaaaaa, 0xaaaaaaaa), (0x87654321, 0x12345678)))

class TestMSMInterface(unittest.TestCase):
  def test_private_allocation_and_submit(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    memory = (ctypes.c_ubyte * mmap.PAGESIZE)()
    fd = Mock()
    fd.mmap.return_value, fd.munmap.return_value = ctypes.addressof(memory), 0
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.allocations = Mock(error_state=None), fd, {}

    def gem_info(_fd, handle, info):
      return Mock(value=0x10000000 if info == msm_drm.MSM_INFO_GET_IOVA else 0)

    with (
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', return_value=Mock(handle=7)) as gem_new,
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=gem_info),
    ):
      buf = iface.alloc(17)
    allocation = buf.meta
    self.assertIsInstance(allocation, MSMAllocation)
    self.assertEqual((allocation.size, allocation.mapped_size), (17, mmap.PAGESIZE))
    self.assertEqual(gem_new.call_args.kwargs['flags'], msm_drm.MSM_BO_WC | msm_drm.MSM_BO_NO_SHARE)

    submit, bos, cmds = iface.prepare_submit(buf.offset(4, 8), 8)
    self.assertEqual((submit.nr_bos, submit.queueid), (1, 0))
    self.assertEqual((bos[0].flags, bos[0].handle, bos[0].presumed), (msm_drm.MSM_SUBMIT_BO_READ, 7, 0x10000000))
    self.assertEqual((cmds[0].submit_offset, cmds[0].size), (4, 8))
    with self.assertRaisesRegex(ValueError, "outside its buffer"): iface.prepare_submit(buf.offset(16, 4), 4)

  def test_fault_and_profile_errors(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.fault_count, iface.sysprof_enabled = Mock(error_state=None), Mock(), 2, False
    with patch.object(iface, '_fault_count', return_value=5), self.assertRaisesRegex(RuntimeError, "2 to 5") as fault:
      iface.on_device_hang()
    self.assertIs(iface.dev.error_state, fault.exception)
    with (
      patch.object(msm_drm, 'DRM_IOCTL_MSM_SET_PARAM', side_effect=OSError(errno.EPERM, 'denied')),
      self.assertRaisesRegex(RuntimeError, 'CAP_SYS_ADMIN'),
    ):
      iface._set_sysprof(2)

  def test_non_msm_node_is_closed(self):
    from tinygrad.runtime.ops_qcom import _open_msm_render_node
    fd = Mock(fd=7)
    def version(_fd, name_len, name):
      ctypes.memmove(name, b"vgem", 4)
      return Mock(name_len=4)
    with (
      patch('tinygrad.runtime.ops_qcom.FileIOInterface', return_value=fd),
      patch.object(msm_drm, 'DRM_IOCTL_VERSION', side_effect=version),
      patch('tinygrad.runtime.ops_qcom.os.close') as close,
    ):
      self.assertIsNone(_open_msm_render_node('/dev/dri/renderD128'))
    close.assert_called_once_with(7)

if __name__ == '__main__':
  unittest.main()
