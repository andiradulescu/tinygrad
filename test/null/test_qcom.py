import ctypes, mmap, sys, unittest
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
    iface, dev, lifetime = object.__new__(KGSLIface), Mock(cmd_buf=cmd_buf), Mock()
    iface.ctx = 0
    dev.iface, dev.synchronize, dev.allocator.free = iface, lifetime.synchronize, lifetime.free
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678)

    dev.cmd_buf_allocator.alloc.return_value = cmd_buf.va_addr + 4
    page = queue._build_gpu_command(dev)
    _, obj = iface.prepare_submit(page, 4, set())
    self.assertEqual(obj.gpuaddr, cmd_buf.va_addr + 4)
    self.assertEqual(tuple(gpu_mem), (0xaaaaaaaa, 0xaaaaaaaa))
    self.assertEqual(tuple(cpu_mem), (0, 0x12345678))

    dev.allocator.alloc.return_value = cmd_buf.offset(size=4)
    queue.bind(dev)
    _, obj = queue.prepared_submit
    self.assertEqual(obj.gpuaddr, cmd_buf.va_addr)
    self.assertEqual(tuple(gpu_mem), (0xaaaaaaaa, 0xaaaaaaaa))
    self.assertEqual(tuple(cpu_mem), (0x12345678, 0x12345678))
    queue._q[0] = 0x87654321
    self.assertEqual(tuple(cpu_mem), (0x87654321, 0x12345678))
    del queue
    self.assertEqual([call[0] for call in lifetime.mock_calls], ["synchronize", "free"])

  def test_symbolic_submit_buffer_replacement(self):
    from tinygrad.dtype import dtypes
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface, QCOMComputeQueue
    from tinygrad.uop.ops import UOp
    memory = (ctypes.c_uint32 * 1)()
    allocations = [MSMAllocation(handle, iova, size, size, ctypes.addressof(memory))
                   for handle,iova,size in [(7, 0x10000000, 4), (9, 0x20000000, 32), (11, 0x30000000, 32)]]
    command = HCQBuffer(allocations[0].iova, 4, meta=allocations[0], view=MMIOInterface(ctypes.addressof(memory), 4))
    iface, dev, submitted = object.__new__(MSMIface), Mock(), []
    iface.allocations = {allocation.handle:allocation for allocation in allocations}
    iface.submit = lambda prepared: submitted.append([bo.handle for bo in prepared[1]]) or 0
    dev.iface, dev.allocator.alloc.return_value = iface, command
    address = UOp.variable("address", 0, 0xffffffffffffffff, dtype=dtypes.uint64)
    queue = QCOMComputeQueue(dev)
    queue._add_buffers(HCQBuffer(address, 32))
    queue.q(0x12345678)
    queue.bind(dev)

    queue.submit(dev, {address.expr:0x20000000})
    queue.submit(dev, {address.expr:0x30000000})

    self.assertEqual(submitted, [[7, 9], [7, 11]])

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestMSMInterface(unittest.TestCase):
  def test_allocation_and_submit(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    memory = [(ctypes.c_ubyte * mmap.PAGESIZE)() for _ in range(2)]
    fd = Mock()
    fd.mmap.side_effect, fd.munmap.return_value = [ctypes.addressof(x) for x in memory], 0
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.allocations = Mock(error_state=None), fd, {}
    iovas = {7:0x10000000, 9:0x20000000}

    def gem_info(_fd, handle, info):
      return Mock(value=iovas[handle] if info == msm_drm.MSM_INFO_GET_IOVA else handle * mmap.PAGESIZE)

    with (
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', side_effect=[Mock(handle=7), Mock(handle=9)]) as gem_new,
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=gem_info),
    ):
      command, data = iface.alloc(17), iface.alloc(32)

    self.assertIsInstance(command.meta, MSMAllocation)
    self.assertEqual((command.meta.size, command.meta.mapped_size), (17, mmap.PAGESIZE))
    self.assertEqual([call.kwargs['flags'] for call in gem_new.call_args_list], [msm_drm.MSM_BO_WC, msm_drm.MSM_BO_WC])

    buffers = {data, data.offset(4, 8), HCQBuffer(data.va_addr, data.size)}
    submit, bos, cmds = iface.prepare_submit(command.offset(4, 8), 8, buffers)
    read_write = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual((submit.nr_bos, submit.queueid), (2, 0))
    self.assertEqual([(bo.flags, bo.handle, bo.presumed) for bo in bos], [
      (msm_drm.MSM_SUBMIT_BO_READ, 7, 0x10000000),
      (read_write, 9, 0x20000000),
    ])
    self.assertEqual((cmds[0].submit_idx, cmds[0].submit_offset, cmds[0].size), (0, 4, 8))
    with self.assertRaisesRegex(ValueError, "outside its buffer"): iface.prepare_submit(command.offset(16, 4), 4, set())

if __name__ == '__main__':
  unittest.main()
