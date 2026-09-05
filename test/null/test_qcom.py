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
    iface, dev = object.__new__(KGSLIface), Mock(ctx=0, cmd_buf=cmd_buf)
    iface.ctx, dev.iface = 0, iface
    dev.cmd_buf_allocator.alloc.return_value = cmd_buf.va_addr + 4
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678)

    page = queue._build_gpu_command(dev)
    _, obj = iface.prepare_submit(page, 4, set())
    self.assertEqual(obj.gpuaddr, cmd_buf.va_addr + 4)
    self.assertEqual((tuple(gpu_mem), tuple(cpu_mem)), ((0xaaaaaaaa, 0xaaaaaaaa), (0, 0x12345678)))

    dev.allocator.alloc.return_value = cmd_buf.offset(size=4)
    queue.bind(dev)
    self.assertEqual(queue.hw_page.va_addr, cmd_buf.va_addr)
    self.assertEqual((tuple(gpu_mem), tuple(cpu_mem)), ((0xaaaaaaaa, 0xaaaaaaaa), (0x12345678, 0x12345678)))
    queue._q[0] = 0x87654321
    self.assertEqual((tuple(gpu_mem), tuple(cpu_mem)), ((0xaaaaaaaa, 0xaaaaaaaa), (0x87654321, 0x12345678)))

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

  def test_submit_buffer_collection(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface, QCOMComputeQueue, QCOMSignal
    memory = [(ctypes.c_ubyte * 4096)() for _ in range(10)]
    allocations = [MSMAllocation(i + 1, 0x10000000 + i * 0x10000, 4096, 4096, ctypes.addressof(mem)) for i,mem in enumerate(memory)]
    bufs = [HCQBuffer(a.iova, a.size, meta=a, view=MMIOInterface(a.cpu_addr, a.size)) for a in allocations]
    command, dummy, wait_buf, timestamp_buf, signal_buf, args_buf, lib, stack, data, border = bufs
    iface, dev, submitted = object.__new__(MSMIface), Mock(), []
    iface.fd, iface.allocations = Mock(), {allocation.handle:allocation for allocation in allocations}
    dev.iface, dev.gpu_id, dev.dummy_buf, dev.dummy_addr = iface, (6, 3, 0), dummy, dummy.va_addr
    dev._stack, dev.border_color_buf, dev.cmd_buf = stack, border, command
    dev.cmd_buf_allocator.alloc.return_value = command.va_addr + 16
    dev.allocator.alloc.side_effect = lambda size, _opts: command.offset(size=size)
    wait_signal, timestamp_signal, signal = [QCOMSignal(buf.offset(size=16)) for buf in (wait_buf, timestamp_buf, signal_buf)]
    prg = Mock(dev=dev, lib_gpu=lib, samp_cnt=1, tex_cnt=0, ibo_cnt=0, NIR=True, hregs=1, fregs=1, brnchstck=0, shared_size=1,
               pvtmem_size_per_item=0, pvtmem_size_total=0, prg_offset=0, hw_stack_offset=0, image_size=128, samp_off=0,
               wgid=0xfc, wgsz=0xfc, lid=0xfc)
    args_state = Mock(buf=args_buf, bufs=(data,), bind_data=[], prg=prg)

    queue = QCOMComputeQueue(dev)
    queue.memory_barrier().wait(wait_signal, 1).timestamp(timestamp_signal).exec(prg, args_state, (1, 1, 1), (1, 1, 1)).signal(signal, 2)
    words = list(queue._q)

    def submit(fd, **kwargs):
      req = kwargs['__payload']
      self.assertIs(fd, iface.fd)
      self.assertEqual((req.nr_cmds, req.queueid, req.flags), (1, 0, msm_drm.MSM_PIPE_3D0))
      bos = (msm_drm.struct_drm_msm_gem_submit_bo * req.nr_bos).from_address(req.bos)
      cmd = msm_drm.struct_drm_msm_gem_submit_cmd.from_address(req.cmds)
      self.assertEqual((cmd.type, cmd.submit_idx), (msm_drm.MSM_SUBMIT_CMD_BUF, 0))
      self.assertEqual(list(command.cpu_view().view(offset=cmd.submit_offset, size=cmd.size, fmt='I')), words)
      submitted.append((cmd.submit_offset, cmd.size, [bo.handle for bo in bos]))
      req.fence = len(submitted)

    with patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_SUBMIT', side_effect=submit):
      queue.submit(dev)
      queue.bind(dev)
      queue.submit(dev)

    self.assertEqual(submitted, [(16, len(words) * 4, list(range(1, 11))), (0, len(words) * 4, list(range(1, 11)))])
    self.assertEqual(dev.last_cmd, 2)

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
