from pathlib import Path

path = Path("tinygrad/runtime/ops_qcom.py")
src = path.read_text()

def replace(old:str, new:str):
  global src
  count = src.count(old)
  assert count == 1, f"expected one match, found {count}: {old.splitlines()[0]!r}"
  src = src.replace(old, new)

replace(
"""  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    super().__init__()
""",
"""  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    self._buffers:dict[HCQBuffer, int|None]|None = {} if dev.iface.submit_requires_buffers else None
    super().__init__()
""")

replace(
"""  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
""",
"""  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _add_buffers(self, *bufs:HCQBuffer):
    if self._buffers is None: return
    for buf in bufs:
      base = buf.base
      if base not in self._buffers: self._buffers[base] = None if isinstance(base.va_addr, int) else self._new_sym(base.va_addr)

  def _resolve_submit_buffers(self) -> set[HCQBuffer]:
    if self._buffers is None: return set()
    buffers:set[HCQBuffer] = set()
    for buf, sym_idx in self._buffers.items():
      if sym_idx is None: buffers.add(buf)
      elif (address:=self._prev_resolved_syms[sym_idx]) is None: raise RuntimeError("QCOM queue has an unresolved symbolic buffer address")
      else: buffers.add(HCQBuffer(address, buf.size))
    return buffers

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
""")

replace(
"""    # TODO: 7xx support.
    if write_back: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(self.dev.dummy_addr), 0) # dirty cache write-back.
""",
"""    # TODO: 7xx support.
    if write_back:
      self._add_buffers(self.dev.dummy_buf)
      self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(self.dev.dummy_addr), 0) # dirty cache write-back.
""")

replace(
"""  def signal(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
""",
"""  def signal(self, signal:QCOMSignal, value=0):
    self._add_buffers(signal.base_buf)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
""")

replace(
"""  def timestamp(self, signal:QCOMSignal):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
""",
"""  def timestamp(self, signal:QCOMSignal):
    self._add_buffers(signal.base_buf)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
""")

replace(
"""  def wait(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),*data64_le(signal.value_addr),
""",
"""  def wait(self, signal:QCOMSignal, value=0):
    self._add_buffers(signal.base_buf)
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),*data64_le(signal.value_addr),
""")

replace(
"""  def bind(self, dev:QCOMDevice):
    self.hw_page = self._build_gpu_command(dev, dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True)))
    self.binded_device = dev
    self.prepared_submit = dev.iface.prepare_submit(self.hw_page, len(self._q) * 4)
    # From now on, the queue is on the device for faster submission.
    self._q = self.hw_page.cpu_view().view(fmt='I')

  def _submit(self, dev:QCOMDevice):
    prepared = self.prepared_submit if self.binded_device == dev else dev.iface.prepare_submit(self._build_gpu_command(dev), len(self._q) * 4)
    dev.last_cmd = dev.iface.submit(prepared)
""",
"""  def bind(self, dev:QCOMDevice):
    self.hw_page = self._build_gpu_command(dev, dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True)))
    self.binded_device = dev
    self.prepared_submit = None if self._buffers is not None else dev.iface.prepare_submit(self.hw_page, len(self._q) * 4, set())
    # From now on, the queue is on the device for faster submission.
    self._q = self.hw_page.cpu_view().view(fmt='I')

  def _submit(self, dev:QCOMDevice):
    if self.binded_device == dev: command, prepared = self.hw_page, self.prepared_submit
    else: command, prepared = self._build_gpu_command(dev), None
    if prepared is None: prepared = dev.iface.prepare_submit(command, len(self._q) * 4, self._resolve_submit_buffers())
    dev.last_cmd = dev.iface.submit(prepared)
""")

replace(
"""  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
""",
"""  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)
    self._add_buffers(args_state.buf, prg.lib_gpu, prg.dev._stack, *args_state.bufs)
    if prg.samp_cnt > 0: self._add_buffers(prg.dev.border_color_buf)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
""")

replace(
"""class KGSLIface:
  count = 1
  renderers = [QCOMCLRenderer, IR3Renderer]
""",
"""class KGSLIface:
  count = 1
  submit_requires_buffers = False
  renderers = [QCOMCLRenderer, IR3Renderer]
""")

replace(
"""      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def prepare_submit(self, command:HCQBuffer, size:int):
    obj = kgsl.struct_kgsl_command_object(gpuaddr=command.va_addr, size=size, flags=kgsl.KGSL_CMDLIST_IB)
""",
"""      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def prepare_submit(self, command:HCQBuffer, size:int, _buffers:set[HCQBuffer]):
    obj = kgsl.struct_kgsl_command_object(gpuaddr=command.va_addr, size=size, flags=kgsl.KGSL_CMDLIST_IB)
""")

replace(
"""class MSMIface:
  count = 1
  renderers = [IR3Renderer]
""",
"""class MSMIface:
  count = 1
  submit_requires_buffers = True
  renderers = [IR3Renderer]
""")

replace(
"""    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.fd, size=mapped_size, flags=msm_drm.MSM_BO_WC | msm_drm.MSM_BO_NO_SHARE)
""",
"""    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.fd, size=mapped_size, flags=msm_drm.MSM_BO_WC)
""")

replace(
"""  def _allocation(self, mem:HCQBuffer) -> MSMAllocation:
    if not isinstance(allocation:=mem.base.meta, MSMAllocation) or self.allocations.get(allocation.handle) is not allocation:
      raise RuntimeError("MSM buffer was not allocated by the MSM DRM interface")
    return allocation

  def prepare_submit(self, command:HCQBuffer, size:int):
    if size <= 0 or size % 4: raise ValueError(f"MSM command size must be a positive multiple of 4, got {size}")
    allocation = self._allocation(command)
    command_offset = int(command.va_addr) - allocation.iova
    if command_offset % 4: raise ValueError(f"MSM command offset must be a multiple of 4, got {command_offset}")
    if command_offset < 0 or size > command.size or command_offset + size > allocation.size:
      raise ValueError("MSM command range is outside its buffer")
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * 1)(msm_drm.struct_drm_msm_gem_submit_bo(
      flags=msm_drm.MSM_SUBMIT_BO_READ, handle=allocation.handle, presumed=allocation.iova))
    cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)(msm_drm.struct_drm_msm_gem_submit_cmd(
      type=msm_drm.MSM_SUBMIT_CMD_BUF, submit_idx=0, submit_offset=command_offset, size=size))
    submit = msm_drm.struct_drm_msm_gem_submit(flags=msm_drm.MSM_PIPE_3D0, nr_bos=1, nr_cmds=1,
                                               bos=ctypes.addressof(bos), cmds=ctypes.addressof(cmds), queueid=0)
    return submit, bos, cmds
""",
"""  def _allocation(self, mem:HCQBuffer) -> MSMAllocation:
    if isinstance(allocation:=mem.base.meta, MSMAllocation):
      if self.allocations.get(allocation.handle) is not allocation: raise RuntimeError(f"MSM GEM handle {allocation.handle} is already freed")
      return allocation
    if not isinstance(mem.va_addr, int): raise RuntimeError("MSM buffer address must be resolved before submission")
    matches = [allocation for allocation in self.allocations.values()
               if allocation.iova <= mem.va_addr and mem.va_addr + mem.size <= allocation.iova + allocation.size]
    if len(matches) != 1: raise RuntimeError("MSM buffer was not allocated by the MSM DRM interface")
    return matches[0]

  def prepare_submit(self, command:HCQBuffer, size:int, buffers:set[HCQBuffer]):
    if size <= 0 or size % 4: raise ValueError(f"MSM command size must be a positive multiple of 4, got {size}")
    allocation = self._allocation(command)
    command_offset = int(command.va_addr) - allocation.iova
    if command_offset % 4: raise ValueError(f"MSM command offset must be a multiple of 4, got {command_offset}")
    if command_offset < 0 or size > command.size or command_offset + size > allocation.size:
      raise ValueError("MSM command range is outside its buffer")

    referenced = {allocation.handle:allocation for allocation in map(self._allocation, buffers)}
    command_flags = msm_drm.MSM_SUBMIT_BO_READ | (msm_drm.MSM_SUBMIT_BO_WRITE if allocation.handle in referenced else 0)
    referenced.pop(allocation.handle, None)
    allocations = [allocation, *[referenced[handle] for handle in sorted(referenced)]]
    read_write = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * len(allocations))(*[
      msm_drm.struct_drm_msm_gem_submit_bo(flags=command_flags if i == 0 else read_write, handle=mem.handle, presumed=mem.iova)
      for i,mem in enumerate(allocations)])
    cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)(msm_drm.struct_drm_msm_gem_submit_cmd(
      type=msm_drm.MSM_SUBMIT_CMD_BUF, submit_idx=0, submit_offset=command_offset, size=size))
    submit = msm_drm.struct_drm_msm_gem_submit(flags=msm_drm.MSM_PIPE_3D0, nr_bos=len(bos), nr_cmds=1,
                                               bos=ctypes.addressof(bos), cmds=ctypes.addressof(cmds), queueid=0)
    return submit, bos, cmds
""")

path.write_text(src)

Path("test/null/test_qcom.py").write_text("""import ctypes, mmap, sys, unittest
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
""")
