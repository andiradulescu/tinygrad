import ctypes, errno, itertools, mmap, platform, struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import kgsl, msm_drm
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer, HWQueue, MMIOInterface

def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number

class TestMSMDRMUAPI(unittest.TestCase):
  def test_struct_layouts_and_ioctl_numbers(self):
    layouts = {
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
      msm_drm.struct_drm_msm_wait_fence: (32, (0, 4, 8, 24)),
      msm_drm.struct_drm_msm_submitqueue: (12, (0, 4, 8)),
    }
    for struct_type, (size, offsets) in layouts.items():
      with self.subTest(struct=struct_type.__name__):
        self.assertEqual(ctypes.sizeof(struct_type), size)
        self.assertEqual(tuple(field[2] for field in struct_type._real_fields_), offsets)

    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE), 0x40086409)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE), 0x40206447)

class HostAllocator:
  def __init__(self): self.memories = []
  def alloc(self, size, _options):
    self.memories.append(memory:=(ctypes.c_ubyte * size)())
    return HCQBuffer(addr:=ctypes.addressof(memory), size, view=MMIOInterface(addr, size))
  def free(self, *_args): pass

class SplitAddressAllocator:
  def __init__(self): self.memories = []
  def alloc(self, size, _options):
    gpu_memory, cpu_memory = (ctypes.c_ubyte * size)(*[0xaa] * size), (ctypes.c_ubyte * size)(*[0xaa] * size)
    self.memories.append((gpu_memory, cpu_memory))
    return HCQBuffer(ctypes.addressof(gpu_memory), size, view=MMIOInterface(ctypes.addressof(cpu_memory), size))
  def free(self, *_args): pass

class RecordingKGSLFile(FileIOInterface):
  def __init__(self): self.commands = []
  def __del__(self): pass
  def ioctl(self, request, arg):
    if request != ioctl_number(kgsl.IOCTL_KGSL_GPU_COMMAND): raise AssertionError(f"unexpected ioctl {request:#x}")
    obj = kgsl.struct_kgsl_command_object.from_address(arg.cmdlist)
    self.commands.append((arg.cmdlist, arg.context_id, arg.cmdsize, obj.gpuaddr, obj.size, obj.flags))
    arg.timestamp = len(self.commands)
    return 0

class RecordingMSMFile(FileIOInterface):
  def __init__(self):
    self.memory = (ctypes.c_ubyte * mmap.PAGESIZE)(*[0xaa] * mmap.PAGESIZE)
    self.cpu_addr, self.get_iova_errno, self.wait_errno, self.close_errno = ctypes.addressof(self.memory), None, None, None
    self.unmap_result = 0
    self.next_handle = 17
    self.news, self.infos, self.unmaps, self.closed_handles = [], [], [], []
    self.submissions, self.waits = [], []

  def __del__(self): pass
  @staticmethod
  def iova(handle): return 0x10000000 + handle * mmap.PAGESIZE

  def ioctl(self, request, arg):
    if request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_NEW):
      self.news.append((arg.size, arg.flags))
      arg.handle, self.next_handle = self.next_handle, self.next_handle + 1
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_INFO):
      self.infos.append((arg.handle, arg.info, arg.value))
      if arg.info == msm_drm.MSM_INFO_GET_OFFSET: arg.value = 0x8000
      elif arg.info == msm_drm.MSM_INFO_GET_IOVA:
        if self.get_iova_errno is not None: raise OSError(self.get_iova_errno, "get iova failed")
        arg.value = self.iova(arg.handle)
      elif arg.info == msm_drm.MSM_INFO_SET_IOVA: raise AssertionError("MSM allocations must use the driver-assigned IOVA")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT):
      bos = (msm_drm.struct_drm_msm_gem_submit_bo * arg.nr_bos).from_address(arg.bos)
      cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * arg.nr_cmds).from_address(arg.cmds)
      self.submissions.append((arg.bos, arg.cmds, arg.flags, arg.queueid,
                               [(bo.flags, bo.handle, bo.presumed) for bo in bos],
                               [(cmd.type, cmd.submit_idx, cmd.submit_offset, cmd.size) for cmd in cmds]))
      arg.fence = len(self.submissions)
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE):
      self.waits.append((arg.fence, arg.timeout.tv_sec, arg.timeout.tv_nsec, arg.queueid))
      if self.wait_errno is not None: raise OSError(self.wait_errno, "wait failed")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE):
      if self.close_errno is not None: raise OSError(self.close_errno, "close failed")
      self.closed_handles.append(arg.handle)
    else: raise AssertionError(f"unexpected ioctl {request:#x}")
    return 0

  def mmap(self, start, size, prot, flags, offset):
    return self.cpu_addr

  def munmap(self, addr, size):
    self.unmaps.append((addr, size))
    return self.unmap_result

class TestQCOMKernelInterfaces(unittest.TestCase):
  @staticmethod
  def make_msm_iface(fd):
    from tinygrad.runtime.ops_qcom import MSMIface
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.queue_id, iface.allocations = SimpleNamespace(last_cmd=0), fd, 3, {}
    return iface

  def test_bound_kgsl_submission_reuses_prepared_request(self):
    from tinygrad.runtime.ops_qcom import KGSLIface, QCOMComputeQueue

    fd, allocator = RecordingKGSLFile(), HostAllocator()
    iface = object.__new__(KGSLIface)
    iface.fd, iface.ctx = fd, 7
    dev = SimpleNamespace(iface=iface, allocator=allocator, last_cmd=0)
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678)
    queue.bind(dev)
    queue.submit(dev).submit(dev)

    self.assertEqual(dev.last_cmd, 2)
    self.assertEqual(fd.commands, [
      (fd.commands[0][0], 7, ctypes.sizeof(kgsl.struct_kgsl_command_object), queue.hw_page.va_addr, 4, kgsl.KGSL_CMDLIST_IB),
      (fd.commands[0][0], 7, ctypes.sizeof(kgsl.struct_kgsl_command_object), queue.hw_page.va_addr, 4, kgsl.KGSL_CMDLIST_IB),
    ])

  def test_bound_queue_writes_commands_through_cpu_mapping(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    allocator = SplitAddressAllocator()
    iface = SimpleNamespace(submit_requires_buffers=False, prepare_submit=lambda *_args: object())
    dev = SimpleNamespace(iface=iface, allocator=allocator, last_cmd=0)
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678, 0x9abcdef0)
    queue.bind(dev)

    gpu_memory, cpu_memory = allocator.memories[0]
    self.assertEqual(list(cpu_memory), list(bytes.fromhex("78563412f0debc9a")))
    self.assertEqual(list(gpu_memory), [0xaa] * 8)

  def test_kernel_arguments_use_cpu_mapping(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState

    gpu_memory, cpu_memory = (ctypes.c_ubyte * 64)(*[0xaa] * 64), (ctypes.c_ubyte * 64)(*[0xaa] * 64)
    args = HCQBuffer(ctypes.addressof(gpu_memory), 64, view=MMIOInterface(ctypes.addressof(cpu_memory), 64))
    data = HCQBuffer(0x123456789abcdef0, 16)
    signature = ((None, 0, dtypes.float32, (1,)), (None, 1, dtypes.uint32, ()))
    prg = SimpleNamespace(kernargs_alloc_size=64, signature=signature, ibo_cnt=0, tex_cnt=0, samp_cnt=0, NIR=True,
                          tex_to_image=[], consts_info=[(0x12345678, 24, 4)], buf_off=8, tex_off=64, ibo_off=64, samplers=[])

    state = QCOMArgsState(args, prg, (data,), vals=(0x87654321,))
    HWQueue().bind_args_state(state)

    self.assertEqual(bytes(cpu_memory[:8]), bytes(8))
    self.assertEqual(int.from_bytes(cpu_memory[8:16], "little"), data.va_addr)
    self.assertEqual(int.from_bytes(cpu_memory[16:20], "little"), 0x87654321)
    self.assertEqual(int.from_bytes(cpu_memory[24:28], "little"), 0x12345678)
    self.assertEqual(bytes(cpu_memory[28:]), bytes(36))
    self.assertEqual(bytes(gpu_memory), bytes([0xaa] * 64))

  def test_program_upload_uses_cpu_mapping(self):
    from tinygrad.runtime.ops_qcom import QCOMProgram

    lib, image, image_offset, image_desc_offset, reg_desc_offset = bytearray(0x500), b"\x12\x34\x56\x78", 0x400, 0x180, 0x300
    struct.pack_into("I", lib, 0x100, len(image))
    struct.pack_into("I", lib, 0xc0, image_offset)
    struct.pack_into("I", lib, 0x110, image_desc_offset)
    struct.pack_into("I", lib, 0x34, reg_desc_offset)
    struct.pack_into("I", lib, reg_desc_offset + 0x14, 1)
    lib[image_offset:image_offset+len(image)] = image

    allocator = SplitAddressAllocator()
    dev = SimpleNamespace(device="QCOM", renderer=object(), allocator=allocator, prof_prg_counter=itertools.count(),
                          _ensure_stack_size=lambda _size: None)
    QCOMProgram(dev, TinyELF(bytes(lib), "test", Target("QCOM"), ()))

    gpu_memory, cpu_memory = allocator.memories[0]
    self.assertEqual(bytes(cpu_memory), image)
    self.assertEqual(bytes(gpu_memory), bytes([0xaa] * len(image)))

  def test_workgroup_size_uses_cpu_mapping(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    gpu_memory, cpu_memory = (ctypes.c_ubyte * 32)(*[0xaa] * 32), (ctypes.c_ubyte * 32)(*[0xaa] * 32)
    args = HCQBuffer(ctypes.addressof(gpu_memory), 32, view=MMIOInterface(ctypes.addressof(cpu_memory), 32))
    prg = SimpleNamespace(NIR=True, wgsz=1, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                          lib_gpu=HCQBuffer(0x200000, 128), pvtmem_size_per_item=0, pvtmem_size_total=0, hw_stack_offset=0,
                          image_size=128, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc)
    dummy = HCQBuffer(0x300000, 4096)
    dev = SimpleNamespace(iface=SimpleNamespace(submit_requires_buffers=False), gpu_id=(6, 0, 0), dummy_buf=dummy, dummy_addr=dummy.va_addr,
                          _stack=HCQBuffer(0x400000, 4096), border_color_buf=HCQBuffer(0x500000, 4096))
    prg.dev = dev

    QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), (1, 1, 1), (2, 3, 4))

    self.assertEqual(bytes(cpu_memory[4:16]), struct.pack("III", 2, 3, 4))
    self.assertEqual(bytes(gpu_memory), bytes([0xaa] * len(gpu_memory)))

  def test_msm_submission_includes_referenced_buffers_and_is_reusable(self):
    fd = RecordingMSMFile()
    iface = self.make_msm_iface(fd)
    data = iface.alloc(0x100)
    command = iface.alloc(0x200, fill_zeroes=True).offset(0x40, 0x80)
    prepared = iface.prepare_submit(command, command.size, {data})
    self.assertEqual((iface.submit(prepared), iface.submit(prepared)), (1, 2))

    submit_flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual((data.va_addr, data.cpu_view().addr), (fd.iova(17), fd.cpu_addr))
    self.assertEqual((command.va_addr, command.cpu_view().addr), (fd.iova(18) + 0x40, fd.cpu_addr + 0x40))
    self.assertEqual(fd.news, [(mmap.PAGESIZE, msm_drm.MSM_BO_WC)] * 2)
    self.assertEqual(fd.submissions, [
      (fd.submissions[0][0], fd.submissions[0][1], msm_drm.MSM_PIPE_3D0, 3,
       [(submit_flags, 18, fd.iova(18)), (submit_flags, 17, fd.iova(17))],
       [(msm_drm.MSM_SUBMIT_CMD_BUF, 0, 0x40, 0x80)]),
      (fd.submissions[0][0], fd.submissions[0][1], msm_drm.MSM_PIPE_3D0, 3,
       [(submit_flags, 18, fd.iova(18)), (submit_flags, 17, fd.iova(17))],
       [(msm_drm.MSM_SUBMIT_CMD_BUF, 0, 0x40, 0x80)]),
    ])
    self.assertEqual(bytes(fd.memory[:0x200]), bytes(0x200))
    iface.free(command)
    iface.free(data)
    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)] * 2)
    self.assertEqual(fd.closed_handles, [18, 17])

  def test_bound_msm_queue_prepares_referenced_buffers_once(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMSignal

    class RecordingIface:
      submit_requires_buffers = True
      def __init__(self): self.prepared, self.submitted = [], []
      def prepare_submit(self, command, size, buffers):
        self.prepared.append((command, size, buffers))
        return object()
      def submit(self, prepared):
        self.submitted.append(prepared)
        return len(self.submitted)

    allocator, iface = HostAllocator(), RecordingIface()
    dummy = allocator.alloc(0x100, None)
    dev = SimpleNamespace(iface=iface, allocator=allocator, dummy_buf=dummy, dummy_addr=dummy.va_addr, gpu_id=(6, 3, 0), last_cmd=0)
    signal = QCOMSignal(base_buf=allocator.alloc(16, None))
    queue = QCOMComputeQueue(dev).signal(signal, 1)
    queue.bind(dev)
    queue.submit(dev).submit(dev)

    self.assertEqual(len(iface.prepared), 1)
    self.assertEqual(iface.prepared[0][2], {dummy, signal.base_buf})
    self.assertEqual(len(iface.submitted), 2)
    self.assertIs(iface.submitted[0], iface.submitted[1])

  def test_msm_allocation_failure_releases_mapping_and_handle(self):
    fd = RecordingMSMFile()
    fd.get_iova_errno = errno.EBUSY
    with self.assertRaisesRegex(OSError, "get iova failed"): self.make_msm_iface(fd).alloc(17)
    self.assertEqual(fd.unmaps, [])
    self.assertEqual(fd.closed_handles, [17])

  def test_msm_free_failure_keeps_allocation_state_consistent(self):
    fd = RecordingMSMFile()
    iface = self.make_msm_iface(fd)
    buf = iface.alloc(17)

    fd.close_errno = errno.EIO
    with self.assertRaisesRegex(RuntimeError, "Failed to close"): iface.free(buf)
    self.assertIs(iface.allocations[buf.meta.handle], buf.meta)
    self.assertEqual(fd.unmaps, [])

    fd.close_errno, fd.unmap_result = None, -1
    with self.assertRaisesRegex(RuntimeError, "Failed to unmap"): iface.free(buf)
    self.assertNotIn(buf.meta.handle, iface.allocations)
    self.assertEqual(fd.closed_handles, [buf.meta.handle])

  def test_msm_fence_wait_tolerates_timeout_and_reports_driver_error(self):
    from tinygrad.runtime.ops_qcom import MSM_WAIT_SLICE_NS

    fd = RecordingMSMFile()
    iface = self.make_msm_iface(fd)
    iface.dev.last_cmd, fd.wait_errno = 41, errno.ETIMEDOUT
    with patch("tinygrad.runtime.ops_qcom.time.monotonic_ns", return_value=5_000_000_123): iface.sleep(0)
    deadline = 5_000_000_123 + MSM_WAIT_SLICE_NS
    self.assertEqual(fd.waits, [(41, deadline // 1_000_000_000, deadline % 1_000_000_000, 3)])

    fd.wait_errno = errno.EIO
    with self.assertRaisesRegex(RuntimeError, "MSM fence wait failed"): iface.sleep(0)

  def test_msm_rejects_external_pointer_mapping(self):
    with self.assertRaisesRegex(RuntimeError, "external pointer"):
      self.make_msm_iface(RecordingMSMFile()).map(0x1000, 0x1000)

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
