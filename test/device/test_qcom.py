import ctypes, errno, mmap, platform, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import kgsl, msm_drm
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer, MMIOInterface

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
    self.cpu_addr, self.set_iova_errno, self.wait_errno = ctypes.addressof(self.memory), None, None
    self.next_handle = 17
    self.news, self.unmaps, self.closed_handles = [], [], []
    self.submissions, self.waits = [], []

  def __del__(self): pass

  def ioctl(self, request, arg):
    if request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_NEW):
      self.news.append((arg.size, arg.flags))
      arg.handle, self.next_handle = self.next_handle, self.next_handle + 1
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_INFO):
      if arg.info == msm_drm.MSM_INFO_GET_OFFSET: arg.value = 0x8000
      elif arg.info == msm_drm.MSM_INFO_SET_IOVA and self.set_iova_errno is not None: raise OSError(self.set_iova_errno, "set iova failed")
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
    elif request == ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE): self.closed_handles.append(arg.handle)
    else: raise AssertionError(f"unexpected ioctl {request:#x}")
    return 0

  def mmap(self, start, size, prot, flags, offset):
    return self.cpu_addr

  def munmap(self, addr, size):
    self.unmaps.append((addr, size))
    return 0

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

  def test_msm_submission_includes_referenced_buffers_and_is_reusable(self):
    fd = RecordingMSMFile()
    iface = self.make_msm_iface(fd)
    data = iface.alloc(0x100)
    command = iface.alloc(0x200, fill_zeroes=True).offset(0x40, 0x80)
    prepared = iface.prepare_submit(command, command.size, {data})
    self.assertEqual((iface.submit(prepared), iface.submit(prepared)), (1, 2))

    submit_flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual(fd.news, [(mmap.PAGESIZE, msm_drm.MSM_BO_WC)] * 2)
    self.assertEqual(fd.submissions, [
      (fd.submissions[0][0], fd.submissions[0][1], msm_drm.MSM_PIPE_3D0, 3,
       [(submit_flags, 18, fd.cpu_addr), (submit_flags, 17, fd.cpu_addr)],
       [(msm_drm.MSM_SUBMIT_CMD_BUF, 0, 0x40, 0x80)]),
      (fd.submissions[0][0], fd.submissions[0][1], msm_drm.MSM_PIPE_3D0, 3,
       [(submit_flags, 18, fd.cpu_addr), (submit_flags, 17, fd.cpu_addr)],
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
    fd.set_iova_errno = errno.EBUSY
    with self.assertRaisesRegex(OSError, "set iova failed"): self.make_msm_iface(fd).alloc(17)
    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

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
