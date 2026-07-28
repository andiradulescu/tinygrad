import ctypes, platform, unittest
from types import SimpleNamespace
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

class TestQCOMKernelInterfaces(unittest.TestCase):
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
