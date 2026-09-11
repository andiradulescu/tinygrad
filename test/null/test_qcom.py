import ctypes, mmap, struct, sys, unittest
from unittest.mock import Mock, patch
from tinygrad.runtime.autogen import msm_drm

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

    buffers = [(data.buf, 32), (data.buf + 4, 8), (data.buf, 32)]
    submit, bos, cmds = iface.prepare_submit(command.buf + 4, 8, buffers)
    read_write = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual((submit.nr_bos, submit.queueid), (2, 0))
    self.assertEqual([(bo.flags, bo.handle, bo.presumed) for bo in bos], [
      (msm_drm.MSM_SUBMIT_BO_READ, 7, 0x10000000),
      (read_write, 9, 0x20000000),
    ])
    self.assertEqual((cmds[0].submit_idx, cmds[0].submit_offset, cmds[0].size), (0, 4, 8))
    with self.assertRaisesRegex(RuntimeError, "not allocated"): iface.prepare_submit(command.buf + 16, 4, [])

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestMSMReplay(unittest.TestCase):
  def setUp(self):
    from tinygrad import Device
    from tinygrad.helpers import Context
    from tinygrad.runtime.support.hcq import FileIOInterface
    from tinygrad.runtime.ops_qcom import MSMIface, QCOMDevice
    self.enterContext(Context(DEV="MSM+QCOM:IR3"))
    self.memory, self.submitted = {}, []
    fd = Mock(spec=FileIOInterface)
    fd.ioctl.return_value = 0
    def new(_fd, size, flags):
      handle = len(self.memory) + 1
      self.memory[handle] = (ctypes.c_ubyte * size)()
      return Mock(handle=handle)
    def info(_fd, handle, info): return Mock(value=handle * (1 << 28) if info == msm_drm.MSM_INFO_GET_IOVA else handle * mmap.PAGESIZE)
    fd.mmap.side_effect = lambda _addr, _size, _prot, _flags, offset: ctypes.addressof(self.memory[offset // mmap.PAGESIZE])
    self.enterContext(patch('tinygrad.runtime.ops_qcom.glob.glob', return_value=['/dev/dri/renderD128']))
    self.enterContext(patch('tinygrad.runtime.ops_qcom._open_msm_render_node', return_value=(fd, 0x06030002)))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GET_PARAM', return_value=Mock(value=0)))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', side_effect=new))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=info))
    self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE'))
    self.enterContext(patch.object(QCOMDevice, '_select_iface', lambda dev, _: MSMIface(dev, 0)))
    self.dev = QCOMDevice('QCOM')
    self.dev.rtalloc_size = 1 << 20
    getitem = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), '__getitem__', lambda obj, key: self.dev if key == 'QCOM' else getitem(obj, key)))
    self.submit = self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_SUBMIT', side_effect=self.record_submit))

  def record_submit(self, _fd, **kwargs):
    req = kwargs['__payload']
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * req.nr_bos).from_address(req.bos)
    cmd = msm_drm.struct_drm_msm_gem_submit_cmd.from_address(req.cmds)
    self.assertEqual((req.queueid, req.nr_cmds, req.flags), (0, 1, msm_drm.MSM_PIPE_3D0))
    self.assertEqual(cmd.type, msm_drm.MSM_SUBMIT_CMD_BUF)
    allocation = self.dev.iface.allocations[bos[cmd.submit_idx].handle]
    self.submitted.append(([bo.handle for bo in bos], bytes(self.memory[allocation.handle])))
    # This is an ioctl contract test, with no GPU execution. Complete the mocked timeline so host replay can proceed.
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def tearDown(self):
    self.dev.iface.submit_error = None
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def compile_signal(self):
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp, Ops, KernelInfo
    from tinygrad.runtime.support.hcq2 import make_submit, lower_call, HCQInfo, hcq_link
    from tinygrad.engine.realize import lower_and_compile
    buf = UOp.param(0, dtypes.float32, 257, device='QCOM')
    submit = make_submit(UOp(Ops.INS, arg=('store', dtypes.void), src=(buf, UOp.const(1, dtypes.uint64))),
                         devs=('QCOM',), queue='COMPUTE:0')
    call = UOp.sink(submit, arg=KernelInfo('submit_test')).call(buf, aux=HCQInfo(('QCOM',)))
    return hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lower_call(call),))), allow_cache=False)

  def test_replacement_views(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    from test.helpers import call_is_hcq
    linear = self.compile_signal()
    self.assertTrue(any(call_is_hcq(c) for c in linear.src))
    for offset in (0, 68):
      with self.subTest(offset=offset):
        bases = [Buffer('QCOM', 274, dtypes.float32, preallocate=True) for _ in range(2)]
        bufs = [b.view(257, dtypes.float32, offset).ensure_allocated() for b in bases]
        self.assertNotEqual(bufs[0]._buf, bufs[1]._buf)
        for i in range(4):
          run_linear(linear, input_uops=[UOp.from_buffer(bufs[i % 2])], jit=True)
          handles, command = self.submitted[-1]
          active, other = bufs[i % 2], bufs[1 - i % 2]
          self.assertIn(active.meta.handle, handles)
          self.assertNotIn(other.meta.handle, handles)
          self.assertIn(struct.pack('<Q', active._buf), command)
          self.assertNotIn(struct.pack('<Q', other._buf), command)

  def test_submit_error_propagates(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    linear = self.compile_signal()
    buf = Buffer('QCOM', 257, dtypes.float32, preallocate=True)
    error = OSError(22, 'submit rejected')
    self.submit.side_effect = error
    run_linear(linear, input_uops=[UOp.from_buffer(buf)], jit=True)
    with self.assertRaises(OSError) as caught: self.dev.synchronize()
    self.assertIs(caught.exception, error)
    run_linear(linear, input_uops=[UOp.from_buffer(buf)], jit=True) # a latched error stops later submissions
    with self.assertRaises(OSError): self.dev.synchronize()
    self.assertEqual(self.submit.call_count, 1)

if __name__ == '__main__':
  unittest.main()
