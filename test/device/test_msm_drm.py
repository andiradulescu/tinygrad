import ctypes, errno, mmap, unittest
from types import SimpleNamespace
from unittest.mock import patch

import tinygrad.runtime.autogen as autogen
from tinygrad.helpers import mv_address
from tinygrad.runtime.autogen import msm_drm
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer


def ioctl_number(ioctl):
  direction, base, number, struct_type = ioctl.args
  return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number


class RecordingMSMFile(FileIOInterface):
  def __init__(self):
    self.memory = bytearray([0xaa] * mmap.PAGESIZE)
    self.cpu_addr = mv_address(memoryview(self.memory))
    self.news, self.infos, self.mmaps, self.unmaps, self.closed_handles = [], [], [], [], []
    self.new_queues, self.submissions, self.waits, self.closed_queues, self.imports = [], [], [], [], []
    self.is_msm, self.chip_id, self.import_handle = True, 0x06030000, 19
    self.set_iova_errno, self.get_iova_errno, self.wait_errno, self.close_errno, self.munmap_result = None, None, None, None, 0

  def __del__(self): pass

  def ioctl(self, request, arg):
    if request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_NEW):
      self.news.append((arg.size, arg.flags))
      arg.handle = 17
    elif request == ioctl_number(msm_drm.DRM_IOCTL_PRIME_FD_TO_HANDLE):
      self.imports.append((arg.fd, arg.flags))
      arg.handle = self.import_handle
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_INFO):
      self.infos.append((arg.handle, arg.info, arg.value))
      if arg.info == msm_drm.MSM_INFO_GET_OFFSET: arg.value = 0x8000
      elif arg.info == msm_drm.MSM_INFO_SET_IOVA and self.set_iova_errno is not None: raise OSError(self.set_iova_errno, "set iova failed")
      elif arg.info == msm_drm.MSM_INFO_GET_IOVA:
        if self.get_iova_errno is not None: raise OSError(self.get_iova_errno, "get iova failed")
        arg.value = 0x1234_0000
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM):
      if not self.is_msm: raise OSError(errno.ENOTTY, "not msm")
      arg.value = 630 if arg.param == msm_drm.MSM_PARAM_GPU_ID else self.chip_id
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW):
      self.new_queues.append((arg.flags, arg.prio))
      arg.id = 3
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT):
      bos_type = msm_drm.struct_drm_msm_gem_submit_bo * arg.nr_bos
      cmds_type = msm_drm.struct_drm_msm_gem_submit_cmd * arg.nr_cmds
      bos = [(bo.flags, bo.handle, bo.presumed) for bo in bos_type.from_address(arg.bos)]
      cmds = [(cmd.type, cmd.submit_idx, cmd.submit_offset, cmd.size) for cmd in cmds_type.from_address(arg.cmds)]
      self.submissions.append((arg.flags, arg.queueid, bos, cmds))
      arg.fence = 42
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE):
      self.waits.append((arg.fence, arg.flags, arg.timeout.tv_sec, arg.timeout.tv_nsec, arg.queueid))
      if self.wait_errno is not None: raise OSError(self.wait_errno, "wait failed")
    elif request == ioctl_number(msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE): self.closed_queues.append(arg.value)
    elif request == ioctl_number(msm_drm.DRM_IOCTL_GEM_CLOSE):
      self.closed_handles.append(arg.handle)
      if self.close_errno is not None: raise OSError(self.close_errno, "close failed")
    return 0

  def mmap(self, start, size, prot, flags, offset):
    self.mmaps.append((start, size, prot, flags, offset))
    return self.cpu_addr

  def munmap(self, addr, size):
    self.unmaps.append((addr, size))
    return self.munmap_result


def make_iface(fd):
  from tinygrad.runtime.ops_qcom import MSMIface

  iface = object.__new__(MSMIface)
  iface.dev, iface.fd, iface.queue_id, iface.allocations = SimpleNamespace(last_cmd=0), fd, 3, {}
  return iface


def make_buffer(handle, iova, size):
  from tinygrad.runtime.ops_qcom import MSMAllocation

  return HCQBuffer(iova, size, meta=MSMAllocation(handle, iova, size, None))


class TestMSMDRMUAPI(unittest.TestCase):
  def test_autogen_targets_aarch64(self):
    with patch.object(autogen, "load") as load:
      autogen.__getattr__("msm_drm")

    args = load.call_args.kwargs["args"]
    self.assertIn("--target=aarch64-linux-gnu", args)
    self.assertIn("-I{}/usr/include/aarch64-linux-gnu", args)

  def test_struct_layouts(self):
    layouts = {
      msm_drm.struct_drm_msm_timespec: (16, (0, 8)),
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
      msm_drm.struct_drm_msm_wait_fence: (32, (0, 4, 8, 24)),
      msm_drm.struct_drm_msm_submitqueue: (12, (0, 4, 8)),
      msm_drm.struct_drm_prime_handle: (12, (0, 4, 8)),
    }
    for struct_type, (size, offsets) in layouts.items():
      with self.subTest(struct=struct_type.__name__):
        self.assertEqual(struct_type.SIZE, size)
        self.assertEqual(ctypes.sizeof(struct_type), size)
        self.assertEqual(tuple(field[2] for field in struct_type._real_fields_), offsets)

  def test_ioctl_numbers_include_linux_struct_sizes(self):
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE), 0x40206447)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_PRIME_FD_TO_HANDLE), 0xC00C642E)


class TestMSMIface(unittest.TestCase):
  def test_init_selects_msm_render_node_and_a630(self):
    bad, good = RecordingMSMFile(), RecordingMSMFile()
    bad.is_msm = False
    with patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD129", "/dev/dri/renderD128"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", side_effect=[bad, good]):
      from tinygrad.runtime.ops_qcom import MSMIface
      iface = MSMIface(SimpleNamespace(), 0)

    self.assertIs(iface.fd, good)
    self.assertEqual((iface.chip_id, iface.gpu_id, iface.queue_id), (0x06030000, (6, 3, 0), 3))
    self.assertEqual(good.new_queues, [(0, 0)])

  def test_init_rejects_other_devices_and_gpu_generations(self):
    from tinygrad.runtime.ops_qcom import MSMIface

    with self.assertRaisesRegex(RuntimeError, "QCOM:1 does not exist"): MSMIface(SimpleNamespace(), 1)

    fd = RecordingMSMFile()
    fd.chip_id = 0x06040000
    with patch("tinygrad.runtime.ops_qcom.glob.glob", return_value=["/dev/dri/renderD128"]), \
         patch("tinygrad.runtime.ops_qcom.FileIOInterface", return_value=fd):
      with self.assertRaisesRegex(RuntimeError, "requires Adreno 630"): MSMIface(SimpleNamespace(), 0)

  def test_alloc_uses_cpu_mapping_as_iova(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    buf = iface.alloc(17, fill_zeroes=True)

    self.assertEqual((buf.va_addr, buf.cpu_view().addr, buf.size), (fd.cpu_addr, fd.cpu_addr, 17))
    self.assertEqual(fd.news, [(mmap.PAGESIZE, msm_drm.MSM_BO_WC)])
    self.assertEqual(fd.mmaps, [(0, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0x8000)])
    self.assertIn((17, msm_drm.MSM_INFO_SET_IOVA, fd.cpu_addr), fd.infos)
    self.assertEqual(fd.memory[:17], bytes(17))

    iface.free(buf)
    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])
    self.assertEqual(iface.allocations, {})

  def test_alloc_cleans_up_if_iova_assignment_fails(self):
    fd = RecordingMSMFile()
    fd.set_iova_errno = errno.EBUSY

    with self.assertRaisesRegex(OSError, "set iova failed"): make_iface(fd).alloc(17)

    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

  def test_import_rejects_missing_fd_and_invalid_ranges(self):
    iface = make_iface(RecordingMSMFile())
    with self.assertRaisesRegex(ValueError, "DMA-BUF fd"): iface.map(0x1000, 16)
    with self.assertRaisesRegex(ValueError, "non-negative"): iface.map(0x1000, 16, 9, -1)
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=0x100)):
      with self.assertRaisesRegex(ValueError, "exceeds DMA-BUF size"): iface.map(0x1000, 0x20, 9, 0xf0)

  def test_import_preserves_offset_and_submits_base_iova(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    command = make_buffer(11, 0x1000_0000, 0x1000)
    iface.allocations[11] = command.meta
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=mmap.PAGESIZE)):
      data = iface.map(fd.cpu_addr + 0x40, 17, 9, 0x40)

    self.assertEqual((data.va_addr, data.cpu_view().addr, data.size), (0x1234_0040, fd.cpu_addr + 0x40, 17))
    self.assertEqual(fd.imports, [(9, 0)])
    iface.submit(command, 4, {data})
    flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertIn((flags, 19, 0x1234_0000), fd.submissions[0][2])

    iface.free(data)
    self.assertEqual(fd.unmaps, [])
    self.assertEqual(fd.closed_handles, [19])

  def test_repeated_import_closes_once_after_last_free(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    command = make_buffer(11, 0x1000_0000, 0x1000)
    iface.allocations[11] = command.meta
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=mmap.PAGESIZE)):
      first = iface.map(fd.cpu_addr, 17, 9)
      second = iface.map(fd.cpu_addr + 0x40, 17, 9, 0x40)

    self.assertIs(first.meta, second.meta)
    self.assertEqual(first.meta.references, 2)
    self.assertEqual(fd.imports, [(9, 0), (9, 0)])
    self.assertEqual(sum(info == msm_drm.MSM_INFO_GET_IOVA for _,info,_ in fd.infos), 1)
    iface.submit(command, 4, {first, second})
    self.assertEqual([handle for _,handle,_ in fd.submissions[0][2]].count(19), 1)

    iface.free(first)
    self.assertEqual(fd.closed_handles, [])
    self.assertIs(iface.allocations[19], second.meta)
    iface.free(second)
    self.assertEqual(fd.closed_handles, [19])
    self.assertNotIn(19, iface.allocations)

  def test_reimported_owned_allocation_retains_cpu_mapping(self):
    fd = RecordingMSMFile()
    iface, fd.import_handle = make_iface(fd), 17
    original = iface.alloc(17)
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=mmap.PAGESIZE)):
      imported = iface.map(fd.cpu_addr + 0x40, 17, 9, 0x40)

    iface.free(original)
    self.assertEqual(fd.unmaps, [])
    iface.free(imported)
    self.assertEqual(fd.unmaps, [(fd.cpu_addr, mmap.PAGESIZE)])
    self.assertEqual(fd.closed_handles, [17])

  def test_import_closes_handle_if_iova_lookup_fails(self):
    fd = RecordingMSMFile()
    iface, fd.get_iova_errno = make_iface(fd), errno.EIO
    with patch("tinygrad.runtime.ops_qcom.os.fstat", return_value=SimpleNamespace(st_size=mmap.PAGESIZE)):
      with self.assertRaisesRegex(OSError, "get iova failed"): iface.map(fd.cpu_addr, 17, 9)
    self.assertEqual(fd.closed_handles, [19])

  def test_free_can_retry_unmap_and_close_failures(self):
    for failure in ("unmap", "close"):
      with self.subTest(failure=failure):
        fd = RecordingMSMFile()
        iface = make_iface(fd)
        buf = iface.alloc(17)
        if failure == "unmap": fd.munmap_result = -1
        else: fd.close_errno = errno.EIO

        with self.assertRaisesRegex(RuntimeError, f"Failed to {failure}"): iface.free(buf)
        self.assertIs(iface.allocations[17], buf.meta)

        fd.munmap_result, fd.close_errno = 0, None
        iface.free(buf)
        self.assertNotIn(17, iface.allocations)

  def test_submit_uses_referenced_bos_and_command_offset(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)
    command_base, data = make_buffer(11, 0x1000_0000, 0x1000), make_buffer(12, 0x2000_0000, 0x2000)
    iface.allocations = {11: command_base.meta, 12: data.meta}

    fence = iface.submit(command_base.offset(0x40, 0x80), 0x80, {data, data.offset(0x100, 0x100)})

    self.assertEqual(fence, 42)
    flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    self.assertEqual(fd.submissions, [(msm_drm.MSM_PIPE_3D0, 3, [(flags, 11, 0x1000_0000), (flags, 12, 0x2000_0000)],
                                       [(msm_drm.MSM_SUBMIT_CMD_BUF, 0, 0x40, 0x80)])])

  def test_wait_uses_absolute_deadline_and_reports_driver_errors(self):
    from tinygrad.runtime.ops_qcom import MSM_WAIT_SLICE_NS

    fd = RecordingMSMFile()
    iface = make_iface(fd)
    iface.dev.last_cmd, fd.wait_errno = 41, errno.ETIMEDOUT
    with patch("tinygrad.runtime.ops_qcom.time.monotonic_ns", return_value=5_000_000_123): iface.sleep(0)

    deadline = 5_000_000_123 + MSM_WAIT_SLICE_NS
    self.assertEqual(fd.waits, [(41, 0, deadline // 1_000_000_000, deadline % 1_000_000_000, 3)])

    fd.wait_errno = errno.EIO
    with self.assertRaisesRegex(RuntimeError, "MSM fence wait failed"): iface.sleep(0)

  def test_device_fini_closes_submitqueue_once(self):
    fd = RecordingMSMFile()
    iface = make_iface(fd)

    iface.device_fini()
    iface.device_fini()

    self.assertEqual(fd.closed_queues, [3])


if __name__ == "__main__":
  unittest.main()
