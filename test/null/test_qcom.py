import unittest, sys, ctypes, errno, mmap, struct, os, tempfile
from unittest.mock import Mock, patch
from tinygrad.runtime.autogen import msm_drm, dma_buf

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestQCOMCommands(unittest.TestCase):
  def test_timestamp_destination(self):
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    queue = object.__new__(QCOMComputeQueue)
    queue.devs, queue.blob, queue.patches = ("QCOM",), bytearray(), []
    signal = UOp.alloc((2,), dtypes.uint64, device="QCOM")
    queue.timestamp(signal)
    self.assertEqual(len(queue.patches), 1)
    offset, address = queue.patches[0]
    self.assertEqual(offset, 12)
    address = address.substitute({signal.getaddr(("QCOM",)): UOp.const(0x1000, dtypes.uint64)}).simplify()
    self.assertEqual(address.val, 0x1008, "timestamp must leave the signal word intact")

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestKGSLInterface(unittest.TestCase):
  def test_buffer_host_mapping_lifetime(self):
    from tinygrad import Device
    from tinygrad.device import Buffer, BufferSpec
    from tinygrad.runtime.ops_qcom import KGSLIface, QCOMAllocator
    from tinygrad.runtime.autogen import kgsl
    iface = object.__new__(KGSLIface)
    iface.fd = Mock()
    dev = Mock(device="QCOM", iface=iface)
    dev.allocator = QCOMAllocator(dev)
    getitem = type(Device).__getitem__
    src = Buffer("CPU", 16, options=BufferSpec(nolru=True), preallocate=True)
    src.host[:] = data = bytes(range(16))
    with patch.object(type(Device), '__getitem__', lambda obj, key: dev if key == 'QCOM' else getitem(obj, key)), \
         patch('tinygrad.runtime.ops_qcom.dcache_flush', return_value=Mock()), \
         patch.object(kgsl, 'IOCTL_KGSL_MAP_USER_MEM', return_value=Mock(gpuaddr=0x20000000)) as map_mem, \
         patch.object(kgsl, 'IOCTL_KGSL_SHAREDMEM_FREE') as free:
      first = src.get_buf('QCOM')
      self.assertEqual(first, 0x20000000 + (src._buf & 0xfff))
      self.assertEqual(src.get_buf('QCOM'), first)
      self.assertEqual(map_mem.call_count, 1)
      self.assertEqual(bytes(src.as_memoryview()), data)
      src.deallocate()
      free.assert_called_once_with(iface.fd, gpuaddr=0x20000000)
      with self.assertRaisesRegex(RuntimeError, 'Cannot map NULL'):
        dev.allocator.map(Buffer('NULL', 16, preallocate=True))

  def test_mapping_lifetime_and_errors(self):
    from tinygrad.runtime.ops_qcom import KGSLIface
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.support.system import FileIOInterface
    memory = (ctypes.c_ubyte * 8192)(*([37] * 8192))
    ptr = ctypes.addressof(memory) + 19
    iface = object.__new__(KGSLIface)
    iface.fd = Mock()
    with patch('tinygrad.runtime.ops_qcom.dcache_flush', return_value=Mock()), \
         patch.object(kgsl, 'IOCTL_KGSL_MAP_USER_MEM', return_value=Mock(gpuaddr=0x20000000)) as map_mem, \
         patch.object(kgsl, 'IOCTL_KGSL_SHAREDMEM_FREE') as free, patch.object(FileIOInterface, 'munmap') as munmap:
      storage = iface.map(ptr, 80)
      self.assertEqual(bytes(storage.host[:]), bytes([37] * 80))
      self.assertEqual(storage.buf, 0x20000000 + (ptr & 0xfff))
      self.assertEqual(map_mem.call_args.kwargs['hostptr'] % 4096, 0)
      self.assertGreaterEqual(map_mem.call_args.kwargs['len'], (ptr & 0xfff) + 80)
      iface.free(storage)
      free.assert_called_once_with(iface.fd, gpuaddr=0x20000000)
      munmap.assert_not_called()
      map_mem.side_effect = OSError(errno.EFAULT, 'already mapped')
      external = iface.map(ptr, 80)
      self.assertEqual((external.buf, bytes(external.host[:])), (ptr, bytes([37] * 80)))
      iface.free(external)
      self.assertEqual(free.call_count, 1)
      map_mem.side_effect = OSError(errno.EIO, 'mapping failed')
      with self.assertRaisesRegex(RuntimeError, 'Failed to map external pointer') as caught: iface.map(ptr, 80)
      self.assertEqual(caught.exception.__cause__.errno, errno.EIO)

  def test_owned_allocation_release(self):
    from tinygrad.runtime.ops_qcom import KGSLIface
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.support.system import FileIOInterface
    memory = (ctypes.c_ubyte * 4096)()
    iface = object.__new__(KGSLIface)
    iface.fd = Mock()
    iface.fd.mmap.return_value = ctypes.addressof(memory)
    with patch.object(kgsl, 'IOCTL_KGSL_GPUOBJ_ALLOC', return_value=Mock(id=7, mmapsize=4096)), \
         patch.object(kgsl, 'IOCTL_KGSL_GPUOBJ_FREE') as free, patch.object(FileIOInterface, 'munmap') as munmap:
      storage = iface.alloc(19)
      storage.host[:] = bytes(range(19))
      self.assertEqual(bytes(memory[:19]), bytes(range(19)))
      iface.free(storage)
      free.assert_called_once_with(iface.fd, id=7)
      munmap.assert_called_once_with(ctypes.addressof(memory), 4096)

  def test_wait_preserves_signal_and_timeout(self):
    from tinygrad.runtime.ops_qcom import KGSLIface
    from tinygrad.runtime.autogen import kgsl
    iface = object.__new__(KGSLIface)
    iface.fd, iface.ctx, iface.dev = Mock(), 7, Mock(wait_timeout_ms=10000)
    with patch.object(kgsl, 'IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID', return_value=Mock(timestamp=23)) as read, \
         patch.object(kgsl, 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID', side_effect=OSError(errno.EINTR, 'interrupted')) as wait:
      iface.wait_signal([0], 1, timeout=5)
      self.assertEqual((wait.call_args.kwargs['timestamp'], wait.call_args.kwargs['timeout']), (23, 5))
      iface.wait_signal([1], 1)
      self.assertEqual((read.call_count, wait.call_count), (1, 1))

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
  def test_imported_alias_lifetime_and_range(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    from tinygrad.runtime.support.system import FileIOInterface
    iface = object.__new__(MSMIface)
    iface.fd, iface.allocations = Mock(), {}
    iface.fd.munmap.side_effect = FileIOInterface.munmap
    with tempfile.TemporaryFile() as file, patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=7)), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', return_value=Mock(value=0x10000000)), \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close:
      file.truncate(mmap.PAGESIZE)
      file.write(bytes(range(32)))
      file.flush()
      borrowed = os.dup(file.fileno())
      self.assertEqual(iface._allocation_index, [])
      first, second = iface.map(0, 16, borrowed, 8), iface.map(0, 8, borrowed, 16)
      self.assertIs(iface._allocation(second.buf, 8), first.meta)
      os.close(borrowed)
      retained = first.meta.dma_buf_fd
      iface.free(first)
      self.assertEqual(ctypes.string_at(second.meta.cpu_addr + 16, 8), bytes(range(16, 24)))
      self.assertEqual(os.fstat(retained).st_size, mmap.PAGESIZE)
      self.assertEqual(second.buf, 0x10000010)
      close.assert_not_called()
      with self.assertRaises(ValueError): iface.map(0, 16, file.fileno(), mmap.PAGESIZE - 8)
      iface.free(second)
      self.assertEqual(close.call_count, 1)
      with self.assertRaises(OSError): os.fstat(retained)
      self.assertFalse(iface.allocations)
      with self.assertRaisesRegex(RuntimeError, "not allocated"): iface._allocation(second.buf, 8)

  def test_fd_only_buffer_owns_import(self):
    from tinygrad.device import BufferSpec
    from tinygrad.runtime.ops_qcom import MSMIface, QCOMAllocator
    from tinygrad.runtime.support.system import FileIOInterface
    iface = object.__new__(MSMIface)
    iface.fd, iface.allocations = Mock(), {}
    iface.fd.munmap.side_effect = FileIOInterface.munmap
    allocator = QCOMAllocator(Mock(iface=iface))
    with tempfile.TemporaryFile() as file, patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=7)), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', return_value=Mock(value=0x10000000)), \
         patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close, patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC'):
      file.truncate(mmap.PAGESIZE)
      spec = BufferSpec(external_fd=file.fileno(), external_offset=8)
      storage = allocator.alloc(16, spec)
      retained = storage.meta.dma_buf_fd
      allocator._copyin(storage.buf, memoryview(bytearray(b'borrowed storage')))
      file.seek(8)
      self.assertEqual(file.read(16), b'borrowed storage')
      allocator.free(storage, 16, spec)
      close.assert_called_once_with(iface.fd, handle=7)
      self.assertFalse(any(allocator.cache.values()))
      with self.assertRaises(OSError): os.fstat(retained)
      self.assertEqual(os.fstat(file.fileno()).st_size, mmap.PAGESIZE)

  def test_import_cleanup_on_partial_failure(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    from tinygrad.runtime.support.system import FileIOInterface
    for stage in ('dup', 'info', 'mmap'):
      with self.subTest(stage=stage), tempfile.TemporaryFile() as file:
        file.truncate(mmap.PAGESIZE)
        retained = None if stage == 'dup' else os.dup(file.fileno())
        iface = object.__new__(MSMIface)
        iface.fd, iface.allocations = Mock(), {}
        info = Mock(return_value=Mock(value=0x10000000))
        if stage == 'info': info.side_effect = OSError(errno.EIO, stage)
        try:
          with patch.object(msm_drm, 'DRM_IOCTL_PRIME_FD_TO_HANDLE', return_value=Mock(handle=7)), \
               patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close, \
               patch.object(os, 'dup', side_effect=OSError(errno.EIO, stage) if stage == 'dup' else lambda _: retained), \
               patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', info), \
               patch.object(FileIOInterface, '_mmap', side_effect=OSError(errno.EIO, stage) if stage == 'mmap' else Mock()):
            with self.assertRaises(OSError) as caught: iface.map(0, 16, file.fileno())
          self.assertEqual((caught.exception.errno, caught.exception.strerror), (errno.EIO, stage))
          close.assert_called_once_with(iface.fd, handle=7)
          self.assertFalse(iface.allocations)
          if retained is not None:
            with self.assertRaises(OSError): os.fstat(retained)
        finally:
          if retained is not None:
            try: os.close(retained)
            except OSError: pass

  def test_dma_buf_cpu_copy_retries_and_finalizes(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface, QCOMAllocator
    data = (ctypes.c_ubyte * mmap.PAGESIZE)()
    allocation = MSMAllocation(7, 0x10000000, mmap.PAGESIZE, mmap.PAGESIZE, ctypes.addressof(data), 3)
    iface = object.__new__(MSMIface)
    iface.allocations = {allocation.handle: allocation}
    allocator = object.__new__(QCOMAllocator)
    allocator.dev = Mock(iface=iface)
    source, result = bytearray(b'copy bytes'), bytearray(len(b'copy bytes'))
    for copy, args, flags in ((allocator._copyin, (allocation.iova + 8, memoryview(source)), dma_buf.DMA_BUF_SYNC_WRITE),
                              (allocator._copyout, (memoryview(result), allocation.iova + 8), dma_buf.DMA_BUF_SYNC_READ)):
      with self.subTest(flags=flags), patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC', side_effect=[OSError(errno.EINTR, 'start'), None,
                                                                                                  OSError(errno.EAGAIN, 'end'), None]) as sync:
        copy(*args)
        self.assertEqual([call.kwargs['flags'] for call in sync.call_args_list],
                         [flags, flags, flags | dma_buf.DMA_BUF_SYNC_END, flags | dma_buf.DMA_BUF_SYNC_END])
    self.assertEqual(result, source)
    blocked = bytearray(b'not copied')
    with patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC', side_effect=[OSError(errno.EIO, 'start')]) as sync:
      with self.assertRaises(OSError) as caught: allocator._copyin(allocation.iova + 32, memoryview(blocked))
      self.assertEqual((caught.exception.errno, caught.exception.strerror), (errno.EIO, 'start'))
      self.assertEqual(ctypes.string_at(allocation.cpu_addr + 32, len(blocked)), bytes(len(blocked)))
      self.assertEqual(sync.call_count, 1)
    written = bytearray(b'end error')
    with patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC', side_effect=[None, OSError(errno.EIO, 'end')]) as sync:
      with self.assertRaises(OSError) as caught: allocator._copyin(allocation.iova + 64, memoryview(written))
      self.assertEqual((caught.exception.errno, caught.exception.strerror), (errno.EIO, 'end'))
      self.assertEqual(ctypes.string_at(allocation.cpu_addr + 64, len(written)), written)
      self.assertEqual(sync.call_count, 2)
    with patch.object(dma_buf, 'DMA_BUF_IOCTL_SYNC', side_effect=[None, OSError(errno.EINTR, 'end'), None]) as sync:
      with self.assertRaisesRegex(OSError, 'body'):
        with iface.cpu_access(allocation, dma_buf.DMA_BUF_SYNC_WRITE): raise OSError('body')
      self.assertEqual([call.kwargs['flags'] for call in sync.call_args_list], [dma_buf.DMA_BUF_SYNC_WRITE,
                                                                                   dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END,
                                                                                   dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END])

  def test_discovery_and_selection(self):
    from tinygrad import Context
    from tinygrad.runtime.ops_qcom import QCOMDevice, MSMIface
    nodes = {'/dev/dri/renderD128': (b'other', 0x06030002), '/dev/dri/renderD129': (b'msm', 0x07030000),
             '/dev/dri/renderD130': (b'msm', 0x06030002)}
    def open_node(path, flags):
      if path not in nodes: raise FileNotFoundError(path)
      return Mock(path=path)
    def version(fd, name_len, name):
      value = nodes[fd.path][0]
      name[:len(value)] = value
      return Mock(name_len=len(value))
    def param(fd, pipe, param): return Mock(value=nodes[fd.path][1] if param == msm_drm.MSM_PARAM_CHIP_ID else 0)
    with patch('tinygrad.runtime.ops_qcom.glob.glob', return_value=list(nodes)), \
         patch('tinygrad.runtime.ops_qcom.FileIOInterface', side_effect=open_node), \
         patch.object(msm_drm, 'DRM_IOCTL_VERSION', side_effect=version), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GET_PARAM', side_effect=param):
      for target in ('QCOM', 'MSM+QCOM'):
        with self.subTest(target=target), Context(DEV=target):
          dev = object.__new__(QCOMDevice)
          iface = dev._select_iface('QCOM')
          self.assertIsInstance(iface, MSMIface)
          self.assertEqual((iface.fd.path, iface.gpu_id), ('/dev/dri/renderD130', (6, 3, 0)))
      with self.assertRaisesRegex(RuntimeError, 'does not exist'): MSMIface(dev, 1)
      nodes.pop('/dev/dri/renderD130')
      with patch('tinygrad.runtime.ops_qcom.glob.glob', return_value=list(nodes)):
        with self.assertRaisesRegex(RuntimeError, 'No A630'): MSMIface(dev, 0)

  def test_allocation_failure_releases_handle(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    for stage in ('info', 'mmap'):
      with self.subTest(stage=stage):
        iface = object.__new__(MSMIface)
        iface.fd, iface.allocations = Mock(), {}
        iface.fd.mmap.side_effect = OSError(errno.EIO, 'mmap')
        with patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', return_value=Mock(handle=7)), \
             patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=OSError(errno.EIO, 'info') if stage == 'info' else None,
                          return_value=Mock(value=0x10000000)), patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE') as close:
          with self.assertRaises(OSError) as caught: iface.alloc(32)
          self.assertEqual(caught.exception.strerror, stage)
          close.assert_called_once_with(iface.fd, handle=7)
          self.assertFalse(iface.allocations)
          iface.fd.munmap.assert_not_called()

  def test_fence_deadline_and_error_precedence(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd = Mock(wait_timeout_ms=10000, device='QCOM'), Mock()
    iface.submit_error, iface.last_fence = None, 23
    with patch('tinygrad.runtime.ops_qcom.time.monotonic_ns', return_value=1_500_000_000), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_WAIT_FENCE') as wait:
      for timeout, expected in ((0, (1, 500_000_000)), (None, (11, 500_000_000))):
        iface.wait_signal([0], 1, timeout)
        deadline = wait.call_args.kwargs['timeout']
        self.assertEqual((deadline.tv_sec, deadline.tv_nsec), expected)
        self.assertEqual(wait.call_args.kwargs['fence'], 23)
      wait.side_effect = OSError(errno.EIO, 'wait')
      with self.assertRaises(OSError): iface.wait_signal([0], 1)
      iface.submit_error = error = OSError(errno.EINVAL, 'submit')
      with self.assertRaises(OSError) as caught: iface.wait_signal([1], 1)
      self.assertIs(caught.exception, error)
      self.assertEqual(wait.call_count, 3)

  def test_fence_wait_retries_and_timeout(self):
    from tinygrad.runtime.ops_qcom import MSMIface
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd = Mock(wait_timeout_ms=10000, device='QCOM'), Mock()
    iface.submit_error, iface.last_fence = None, 7
    with patch.object(msm_drm, 'DRM_IOCTL_MSM_WAIT_FENCE', side_effect=[OSError(errno.EINTR, 'interrupted'),
         OSError(errno.EAGAIN, 'try again'), OSError(errno.ETIMEDOUT, 'deadline')]) as wait:
      with self.assertRaisesRegex(RuntimeError, 'signal wait timed out'): iface.wait_signal([0], 1, timeout=5)
      self.assertEqual(wait.call_count, 3)
      deadlines = [bytes(c.kwargs['timeout']) for c in wait.call_args_list]
      self.assertEqual(len(set(deadlines)), 1, 'interruption must not extend the deadline')
      wait.reset_mock()
      iface.wait_signal([1], 1)
      wait.assert_not_called()

  def test_allocation_lookup_boundaries(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    iface = object.__new__(MSMIface)
    iface.allocations = {9: MSMAllocation(9, 0x2000, 64, 4096, 0), 7: MSMAllocation(7, 0x1000, 32, 4096, 0)}
    for address, size, handle in ((0x1000, 32, 7), (0x1004, 8, 7), (0x101f, 1, 7), (0x2000, 64, 9), (0x203f, 1, 9)):
      with self.subTest(address=address, size=size): self.assertEqual(iface._allocation(address, size).handle, handle)
    for address, size in ((0xfff, 1), (0x101f, 2), (0x1020, 1), (0x1800, 4), (0x2040, 1)):
      with self.subTest(address=address, size=size), self.assertRaisesRegex(RuntimeError, 'not allocated'):
        iface._allocation(address, size)

  def test_allocation_and_submit(self):
    from tinygrad.runtime.ops_qcom import MSMAllocation, MSMIface
    memory = [(ctypes.c_ubyte * mmap.PAGESIZE)() for _ in range(3)]
    fd = Mock()
    fd.mmap.side_effect, fd.munmap.return_value = [ctypes.addressof(x) for x in memory], 0
    iface = object.__new__(MSMIface)
    iface.dev, iface.fd, iface.allocations = Mock(error_state=None), fd, {}
    iovas = {7:0x10000000, 9:0x20000000, 11:0x30000000}

    def gem_info(_fd, handle, info):
      return Mock(value=iovas[handle] if info == msm_drm.MSM_INFO_GET_IOVA else handle * mmap.PAGESIZE)

    with (
      patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', side_effect=[Mock(handle=7), Mock(handle=9), Mock(handle=11)]) as gem_new,
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
    with patch.object(msm_drm, 'DRM_IOCTL_GEM_CLOSE'), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_NEW', return_value=Mock(handle=11)), \
         patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_INFO', side_effect=gem_info):
      iface.free(data)
      replacement = iface.alloc(32)
      with self.assertRaisesRegex(RuntimeError, "not allocated"): iface.prepare_submit(command.buf, 4, [(data.buf, 4)])
      _, bos, _ = iface.prepare_submit(command.buf, 4, [(replacement.buf, 32)])
      self.assertEqual([bo.handle for bo in bos], [7, 11])

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestMSMReplay(unittest.TestCase):
  def setUp(self):
    from tinygrad import Device
    from tinygrad.helpers import Context
    from tinygrad.runtime.support.system import FileIOInterface
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
    getitem = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), '__getitem__', lambda obj, key: self.dev if key == 'QCOM' else getitem(obj, key)))
    self.submit = self.enterContext(patch.object(msm_drm, 'DRM_IOCTL_MSM_GEM_SUBMIT', side_effect=self.record_submit))

  def test_from_blob_requires_fd(self):
    from tinygrad import Tensor
    from tinygrad.device import Buffer
    self.dev.synchronize()
    cached = Buffer('QCOM', 16, preallocate=True)
    handle = cached.meta.handle
    cached.deallocate()
    cache = {key: list(values) for key, values in self.dev.allocator.cache.items()}
    for ptr in (0, 4096):
      with self.subTest(ptr=ptr):
        allocations = set(self.dev.iface.allocations)
        with self.assertRaisesRegex(ValueError, 'MSM DRM'):
          Tensor.from_blob(ptr, (4,), dtype='uint8', device='QCOM')
        self.assertTrue(all(self.dev.allocator.cache[key] == values for key, values in cache.items()))
        self.assertIn(handle, self.dev.iface.allocations)
        self.assertEqual(set(self.dev.iface.allocations), allocations)
    owned = Tensor.empty(4, dtype='uint8', device='QCOM').realize().uop.buffer.ensure_allocated()
    self.assertIsNone(owned.options.external_ptr)
    self.assertIn(owned.meta.handle, self.dev.iface.allocations)

  def record_submit(self, _fd, **kwargs):
    req = kwargs['__payload']
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * req.nr_bos).from_address(req.bos)
    cmd = msm_drm.struct_drm_msm_gem_submit_cmd.from_address(req.cmds)
    self.assertEqual((req.queueid, req.nr_cmds, req.flags), (0, 1, msm_drm.MSM_PIPE_3D0))
    self.assertEqual(cmd.type, msm_drm.MSM_SUBMIT_CMD_BUF)
    allocation = self.dev.iface.allocations[bos[cmd.submit_idx].handle]
    self.submitted.append(([bo.handle for bo in bos], bytes(self.memory[allocation.handle]),
                           b"".join(bytes(self.memory[bo.handle]) for bo in bos)))
    # This is an ioctl contract test, with no GPU execution. Complete the mocked timeline so host replay can proceed.
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def tearDown(self):
    self.dev.iface.submit_error = None
    self.dev.timeline.host.view(fmt='Q')[0] = self.dev.timeline.host.view(fmt='Q')[1]

  def compile_signal(self, nested=False):
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp, Ops, KernelInfo
    from tinygrad.runtime.support.hcq2 import make_submit, lower_call, HCQInfo, hcq_link, ins
    from tinygrad.engine.realize import lower_and_compile
    buf = UOp.param(0, dtypes.float32, 257, device='QCOM')
    signal = UOp(Ops.LINEAR, src=(buf.getaddr(('QCOM',)),), arg='kernargs') if nested else buf
    submit = make_submit(ins('store', signal, UOp.const(1, dtypes.uint64)), dev='QCOM', queue='COMPUTE:0', deps=(buf,))
    call = UOp.sink(submit, arg=KernelInfo('submit_test')).call(buf, aux=HCQInfo(('QCOM',)))
    return hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lower_call(call),))))

  def test_replacement_views(self, nested=False):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    from test.helpers import call_is_hcq
    linear = self.compile_signal(nested)
    self.assertTrue(any(call_is_hcq(c) for c in linear.src))
    for offset in (0, 68):
      with self.subTest(offset=offset):
        bases = [Buffer('QCOM', 274 * 4, preallocate=True) for _ in range(2)]
        bufs = [b.view(257 * 4, offset).ensure_allocated() for b in bases]
        self.assertNotEqual(bufs[0]._buf, bufs[1]._buf)
        for i in range(4):
          run_linear(linear, input_uops=[UOp.from_buffer(bufs[i % 2], dtypes.float32)], jit=True)
          handles, command, contents = self.submitted[-1]
          if nested: command = contents
          active, other = bufs[i % 2], bufs[1 - i % 2]
          self.assertIn(active.get_storage().meta.handle, handles)
          self.assertNotIn(other.get_storage().meta.handle, handles)
          self.assertIn(struct.pack('<Q', active._buf), command)
          self.assertNotIn(struct.pack('<Q', other._buf), command)

  def test_replacement_views_in_kernargs(self): self.test_replacement_views(nested=True)

  def test_interrupted_submit_replays(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    linear = self.compile_signal()
    buf = Buffer('QCOM', 257 * 4, preallocate=True)
    for error in (errno.EINTR, errno.EAGAIN):
      with self.subTest(error=error):
        self.submit.reset_mock()
        def interrupted(fd, **kwargs):
          if self.submit.call_count == 1: raise OSError(error, 'temporary submit interruption')
          self.record_submit(fd, **kwargs)
        self.submit.side_effect = interrupted
        run_linear(linear, input_uops=[UOp.from_buffer(buf, dtypes.float32)], jit=True)
        self.dev.synchronize()
        self.assertEqual(self.submit.call_count, 2)

  def test_submit_error_propagates(self):
    from tinygrad import dtypes
    from tinygrad.device import Buffer
    from tinygrad.uop.ops import UOp
    from tinygrad.engine.realize import run_linear
    linear = self.compile_signal()
    buf = Buffer('QCOM', 257 * 4, preallocate=True)
    error = OSError(22, 'submit rejected')
    self.submit.side_effect = error
    run_linear(linear, input_uops=[UOp.from_buffer(buf, dtypes.float32)], jit=True)
    with self.assertRaises(OSError) as caught: self.dev.synchronize()
    self.assertIs(caught.exception, error)
    run_linear(linear, input_uops=[UOp.from_buffer(buf, dtypes.float32)], jit=True) # a latched error stops later submissions
    with self.assertRaises(OSError): self.dev.synchronize()
    self.assertEqual(self.submit.call_count, 1)

if __name__ == '__main__': unittest.main()
