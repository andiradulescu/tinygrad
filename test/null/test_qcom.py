import unittest, sys, ctypes, errno
from unittest.mock import Mock, patch

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

if __name__ == '__main__': unittest.main()
