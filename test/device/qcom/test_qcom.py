import ctypes, platform, unittest, os, mmap, struct, contextlib
from tinygrad import Device
from tinygrad.renderer.cstyle import ClangRenderer

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush()(ctypes.addressof(buf), 0)

@unittest.skipUnless(Device.DEFAULT == "QCOM" and type(getattr(Device.default, "iface", None)).__name__ == "MSMIface", "MSM DRM required")
class TestMSMDmaBuf(unittest.TestCase):
  def test_import_aliases_survive_original_fd(self):
    import fcntl, pickle
    from tinygrad import Tensor, TinyJit
    from tinygrad.device import Buffer, BufferSpec
    from tinygrad.runtime.autogen import dma_buf
    with contextlib.ExitStack() as stack:
      heap = os.open('/dev/dma_heap/system', os.O_RDONLY | os.O_CLOEXEC)
      stack.callback(os.close, heap)
      request = bytearray(struct.pack('QIIQ', mmap.PAGESIZE, 0, os.O_RDWR | os.O_CLOEXEC, 0))
      fcntl.ioctl(heap, (3 << 30) | (len(request) << 16) | (ord('H') << 8), request)
      fd = struct.unpack('QIIQ', request)[1]
      stack.callback(os.close, fd)
      mem = stack.enter_context(mmap.mmap(fd, mmap.PAGESIZE))
      address = ctypes.addressof(ctypes.c_char.from_buffer(mem))
      first = Tensor.from_blob(address + 17, (257,), fd=fd, offset=17, dtype='uint8', device='QCOM')
      second = Tensor.from_blob(address + 17, (257,), fd=fd, offset=17, dtype='uint8', device='QCOM')
      retained = second.uop.buffer.meta.dma_buf_fd
      del first
      run = TinyJit(lambda x: (x.cast('float32') + 1).realize())
      for value in (16, 64, 128, 235):
        dma_buf.DMA_BUF_IOCTL_SYNC(fd, flags=dma_buf.DMA_BUF_SYNC_WRITE)
        mem[17:274] = bytes([value]) * 257
        dma_buf.DMA_BUF_IOCTL_SYNC(fd, flags=dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END)
        self.assertEqual(second.tolist(), [value] * 257)
        self.assertEqual(run(second).tolist(), [value + 1] * 257)
      # Each import retains its own lifetime; the caller can close its descriptor before a GPU read.
      borrowed = os.dup(fd)
      third = Tensor.from_blob(address + 17, (257,), fd=borrowed, offset=17, dtype='uint8', device='QCOM')
      os.close(borrowed)
      self.assertEqual((third.cast('float32') + 1).tolist(), [236] * 257)
      imported = Buffer('QCOM', 257, options=BufferSpec(external_fd=fd, external_offset=17), preallocate=True)
      self.assertEqual(bytes(imported.as_memoryview()), bytes([235]) * 257)
      imported.deallocate()
      snapshot = pickle.loads(pickle.dumps(second))
      dma_buf.DMA_BUF_IOCTL_SYNC(fd, flags=dma_buf.DMA_BUF_SYNC_WRITE)
      mem[17:274] = bytes([0]) * 257
      dma_buf.DMA_BUF_IOCTL_SYNC(fd, flags=dma_buf.DMA_BUF_SYNC_WRITE | dma_buf.DMA_BUF_SYNC_END)
      self.assertEqual(snapshot.tolist(), [235] * 257)
      del third, second, run, snapshot
      with self.assertRaises(OSError): os.fstat(retained)

if __name__ == '__main__':
  unittest.main()
