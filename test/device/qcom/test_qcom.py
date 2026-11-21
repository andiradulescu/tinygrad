import ctypes, mmap, os, platform, struct, sys, unittest
from tinygrad import Device, Tensor
from tinygrad.renderer.cstyle import ClangRenderer

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush()(ctypes.addressof(buf), 0)

  @unittest.skipUnless(sys.platform == "linux", "Linux DMA heap required")
  def test_msm_dmabuf_alias(self):
    import fcntl
    from tinygrad.runtime.ops_qcom import MSMIface, DMA_BUF_IOCTL_SYNC, DMA_BUF_SYNC_WRITE, DMA_BUF_SYNC_END
    if Device.DEFAULT != "QCOM" or not isinstance(getattr(Device.default, "iface", None), MSMIface):
      self.skipTest("QCOM MSM DRM required")
    if not os.path.exists(heap_path:="/dev/dma_heap/system"): self.skipTest("system DMA heap required")

    heap = os.open(heap_path, os.O_RDONLY | os.O_CLOEXEC)
    try:
      # include/uapi/linux/dma-heap.h: _IOWR('H', 0, struct dma_heap_allocation_data)
      request = bytearray(struct.pack("=QIIQ", mmap.PAGESIZE, 0, os.O_RDWR | os.O_CLOEXEC, 0))
      fcntl.ioctl(heap, (3 << 30) | (len(request) << 16) | (ord('H') << 8), request)
      _, fd, _, _ = struct.unpack("=QIIQ", request)
    finally: os.close(heap)

    values = tuple(range(64))
    data = struct.pack("=64f", *values)
    try:
      with mmap.mmap(fd, mmap.PAGESIZE) as mem:
        DMA_BUF_IOCTL_SYNC(fd, DMA_BUF_SYNC_WRITE)
        try: mem[32:32+len(data)] = data
        finally: DMA_BUF_IOCTL_SYNC(fd, DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_END)
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
        duplicate = os.dup(fd)
        try: alias = Tensor.from_blob(ptr + 64, (16,), fd=duplicate, offset=64, dtype="uint8", device="QCOM")
        finally: os.close(duplicate)
      first = Tensor.from_blob(0, (len(values),), fd=fd, offset=32, dtype="float32", device="QCOM")
    finally: os.close(fd)

    self.assertEqual((first * 2 + 1).tolist(), [v * 2 + 1 for v in values])
    first.uop.buffer.deallocate()
    self.assertEqual((alias.cast("int32") + 1).tolist(), [v + 1 for v in data[32:48]])
    self.assertEqual(alias.tolist(), list(data[32:48]))

if __name__ == '__main__':
  unittest.main()
