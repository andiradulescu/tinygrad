import ctypes, errno, mmap, os, pickle, struct, tempfile, threading, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.device import Buffer
from tinygrad.runtime.autogen import msm_drm
from tinygrad.runtime.support.memory import TLSFAllocator
try:
  import fcntl
  from tinygrad.runtime.ops_qcom import KGSLIface, MSMIface, QCOMAllocator
  from tinygrad.runtime.support.system import FileIOInterface
except ModuleNotFoundError as e:
  if e.name != "fcntl": raise
  raise unittest.SkipTest("QCOM MSM requires fcntl") from e

class DRMFile(FileIOInterface):
  # A local stand-in for the kernel boundary. Imports here use real file descriptors, mmap and polling.
  def __init__(self):
    self.handles, self.bindings, self.fences, self.writers, self.next_handle = {}, {}, [], [], 1
    self.fail_bind = self.fail_fence = False

  def ioctl(self, request, arg):
    if isinstance(arg, msm_drm.struct_drm_prime_handle):
      self.last_import = arg.fd
      stat = os.fstat(arg.fd)
      identity = (stat.st_dev, stat.st_ino)
      if identity not in self.handles.values(): self.handles[self.next_handle], self.next_handle = identity, self.next_handle + 1
      arg.handle = next(handle for handle, key in self.handles.items() if key == identity)
    elif isinstance(arg, msm_drm.struct_drm_gem_close): del self.handles[arg.handle]
    elif isinstance(arg, msm_drm.struct_drm_msm_gem_new):
      self.handles[self.next_handle], arg.handle, self.next_handle = None, self.next_handle, self.next_handle + 1
    elif isinstance(arg, msm_drm.struct_drm_msm_vm_bind):
      op = arg.op
      if op.op == msm_drm.MSM_VM_BIND_OP_MAP:
        if self.fail_bind: raise OSError(errno.EIO, "bind")
        self.bindings[op.iova] = op.range
        read, write = os.pipe()
        if self.fail_fence: os.close(write)
        else:
          self.writers.append(write)
          os.write(write, b"x")
        arg.fence_fd = read
        self.fences.append(read)
      else:
        assert self.bindings.pop(op.iova) == op.range
    elif not isinstance(arg, msm_drm.struct_drm_msm_gem_info): raise AssertionError(f"unexpected ioctl {request:#x}")
    return 0

  def mmap(self, start, size, prot, flags, offset): return self.anon_mmap(start, size, prot, flags | mmap.MAP_ANONYMOUS, offset)

class TestMSMDmaBuf(unittest.TestCase):
  def setUp(self):
    self.drm = DRMFile()
    self.addCleanup(lambda: [os.close(fd) for fd in self.drm.writers])
    self.iface = MSMIface.__new__(MSMIface)
    self.iface.fd, self.iface.bind_queue = self.drm, 2
    self.iface.va_allocator = TLSFAllocator(16*mmap.PAGESIZE, base=0x10000000)
    self.iface._lock, self.iface.dmabufs = threading.RLock(), {}
    dev = SimpleNamespace(device="QCOM", iface=self.iface, synchronize=lambda: None)
    dev.allocator = QCOMAllocator(dev)
    get_device = type(Device).__getitem__
    self.enterContext(patch.object(type(Device), "__getitem__", lambda d, key: dev if key == "QCOM" else get_device(d, key)))
    self.sync_flags, self.sync_errors = [], []
    self.enterContext(patch.object(fcntl, "ioctl", side_effect=self.sync))
    self.file = self.enterContext(tempfile.TemporaryFile())
    self.file.write(bytes(range(128)))
    self.file.truncate(mmap.PAGESIZE + 32)
    self.file.flush()

  def sync(self, fd, request, flags):
    self.assertEqual(request, 0x40086200) # DMA_BUF_IOCTL_SYNC is _IOW('b', 0, struct dma_buf_sync)
    self.assertEqual(os.fstat(fd).st_ino, os.fstat(self.file.fileno()).st_ino)
    self.sync_flags.append(flags.value)
    if self.sync_errors and (error:=self.sync_errors.pop(0)) is not None: raise OSError(error, "sync")
    return 0

  def blob(self, size=8, offset=0, fd=None):
    return Tensor.from_blob(0, (size,), dtype="uint8", device="QCOM", fd=self.file.fileno() if fd is None else fd, offset=offset)

  def test_aliases_and_pickle_snapshot(self):
    borrowed = os.dup(self.file.fileno())
    try: first, alias = self.blob(size=40, offset=32, fd=borrowed), self.blob(size=4, offset=64)
    finally: os.close(borrowed)
    self.assertEqual(len(self.drm.bindings), 1)
    self.assertEqual(list(self.drm.bindings.values()), [2*mmap.PAGESIZE])
    with self.assertRaises(OSError): os.fstat(self.drm.fences[0])
    self.assertEqual(first.tolist(), list(range(32, 72)))
    buffers = []
    payload = pickle.dumps(alias, protocol=5, buffer_callback=buffers.append)
    retained = alias.uop.buffer.meta.fd
    first.uop.buffer.deallocate()
    copied = alias.uop.buffer.as_memoryview(allow_zero_copy=True)
    view = alias.uop.buffer.view(2, 1).ensure_allocated()
    view.copy_from(Buffer("NPY", 2, initial_value=b"xy"))
    self.assertEqual(alias.tolist(), list(b"@xyC"))
    self.assertEqual(os.pread(self.file.fileno(), 4, 64), b"@xyC")
    self.assertEqual(bytes(copied), b"@ABC")
    alias.uop.buffer.deallocate()
    with self.assertRaises(OSError): os.fstat(retained)
    snap = pickle.loads(payload, buffers=buffers)
    self.assertEqual(snap.tolist(), list(b"@ABC"))
    snap.uop.buffer.deallocate()
    self.assertEqual((self.drm.handles, self.drm.bindings), ({}, {}))

  def test_range_and_bind_failure(self):
    with self.assertRaisesRegex(ValueError, "exceeds DMA-BUF size"): self.blob(offset=mmap.PAGESIZE + 32)
    with self.assertRaisesRegex(ValueError, "non-negative"): self.blob(offset=-1)
    with self.assertRaisesRegex(ValueError, "non-empty"): self.blob(size=0)
    with self.assertRaisesRegex(ValueError, "offset requires an fd"):
      Tensor.from_blob(0, (4,), dtype="uint8", device="QCOM", offset=1)
    with self.assertRaisesRegex(ValueError, "require a DMA-BUF fd"):
      Tensor.from_blob(0, (4,), dtype="uint8", device="QCOM")
    with patch.object(Device["QCOM"], "iface", KGSLIface.__new__(KGSLIface)):
      with self.assertRaisesRegex(ValueError, "requires MSM DRM"): self.blob()
    for stage in ("bind", "fence"):
      with self.subTest(stage=stage), patch.object(self.drm, f"fail_{stage}", True):
        with self.assertRaisesRegex(OSError, stage): self.blob()
        with self.assertRaises(OSError): os.fstat(self.drm.last_import)
        self.assertEqual((self.drm.handles, self.drm.bindings, self.iface.dmabufs), ({}, {}, {}))
        self.assertEqual(len(self.iface.va_allocator.blocks), 1)

  def test_gpu_base_alignment(self):
    for dtype, offset in (("float32", 1), ("float32", 4), ("uint8", 17)):
      with self.subTest(dtype=dtype, offset=offset), self.assertRaisesRegex(ValueError, "aligned to 32 bytes"):
        Tensor.from_blob(0, (4,), fd=self.file.fileno(), offset=offset, dtype=dtype, device="QCOM")
    self.assertEqual(self.drm.handles, {})

    values = (1.0, 3.0, 5.0, 7.0)
    os.pwrite(self.file.fileno(), struct.pack("=4f", *values), 32)
    aligned = Tensor.from_blob(0, (4,), fd=self.file.fileno(), offset=32, dtype="float32", device="QCOM")
    self.assertEqual(aligned.tolist(), list(values))
    aligned.uop.buffer.deallocate()

  def test_mmap_failure_keeps_caller_fd(self):
    with tempfile.NamedTemporaryFile() as backing:
      backing.truncate(mmap.PAGESIZE)
      fd = os.open(backing.name, os.O_RDONLY)
      try:
        with self.assertRaises(PermissionError): self.blob(fd=fd)
        self.assertEqual(os.fstat(fd).st_size, mmap.PAGESIZE)
        with self.assertRaises(OSError): os.fstat(self.drm.last_import)
      finally: os.close(fd)
    self.assertEqual((self.drm.handles, self.drm.bindings), ({}, {}))

  def test_sync_retries_and_failure(self):
    blob = self.blob(size=2)
    self.sync_errors = [errno.EINTR, None, errno.EAGAIN, None]
    self.assertEqual(blob.tolist(), [0, 1])
    self.assertEqual(self.sync_flags, [1, 1, 5, 5])
    src = Buffer("NPY", 2, initial_value=b"ab")
    self.sync_errors = [errno.EIO]
    with self.assertRaisesRegex(OSError, "sync"): blob.uop.buffer.copy_from(src)
    self.assertEqual(os.pread(self.file.fileno(), 2, 0), b"\x00\x01")
    self.assertEqual(self.sync_flags[-1:], [2])
    self.sync_errors = [None, errno.EIO]
    with self.assertRaisesRegex(OSError, "sync"): blob.uop.buffer.copy_from(src)
    self.assertEqual(self.sync_flags[-2:], [2, 6])
    blob.uop.buffer.deallocate()

  def test_non_qcom_uses_pointer(self):
    buf = (ctypes.c_ubyte * 4)(7, 8, 9, 10)
    blob = Tensor.from_blob(ctypes.addressof(buf) + 1, (2,), fd=self.file.fileno(), offset=1, dtype="uint8", device="NPY")
    self.assertEqual(blob.tolist(), [8, 9])
    buf[1] = 77
    self.assertEqual(blob.tolist(), [77, 9])
