# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_dma_buf_sync(c.Struct):
  SIZE = 8
  flags: int
__u64: TypeAlias = ctypes.c_uint64
struct_dma_buf_sync.register_fields([('flags', ctypes.c_uint64, 0)])
@c.record
class struct_dma_buf_export_sync_file(c.Struct):
  SIZE = 8
  flags: int
  fd: int
__u32: TypeAlias = ctypes.c_uint32
__s32: TypeAlias = ctypes.c_int32
struct_dma_buf_export_sync_file.register_fields([('flags', ctypes.c_uint32, 0), ('fd', ctypes.c_int32, 4)])
@c.record
class struct_dma_buf_import_sync_file(c.Struct):
  SIZE = 8
  flags: int
  fd: int
struct_dma_buf_import_sync_file.register_fields([('flags', ctypes.c_uint32, 0), ('fd', ctypes.c_int32, 4)])
DMA_BUF_SYNC_READ = (1 << 0)
DMA_BUF_SYNC_WRITE = (2 << 0)
DMA_BUF_SYNC_RW = (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
DMA_BUF_SYNC_START = (0 << 2)
DMA_BUF_SYNC_END = (1 << 2)
DMA_BUF_SYNC_VALID_FLAGS_MASK = (DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END)
DMA_BUF_NAME_LEN = 32
DMA_BUF_BASE = 'b'
DMA_BUF_IOCTL_SYNC = _IOW(DMA_BUF_BASE, 0, struct_dma_buf_sync)
DMA_BUF_SET_NAME_A = _IOW(DMA_BUF_BASE, 1, __u32)
DMA_BUF_SET_NAME_B = _IOW(DMA_BUF_BASE, 1, __u64)
DMA_BUF_IOCTL_EXPORT_SYNC_FILE = _IOWR(DMA_BUF_BASE, 2, struct_dma_buf_export_sync_file)
DMA_BUF_IOCTL_IMPORT_SYNC_FILE = _IOW(DMA_BUF_BASE, 3, struct_dma_buf_import_sync_file)