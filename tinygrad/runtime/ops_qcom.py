from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, contextlib, glob, errno, time, bisect
from dataclasses import dataclass
from typing import Any, cast
from tinygrad.device import Compiled, BufferStorage, BufferSpec, Buffer, Device, Allocator, TinyELF
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, ccall, cstruct, patch, unwrap_view, layout_args, pack_args, make_program
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.system import FileIOInterface, System
from tinygrad.runtime.autogen import kgsl, mesa, libc, msm_drm, dma_buf
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, round_up, ceildiv, prod, is_image_shape, next_power2, flatten, to_name, PROFILE, IMAGE
from tinygrad.helpers import dedup, to_tuple
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.engine.realize import get_call_kernel_args
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, 1), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg=('__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");', dtypes.void))
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg=('__asm__ volatile("dsb sy" ::: "memory");', dtypes.void)),
                  arg=KernelInfo(name="dcache_flush", opts_to_apply=()))
  prg = to_program(sink, Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.to_elf())

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

@uopfunc
def qcom_submit(req:UOp, ret:UOp, fd:UOp) -> UOp: # the kernel driver takes the command buffer
  idir, base, nr, struct_t = kgsl.IOCTL_KGSL_GPU_COMMAND.args
  ioctl_cmd = (idir << 30) | (ctypes.sizeof(struct_t) << 16) | (base << 8) | nr
  return ret.index(0).store(ccall(libc.dll.ioctl, fd, UOp.const(ioctl_cmd, dtypes.uint32), req.index(0))).sink()

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
def _msm_submit(device_id:int, offset:int, size:int, count:int, refs:int) -> int:
  return cast(QCOMDevice, Device["QCOM" if device_id == 0 else f"QCOM:{device_id}"]).iface._submit(offset, size, count, refs)
# CPUProgram links custom calls through DLL symbols; keep this MSM-only callback alive for linked submissions.
setattr(libc.dll, "msm_submit", _msm_submit)

@uopfunc
def msm_submit_call(refs:UOp, ret:UOp, device_id:UOp, offset:UOp, size:UOp, count:UOp) -> UOp:
  call = UOp.custom_function("msm_submit", dtype=dtypes.int32).call(device_id, offset, size, count, refs.index(0))
  return ret.index(0).store(call).sink()

class QCOMComputeQueue(HWQueue):
  dev:QCOMDevice
  def cmd(self, opcode:int, *vals): self.q(pkt7_hdr(opcode, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def reg(self, reg:int, *vals): self.q(pkt4_hdr(reg, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    # TODO: 7xx support.
    if write_back: # dirty cache write-back, into the device's dummy buffer
      dummy = UOp.alloc((0x1000,), dtypes.uint8, 0, device=self.devs[0], name="dummy")
      self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, dummy.getaddr(self.devs), 0)
    if invalidate: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE) # invalidate cache lines (following reads from RAM).
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self): self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)

  def signal(self, signal:UOp, value:UOp):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gpu_id[:2] < (7, 3):
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), signal.getaddr(self.devs), value.cast(dtypes.uint32))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')

  def timestamp(self, signal:UOp):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True),
             signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))

  def wait(self, signal:UOp, value:UOp):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY), signal.getaddr(self.devs),
             value.cast(dtypes.uint32), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))

  def kernargs(self, call:UOp, prg:UOp, data:QCOMProgramData) -> UOp:
    kargs = list(zip(get_call_kernel_args(call, prg, self.devs), data.signature))
    uavs = [(dt,shape,a) for a,(_,_,dt,shape) in kargs if is_image_shape(shape)]
    # NIR can reorder images to different texture slots
    ibos, texs = uavs[:data.ibo_cnt], [uavs[data.ibo_cnt + (data.tex_to_image[i] if data.NIR else i)] for i in range(data.tex_cnt)]

    args = [(off, UOp.const(val, dtypes.uint32 if sz == 4 else dtypes.uint16)) for val,off,sz in data.consts_info]
    args += layout_args(data.samplers, data.samp_off)
    ubos = [a for a,(_,_,_,shape) in kargs if not is_image_shape(shape)]
    if data.NIR:
      args += layout_args(ubos, data.buf_off)
      if data.wgsz != 0xfc: args += layout_args(list(prg.arg.local_size), data.wgsz * 4)
    else: args += list(zip(data.buf_offs, ubos))

    def _tex(b, ibo=False):
      imgdt, shape, addr = b
      pitch = shape[1] * 4 * imgdt.itemsize
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=shape[1], height=shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=ctz(pitch)-6), 0, addr,
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]
    args += layout_args(flatten(map(_tex, texs)), data.tex_off) + layout_args(flatten(map(functools.partial(_tex, ibo=True), ibos)), data.ibo_off)
    return UOp(Ops.LINEAR, src=tuple(pack_args(args, data.kernargs_alloc_size)), arg="kernargs")

  def exec(self, call:UOp, prg:UOp):
    data, lib = qcom_build_program(self.dev, prg, self.devs)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    if data.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

    args_addr, lib_addr = self.kernargs(call, prg, data).getaddr(self.devs), lib.getaddr(self.devs)
    stack_addr = UOp.alloc((data.hw_stack_offset * 4,), dtypes.uint8, 0, device=self.devs[0], name="stack").getaddr(self.devs)

    self.cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
    self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
    self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x0)
    self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)) # is this right? mesa uses 1
    self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)) # mesa also uses 1
    self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL,
                                                                constant_demotion_enable=data.NIR))
    self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
    self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL))
    self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)

    self.reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
             qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
             global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
             cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0,
             qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=data.hregs, fullregfootprint=data.fregs, branchstack=data.brnchstck),
             qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=data.shared_size), # should this be CONSTLEN_512?
             0, data.prg_offset, lib_addr,
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=data.pvtmem_size_per_item), stack_addr,
             qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=data.pvtmem_size_total))

    # the kernargs sit in the cmdbuf, so the const upload is sized to them (in vec4s) rather than to the whole constlen: it must not read past
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=data.kernargs_alloc_size // 16), args_addr)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=ceildiv(data.image_size, 128)), lib_addr)

    self.reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(data.hw_stack_offset))
    # image_size is in bytes, but INSTR_SIZE is measured in units of instruction groups (16 instructions, 8 bytes each)
    # https://elixir.bootlin.com/mesa/mesa-26.1.5/source/src/freedreno/ir3/ir3_shader.h#L719-L723
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(ceildiv(data.image_size, 128)))

    if data.samp_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=data.samp_cnt), args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE,
               UOp.alloc((0x1000,), dtypes.uint8, 0, device=self.devs[0], name="border_color").getaddr(self.devs))

    if data.tex_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=min(16, data.tex_cnt)), args_addr + data.tex_off)
      self.reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, args_addr + data.tex_off)

    if data.ibo_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_SHADER, num_unit=data.ibo_cnt), args_addr + data.ibo_off)
      self.reg(mesa.REG_A6XX_SP_CS_UAV_BASE, args_addr + data.ibo_off)

    self.reg(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, nsamp=data.samp_cnt, ntex=data.tex_cnt, nuav=data.ibo_cnt))

    if data.NIR:
      self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
               qreg.a6xx_sp_cs_const_config_0(wgidconstid=data.wgid, wgsizeconstid=data.wgsz, wgoffsetconstid=0xfc, localidregid=data.lid),
               qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
      self.cmd(mesa.CP_EXEC_CS, 0,
               qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)

    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)

  def submit(self, cmdbuf:UOp) -> UOp: return self.dev.iface.submit(self, cmdbuf)

class QCOMProgramData:
  def __init__(self, dev:QCOMDevice, obj:TinyELF):
    self.signature, self.name, self.NIR = obj.signature, obj.name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      self.prg_offset, self.brnchstck, self.image_size, self.pvtmem, self.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
      self.tex_to_image = v.image_mapping.tex_to_image[:]
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else: self._parse_lib(obj.lib)

    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total: int = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset: int = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    self.kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size]

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

def qcom_build_program(dev:QCOMDevice, prg:UOp, devs:tuple[str, ...]) -> tuple[QCOMProgramData, UOp]:
  image = bytes((data:=QCOMProgramData(dev, prg.to_elf())).image).ljust(round_up(len(data.image), 4), b"\x00")
  return data, patch(make_program(prg, len(image), devs[0]), [], image)

class QCOMAllocator(Allocator['QCOMDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.external_fd is not None:
      if not isinstance(self.dev.iface, MSMIface): raise ValueError("DMA-BUF import requires MSM DRM")
      return self.dev.iface.map(options.external_ptr or 0, size, options.external_fd, options.external_offset)
    if options.external_ptr is not None and (options.external_ptr or isinstance(self.dev.iface, MSMIface)):
      return self.dev.iface.map(options.external_ptr, size)
    return self.dev.iface.alloc(size)

  def do_free(self, storage:BufferStorage, options:BufferSpec):
    super().do_free(storage, options)
    if options.external_fd is not None and options.external_ptr is not None: self._free(storage, options)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(storage)
  def _map(self, buf:Buffer) -> BufferStorage:
    if buf.device.split(":")[0] not in {"CPU", "PYTHON", "NPY"}: raise RuntimeError(f"Cannot map {buf.device} on {self.dev.device}")
    if buf.options.external_fd is not None and isinstance(self.dev.iface, MSMIface):
      return self.dev.iface.map(buf._buf, buf.nbytes, buf.options.external_fd, buf.options.external_offset)
    return self.dev.iface.map(buf._buf, buf.nbytes)
  def _unmap(self, storage:BufferStorage): self.dev.iface.free(storage)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

  # Only imported MSM storage lacks a host view; other QCOM storage copies through BufferStorage.host.
  def _copyin(self, dest:int, src:memoryview):
    self.dev.synchronize()
    iface = cast(MSMIface, self.dev.iface)
    allocation = iface._allocation(dest, src.nbytes)
    with iface.cpu_access(allocation, dma_buf.DMA_BUF_SYNC_WRITE):
      ctypes.memmove(allocation.cpu_addr + dest - allocation.iova, mv_address(src), src.nbytes)

  def _copyout(self, dest:memoryview, src:int):
    self.dev.synchronize()
    iface = cast(MSMIface, self.dev.iface)
    allocation = iface._allocation(src, dest.nbytes)
    with iface.cpu_access(allocation, dma_buf.DMA_BUF_SYNC_READ):
      ctypes.memmove(mv_address(dest), allocation.cpu_addr + src - allocation.iova, dest.nbytes)

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("qcom_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), lambda s: QCOMComputeQueue(s).encode()),
])

class KGSLIface:
  count = 1
  renderers = [QCOMCLRenderer, IR3Renderer]

  def __init__(self, dev:QCOMDevice, device_id:int):
    if device_id != 0: raise RuntimeError(f"QCOM:{device_id} does not exist (1 KGSL device available)")
    self.dev = dev
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.gpu_id = (info.chip_id >> 24, (info.chip_id >> 16) & 0xFF, (info.chip_id >> 8) & 0xFF)

    # a7xx start with 730x or 'Cxxx', a8xx starts 'Exxx'
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x}")

    if PROFILE and self.gpu_id[:2] < (7, 3):
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    self.var_vals = {"kgsl_fd": self.fd.fd, "kgsl_ctx": self.ctx}

  def alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> BufferStorage:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return BufferStorage(va_addr, (alloc, True), MMIOInterface(va_addr, size, fmt='B'))

  def map(self, ptr:int, size:int) -> BufferStorage:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush()(ptr_line_aligned:=ptr & ~63, ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return BufferStorage(mi.gpuaddr + (ptr - ptr_aligned), (mi, False), MMIOInterface(ptr, size, fmt='B'))
    except OSError as e:
      if e.errno == 14: return BufferStorage(ptr, (None, False), MMIOInterface(ptr, size, fmt='B'))
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def free(self, storage:BufferStorage):
    if storage.meta[0] is None: return # external (gpu) ptr
    if not storage.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=storage.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=storage.meta[0].id)
      FileIOInterface.munmap(storage.buf, storage.meta[0].mmapsize)

  def submit(self, hq:QCOMComputeQueue, cmdbuf:UOp) -> UOp:
    ib, ib_off = unwrap_view(cmdbuf)
    fd, ctxid = [UOp.variable(n, 0, 2**31 - 1, dtypes.int32) for n in ("kgsl_fd", "kgsl_ctx")]
    obj = cstruct(kgsl.struct_kgsl_command_object, gpuaddr=ib.getaddr(hq.devs) + ib_off, size=cmdbuf.max_numel(), flags=kgsl.KGSL_CMDLIST_IB)
    req = cstruct(kgsl.struct_kgsl_gpu_command, cmdlist=obj.getaddr(HCQ_RUNTIME_DEV.device), cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object),
                  numcmds=1, context_id=ctxid)
    return qcom_submit(req.after(cmdbuf), UOp.alloc((1,), dtypes.int32, device=hq.devs[0], name="submit_ret"), fd)

  def wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if sig[0] < value:
      ts = kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, type=kgsl.KGSL_TIMESTAMP_QUEUED).timestamp
      with contextlib.suppress(OSError, RuntimeError):
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=ts, timeout=int(timeout or self.dev.wait_timeout_ms))

  def on_device_hang(self): pass

  def profile_finalize(self):
    with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")

@dataclass
class MSMAllocation:
  handle: int
  iova: int
  size: int
  mapped_size: int
  cpu_addr: int
  dma_buf_fd: int|None = None
  references: int = 1

def _open_msm_render_node(path:str) -> tuple[FileIOInterface, int]|None: # a dropped FileIOInterface closes its fd
  fd = FileIOInterface(path, os.O_RDWR)
  name = (ctypes.c_ubyte * msm_drm.DRM_CLIENT_NAME_MAX_LEN)()
  version = msm_drm.DRM_IOCTL_VERSION(fd, name_len=len(name), name=name)
  if bytes(name[:version.name_len]) != b"msm": return None
  return fd, msm_drm.DRM_IOCTL_MSM_GET_PARAM(fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_CHIP_ID).value

class MSMIface:
  count = 1
  renderers = [IR3Renderer]

  def __init__(self, dev:QCOMDevice, device_id:int):
    if device_id != 0: raise RuntimeError(f"QCOM:{device_id} does not exist (1 MSM DRM device available)")
    self.dev, last_error = dev, None
    for path in sorted(glob.glob("/dev/dri/renderD*")):
      try: node = _open_msm_render_node(path)
      except OSError as e:
        last_error = e
        continue
      if node is None: continue
      fd, chip_id = node
      gpu_id = (chip_id >> 24, (chip_id >> 16) & 0xff, (chip_id >> 8) & 0xff)
      if gpu_id != (6, 3, 0): continue
      self.fd, self.gpu_id = fd, gpu_id
      break
    else:
      if last_error is not None: raise RuntimeError("Failed to open MSM DRM render node") from last_error
      raise RuntimeError("No A630 MSM DRM render node found")

    self.allocations:dict[int, MSMAllocation] = {}
    self.fault_count = self._fault_count()
    self.last_fence = 0
    self.submit_error:Exception|None = None
    self.var_vals = {"msm_device_id": device_id}

  def _fault_count(self) -> int:
    return msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_FAULTS).value

  def alloc(self, size:int, fill_zeroes=False) -> BufferStorage:
    if size <= 0: raise ValueError(f"MSM allocation size must be positive, got {size}")
    mapped_size = round_up(size, mmap.PAGESIZE)
    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.fd, size=mapped_size, flags=msm_drm.MSM_BO_WC)
    cpu_addr = 0
    try:
      iova = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.fd, handle=gem.handle, info=msm_drm.MSM_INFO_GET_IOVA).value
      offset = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.fd, handle=gem.handle, info=msm_drm.MSM_INFO_GET_OFFSET).value
      cpu_addr = self.fd.mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset)
      view = MMIOInterface(cpu_addr, size, fmt='B')
      if fill_zeroes: ctypes.memset(cpu_addr, 0, size)
    except Exception:
      if cpu_addr: self.fd.munmap(cpu_addr, mapped_size)
      with contextlib.suppress(OSError): msm_drm.DRM_IOCTL_GEM_CLOSE(self.fd, handle=gem.handle)
      raise
    allocation = MSMAllocation(gem.handle, iova, size, mapped_size, cpu_addr)
    self.allocations[gem.handle] = allocation
    self.__dict__.pop('_allocation_index', None)
    return BufferStorage(iova, allocation, view)

  def map(self, ptr:int, size:int, fd:int|None=None, offset:int=0) -> BufferStorage:
    if fd is None: raise ValueError("MSM DRM external pointers require a DMA-BUF fd")
    if size <= 0 or offset < 0: raise ValueError("DMA-BUF size must be positive and offset non-negative")
    length = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)
    if offset + size > length: raise ValueError(f"DMA-BUF range [{offset}, {offset + size}) exceeds DMA-BUF size {length}")
    handle = msm_drm.DRM_IOCTL_PRIME_FD_TO_HANDLE(self.fd, fd=fd).handle
    allocation = self.allocations.get(handle)
    if allocation is None:
      retained_fd = None
      try:
        retained_fd = os.dup(fd)
        iova = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.fd, handle=handle, info=msm_drm.MSM_INFO_GET_IOVA).value
        cpu_addr = FileIOInterface._mmap(0, length, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, retained_fd, 0)
      except Exception:
        try: msm_drm.DRM_IOCTL_GEM_CLOSE(self.fd, handle=handle)
        finally:
          if retained_fd is not None: os.close(retained_fd)
        raise
      allocation = self.allocations[handle] = MSMAllocation(handle, iova, length, length, cpu_addr, retained_fd)
      self.__dict__.pop('_allocation_index', None)
    else:
      if allocation.dma_buf_fd is None: allocation.dma_buf_fd = os.dup(fd)
      allocation.references += 1
    return BufferStorage(allocation.iova + offset, allocation)

  @contextlib.contextmanager
  def cpu_access(self, allocation:MSMAllocation, flags:int):
    def sync(flags):
      if allocation.dma_buf_fd is None: return
      while True:
        try:
          dma_buf.DMA_BUF_IOCTL_SYNC(allocation.dma_buf_fd, flags=flags)
          return
        except OSError as e:
          if e.errno not in (errno.EAGAIN, errno.EINTR): raise
    sync(flags)
    try: yield
    finally: sync(flags | dma_buf.DMA_BUF_SYNC_END)

  def free(self, mem:BufferStorage):
    if not isinstance(allocation:=mem.meta, MSMAllocation): raise RuntimeError("MSM buffer was not allocated by the MSM DRM interface")
    if self.allocations.get(allocation.handle) is not allocation: raise RuntimeError(f"MSM GEM handle {allocation.handle} is already freed")
    if allocation.references > 1:
      allocation.references -= 1
      return
    self.fd.munmap(allocation.cpu_addr, allocation.mapped_size)
    msm_drm.DRM_IOCTL_GEM_CLOSE(self.fd, handle=allocation.handle)
    self.allocations.pop(allocation.handle)
    self.__dict__.pop('_allocation_index', None)
    if allocation.dma_buf_fd is not None: os.close(allocation.dma_buf_fd)

  @functools.cached_property
  def _allocation_index(self) -> list[MSMAllocation]: return sorted(self.allocations.values(), key=lambda a: a.iova)

  def _allocation(self, address:int, size:int) -> MSMAllocation:
    allocations = self._allocation_index
    index = bisect.bisect_right(allocations, address, key=lambda a: a.iova) - 1
    if index >= 0 and address + size <= (allocation:=allocations[index]).iova + allocation.size: return allocation
    raise RuntimeError("MSM buffer was not allocated by the MSM DRM interface")

  def prepare_submit(self, command:int, size:int, buffers:list[tuple[int, int]]):
    if size <= 0 or size % 4: raise ValueError(f"MSM command size must be a positive multiple of 4, got {size}")
    allocation = self._allocation(command, size)
    command_offset = command - allocation.iova
    if command_offset % 4: raise ValueError(f"MSM command offset must be a multiple of 4, got {command_offset}")
    if command_offset < 0 or command_offset + size > allocation.size:
      raise ValueError("MSM command range is outside its buffer")

    referenced = {allocation.handle:allocation for allocation in (self._allocation(addr, size) for addr, size in buffers)}
    command_flags = msm_drm.MSM_SUBMIT_BO_READ | (msm_drm.MSM_SUBMIT_BO_WRITE if allocation.handle in referenced else 0)
    referenced.pop(allocation.handle, None)
    allocations = [allocation, *[referenced[handle] for handle in sorted(referenced)]]
    read_write = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
    bos = (msm_drm.struct_drm_msm_gem_submit_bo * len(allocations))()
    for i,(bo,mem) in enumerate(zip(bos, allocations)):
      bo.flags, bo.handle, bo.presumed = command_flags if i == 0 else read_write, mem.handle, mem.iova
    cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)(msm_drm.struct_drm_msm_gem_submit_cmd(
      type=msm_drm.MSM_SUBMIT_CMD_BUF, submit_idx=0, submit_offset=command_offset, size=size))
    submit = msm_drm.struct_drm_msm_gem_submit(flags=msm_drm.MSM_PIPE_3D0, nr_bos=len(bos), nr_cmds=1,
                                               bos=ctypes.addressof(bos), cmds=ctypes.addressof(cmds), queueid=0)
    return submit, bos, cmds

  def submit(self, hq:QCOMComputeQueue, cmdbuf:UOp) -> UOp:
    # The same GETADDRs patch the commands and describe their BOs, including replacement inputs and views.
    buffers = dedup([unwrap_view(cmdbuf)[0], *[unwrap_view(g.src[0])[0] for g in cmdbuf.toposort(enter_calls=True)
                                              if g.op is Ops.GETADDR and to_tuple(g.arg) == hq.devs]])
    refs = UOp.alloc((16 * len(buffers),), dtypes.uint8, device=HCQ_RUNTIME_DEV.device, name="msm_bos")
    refs = patch(refs, [(i * 16 + off, val) for i, b in enumerate(buffers)
                       for off, val in ((0, b.getaddr(hq.devs)), (8, UOp.const(b.nbytes(), dtypes.uint64)))])
    return msm_submit_call(refs.after(cmdbuf), UOp.alloc((1,), dtypes.int32, device=hq.devs[0], name="submit_ret"),
                           UOp.variable("msm_device_id", 0, 2**31 - 1, dtypes.int32),
                           UOp.const(unwrap_view(cmdbuf)[1], dtypes.uint64), UOp.const(cmdbuf.max_numel(), dtypes.uint64),
                           UOp.const(len(buffers), dtypes.uint64))

  def _submit(self, offset:int, size:int, count:int, refs:int) -> int:
    # Exceptions cannot cross a ctypes callback. Latch them, wait_signal raises them on the next wait.
    if self.submit_error is not None: return -1
    try:
      words = (ctypes.c_uint64 * (count * 2)).from_address(refs)
      prepared = self.prepare_submit(words[0] + offset, size, list(zip(words[::2], words[1::2])))
      while True:
        try:
          msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT(self.fd, __payload=prepared[0])
          self.last_fence = prepared[0].fence
          break
        except OSError as e:
          if e.errno not in (errno.EINTR, errno.EAGAIN): raise
      return 0
    except Exception as e:
      self.submit_error = e
      return -1

  def wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if self.submit_error is not None: raise self.submit_error
    if sig[0] >= value: return
    deadline = time.monotonic_ns() + int((timeout if timeout is not None else self.dev.wait_timeout_ms) * 1_000_000)
    wait_until = msm_drm.struct_drm_msm_timespec(tv_sec=deadline // 1_000_000_000, tv_nsec=deadline % 1_000_000_000)
    while True:
      try:
        msm_drm.DRM_IOCTL_MSM_WAIT_FENCE(self.fd, fence=self.last_fence, queueid=0, timeout=wait_until)
        return
      except OSError as e:
        if e.errno == errno.ETIMEDOUT: raise RuntimeError(f"{self.dev.device} signal wait timed out") from e
        if e.errno not in (errno.EINTR, errno.EAGAIN): raise

  def on_device_hang(self):
    fault_count, previous_fault_count = self._fault_count(), self.fault_count
    self.fault_count = fault_count
    if fault_count > previous_fault_count:
      self.submit_error = error = RuntimeError(f"MSM GPU faults increased from {previous_fault_count} to {fault_count}")
      raise error

  def profile_finalize(self): pass

class QCOMDevice(Compiled):
  ifaces = [KGSLIface, MSMIface]
  _stack:Buffer|None
  timestamp_divider = 19.2

  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    self.iface = self._select_iface(device)
    self.gpu_id, self._stack = self.iface.gpu_id, None

    super().__init__(device, QCOMAllocator(self), self.iface.renderers, None,
                     arch=("a%d%d%d" + (",IMAGE_PITCH_ALIGNMENT=64" if IMAGE else "")) % self.gpu_id)

    self.var_vals = self.iface.var_vals

    # the cache flush target, and zeros: the samplers clamp to a black border
    dummy = Buffer(self.device, 0x1000, options=BufferSpec(nolru=True), preallocate=True, allocator=self.allocator)
    border_color = Buffer(self.device, 0x1000, options=BufferSpec(nolru=True), initial_value=bytes(0x1000), allocator=self.allocator)
    stack = {to_name(self.device, "stack"): lambda b: self._ensure_stack_size(b.max_numel())}
    Compiled.link_bufs |= stack | {to_name(self.device, "dummy"): dummy, to_name(self.device, "border_color"): border_color}

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    self.iface.wait_signal(sig, value, timeout)
    super()._wait_signal(sig, value, timeout)

  def on_device_hang(self):
    self.iface.on_device_hang()
    super().on_device_hang()

  def _ensure_stack_size(self, sz:int) -> Buffer: # one stack for the device, grown to the deepest program's private memory
    if self._stack is None or self._stack.nbytes < sz:
      if self._stack is not None: self.synchronize()
      self._stack = Buffer(self.device, sz, options=BufferSpec(nolru=True), preallocate=True)
    return self._stack

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    self.iface.profile_finalize()
