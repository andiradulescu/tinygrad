import ctypes, itertools, platform, struct, unittest
from types import SimpleNamespace
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.helpers import mv_address, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import HCQBuffer

class FakeAllocator:
  def __init__(self): self.memories = []
  def alloc(self, size, options):
    self.memories.append(memory:=bytearray(size))
    return HCQBuffer(mv_address(memoryview(memory)), size)
  def free(self, *args): pass

class TestQCOM(unittest.TestCase):
  def test_opencl_program_uses_instruction_groups(self):
    from tinygrad.runtime.ops_qcom import QCOMProgram

    lib, image, image_offset, image_desc_offset, reg_desc_offset = bytearray(0x500), bytes(range(129)), 0x400, 0x180, 0x300
    struct.pack_into("I", lib, 0x100, len(image))
    struct.pack_into("I", lib, 0xc0, image_offset)
    struct.pack_into("I", lib, 0x110, image_desc_offset)
    struct.pack_into("I", lib, 0x34, reg_desc_offset)
    lib[image_offset:image_offset+len(image)] = image

    dev = SimpleNamespace(device="QCOM", renderer=object(), allocator=FakeAllocator(), prof_prg_counter=itertools.count(),
                          _ensure_stack_size=lambda size: None)
    prg = QCOMProgram(dev, TinyELF(bytes(lib), "test", Target("QCOM"), ()))

    self.assertEqual(prg.instrlen, 2)

  def test_ir3_program_uses_compiler_instruction_length(self):
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMProgram, pkt4_hdr

    image = bytes(256)
    v, cs = mesa.struct_ir3_shader_variant(), mesa.struct_ir3_const_state()
    v.info.size, v.instrlen = len(image), 1
    v.cs.work_group_id = v.cs.local_invocation_id = 0xfc
    dummy = HCQBuffer(0x300000, 4096)
    dev = SimpleNamespace(device="QCOM", renderer=object.__new__(IR3Renderer), allocator=FakeAllocator(), prof_prg_counter=itertools.count(),
                          _ensure_stack_size=lambda size: None, gpu_id=(6, 3, 0), dummy_buf=dummy, dummy_addr=dummy.va_addr,
                          _stack=HCQBuffer(0x400000, 4096), border_color_buf=HCQBuffer(0x500000, 4096))
    prg = QCOMProgram(dev, TinyELF(bytes(v) + bytes(cs) + image, "test", Target("QCOM"), ()))

    args = HCQBuffer(0x100000, 4096)
    queue = QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), (1, 1, 1), (1, 1, 1))
    register_packet = pkt4_hdr(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1)

    self.assertEqual(queue._q[queue._q.index(register_packet) + 1], v.instrlen)

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
