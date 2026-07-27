import ctypes, itertools, platform, struct, unittest
from types import SimpleNamespace
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad.helpers import mv_address, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import BumpAllocator, HCQBuffer

class FakeAllocator:
  def __init__(self): self.memories = []
  def alloc(self, size, options):
    self.memories.append(memory:=bytearray(size))
    return HCQBuffer(mv_address(memoryview(memory)), size)
  def free(self, *args): pass

class RecordingIface:
  def __init__(self, submit_requires_buffers):
    self.submit_requires_buffers, self.submissions = submit_requires_buffers, []
  def submit(self, command, size, buffers):
    self.submissions.append((command, size, buffers))
    return 42

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
                          _ensure_stack_size=lambda size: None, iface=SimpleNamespace(submit_requires_buffers=False),
                          gpu_id=(6, 3, 0), dummy_buf=dummy, dummy_addr=dummy.va_addr,
                          _stack=HCQBuffer(0x400000, 4096), border_color_buf=HCQBuffer(0x500000, 4096))
    prg = QCOMProgram(dev, TinyELF(bytes(v) + bytes(cs) + image, "test", Target("QCOM"), ()))

    args = HCQBuffer(0x100000, 4096)
    queue = QCOMComputeQueue(dev).exec(prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), (1, 1, 1), (1, 1, 1))
    register_packet = pkt4_hdr(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1)

    self.assertEqual(queue._q[queue._q.index(register_packet) + 1], v.instrlen)

  def test_queue_tracks_submit_buffers_only_when_required(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    def submit(requires_buffers):
      allocator, iface = FakeAllocator(), RecordingIface(requires_buffers)
      cmd_buf = allocator.alloc(4096, None)
      args = allocator.alloc(32, None)
      data, lib, stack, dummy, signal = [HCQBuffer(addr, 4096) for addr in range(0x100000, 0x600000, 0x100000)]
      dev = SimpleNamespace(iface=iface, allocator=allocator, cmd_buf=cmd_buf,
                            cmd_buf_allocator=BumpAllocator(cmd_buf.size, base=int(cmd_buf.va_addr), wrap=True),
                            gpu_id=(6, 3, 0), _stack=stack, border_color_buf=HCQBuffer(0x600000, 4096),
                            dummy_buf=dummy, dummy_addr=dummy.va_addr)
      prg = SimpleNamespace(dev=dev, NIR=True, wgsz=0xfc, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                            lib_gpu=lib, pvtmem_size_per_item=0, pvtmem_size_total=0, hw_stack_offset=0, image_size=128,
                            instrlen=1, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc)
      state = SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=(data,))
      QCOMComputeQueue(dev).exec(prg, state, (1, 1, 1), (1, 1, 1)).signal(SimpleNamespace(value_addr=signal.va_addr, base_buf=signal), 1).submit(dev)
      return iface.submissions[0][2], {args, data, lib, stack, dummy, signal}

    buffers, expected = submit(True)
    self.assertEqual(buffers, expected)
    self.assertEqual(submit(False)[0], set())

  def test_queue_resolves_symbolic_submit_buffer(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    from tinygrad.uop.ops import UOp

    allocator, iface = FakeAllocator(), RecordingIface(True)
    cmd_buf = allocator.alloc(4096, None)
    dev = SimpleNamespace(iface=iface, allocator=allocator, cmd_buf=cmd_buf,
                          cmd_buf_allocator=BumpAllocator(cmd_buf.size, base=int(cmd_buf.va_addr), wrap=True))
    queue, address = QCOMComputeQueue(dev), UOp.variable("address", 0x1000, 0x2000)
    queue._add_buffers(HCQBuffer(address, 4096))
    queue.q(0)
    queue.submit(dev, {"address": 0x1800})

    self.assertEqual({buf.va_addr for buf in iface.submissions[0][2]}, {0x1800})

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
