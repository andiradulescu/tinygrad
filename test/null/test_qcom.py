import struct, sys, unittest
from tinygrad import dtypes
from tinygrad.device import Buffer
from tinygrad.runtime.support.hcq2 import encode_cmdbuf, hcq_link, ins
from tinygrad.uop.ops import Ops, UOp

@unittest.skipIf(sys.platform == "win32", "QCOM is not supported on Windows")
class TestQCOMCommands(unittest.TestCase):
  def test_timestamp_destination(self):
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    queue = object.__new__(QCOMComputeQueue)
    queue.devs, queue.queue, queue.deps, queue.blob, queue.patches = ("CPU",), "COMPUTE:0", (), bytearray(), []
    signal = Buffer("CPU", 16, preallocate=True)
    commands = UOp(Ops.LINEAR, src=(ins("timestamp", UOp.from_buffer(signal, dtypes.uint64)),))
    cmdbuf = hcq_link(UOp(Ops.LINEAR, src=(encode_cmdbuf(queue, commands),))).src[0].without_after.buffer
    wait, packet, _, dst = struct.unpack("<IIIQ", cmdbuf.host[:])
    self.assertEqual((wait >> 16 & 0x7F, packet >> 16 & 0x7F), (mesa.CP_WAIT_FOR_IDLE, mesa.CP_REG_TO_MEM))
    self.assertEqual(dst, signal.host.addr + dtypes.uint64.itemsize, "timestamp must leave the signal word intact")

if __name__ == "__main__":
  unittest.main()
