import unittest, sys

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

if __name__ == '__main__': unittest.main()
