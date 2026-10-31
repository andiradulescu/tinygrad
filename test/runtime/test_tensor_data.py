import unittest, struct, mmap, os, ctypes, tempfile, pickle
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

# format types: https://docs.python.org/3/library/struct.html

class TestTensorBytes(unittest.TestCase):
  def test_bytes(self):
    lst = Tensor(bytes(b"\xaa\xbb\xcc\xdd"))
    assert lst.tolist() == [170, 187, 204, 221]

  def test_float_bytes(self):
    lst = Tensor(bytes(struct.pack("ff", 0.234, 0.8585)), dtype=dtypes.float32)
    assert lst.shape == (2,)
    assert abs(lst.tolist()[0] - 0.234) < 1e-6
    assert abs(lst.tolist()[1] - 0.8585) < 1e-6

class TestTensorData(unittest.TestCase):
  def test_from_blob_fd_borrows_cpu_mapping(self):
    from tinygrad import TinyJit
    with tempfile.TemporaryFile() as f:
      f.truncate(mmap.PAGESIZE)
      with mmap.mmap(f.fileno(), mmap.PAGESIZE) as mem:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mem))
        x = Tensor.from_blob(address + 17, (4,), fd=f.fileno(), offset=17, dtype='uint8', device='CPU')
        run = TinyJit(lambda a: (a + 1).realize())
        for value in (16, 64, 128, 235):
          mem[17:21] = bytes([value]) * 4
          self.assertEqual(run(x).tolist(), [value + 1] * 4)
        snapshot = pickle.loads(pickle.dumps(x))
        mem[17:21] = bytes([0]) * 4
        self.assertEqual(snapshot.tolist(), [235] * 4)
        del run, x, snapshot
        self.assertGreaterEqual(os.fstat(f.fileno()).st_size, mmap.PAGESIZE)

  def test_data(self):
    a = Tensor([1,2,3,4], dtype=dtypes.int32)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == [1,2,3,4]
    assert dat.shape == (4,)
    assert dat[0] == 1
    assert dat[1] == 2

  def test_data_empty(self):
    a = Tensor([], dtype=dtypes.int32)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == []
    assert dat.shape == (0,)

  def test_data_empty_multi_dim(self):
    a = Tensor([], dtype=dtypes.int32).reshape(0, 2)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == []
    assert dat.shape == (0,)

  def test_data_uint8(self):
    a = Tensor([1,2,3,4], dtype=dtypes.uint8)
    dat = a.data()
    assert dat.format == "B"
    assert dat.itemsize == 1
    assert dat[0] == 1
    assert dat[1] == 2

  def test_data_nested(self):
    a = Tensor([[1,2],[3,4]], dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == [[1, 2], [3, 4]]
    assert dat.shape == (2,2)
    assert dat[0, 0] == 1
    assert dat[1, 1] == 4

  def test_data_const(self):
    a = Tensor(3, dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == 3
    assert dat.shape == ()

  def test_const_dtype_for_uop(self):
    self.assertEqual(Tensor.const(UOp.const(1.0).cast(dtypes.float32), dtypes.int8).dtype, dtypes.int8)
    self.assertEqual(Tensor.const(UOp.variable("x", 1, 10).bind(5), dtypes.int32).item(), 5)

  def test_data_float32(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float32)
    dat = a.data()
    assert dat.format == "f"
    assert dat[0, 1] == 2.5

  @unittest.skip("requires python 3.12")
  def test_data_float16(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float16)
    dat = a.data()
    assert dat.format == "e"
    assert dat.shape == (2,2)
    # NOTE: python can't deref float16

  def test_tolist_empty_shapes(self):
    for shape, expected in (((0,), []), ((2, 0), [[], []]), ((0, 2), []),
                            ((2, 0, 3), [[], []]), ((2, 3, 0), [[[], [], []], [[], [], []]])):
      self.assertEqual(Tensor.ones(*shape).tolist(), expected)

if __name__ == '__main__':
  unittest.main()