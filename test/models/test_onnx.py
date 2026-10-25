#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.device import Device
from tinygrad.helpers import fetch, Context

from extra.onnx_helpers import validate
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry

def run_onnx_torch(onnx_model, inputs):
  import torch
  from onnx2torch import convert
  torch_model = convert(onnx_model).float()
  with torch.no_grad():
    torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
  return torch_out

np.random.seed(1337)

class TestOnnxModel(unittest.TestCase):
  def test_conv_image_dtype(self):
    import onnx
    for dtype, onnx_dtype in [(np.float16, onnx.TensorProto.FLOAT16), (np.float32, onnx.TensorProto.FLOAT)]:
      with self.subTest(dtype=dtype), Context(IMAGE=1, FLOAT16=1):
        graph = onnx.helper.make_graph([onnx.helper.make_node("Conv", ["X", "W", "B"], ["Y"])], "conv", [
          onnx.helper.make_tensor_value_info("X", onnx_dtype, (1, 4, 2, 2))], [
          onnx.helper.make_tensor_value_info("Y", onnx_dtype, (1, 4, 2, 2))], [
          onnx.numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=dtype), "W"),
          onnx.numpy_helper.from_array(np.array([0.125, 0.25, 0.5, 1], dtype=dtype), "B")])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        runner = OnnxRunner(Tensor(model.SerializeToString(), device="PYTHON"))
        out = runner({"X": Tensor(np.arange(16, dtype=dtype).reshape(1, 4, 2, 2))})["Y"]
        self.assertEqual(out.dtype, dtypes.float16 if dtype is np.float16 else dtypes.float32)
        expected = np.array([[[[24.125, 28.125], [32.125, 36.125]], [[24.25, 28.25], [32.25, 36.25]],
                              [[24.5, 28.5], [32.5, 36.5]], [[25, 29], [33, 37]]]], dtype=dtype)
        np.testing.assert_array_equal(out.numpy(), expected)

  def test_matmul_gemm_image_dtype(self):
    import onnx
    for op in ("MatMul", "Gemm"):
      for dtype, onnx_dtype in [(np.float16, onnx.TensorProto.FLOAT16), (np.float32, onnx.TensorProto.FLOAT)]:
        with self.subTest(op=op, dtype=dtype), Context(IMAGE=1, FLOAT16=1):
          names = ["X", "W"] if op == "MatMul" else ["X", "W", "B"]
          initializers = [onnx.numpy_helper.from_array(np.ones((4, 4), dtype=dtype), "W")]
          if op == "Gemm": initializers.append(onnx.numpy_helper.from_array(np.full(4, 0.5, dtype=dtype), "B"))
          graph = onnx.helper.make_graph([onnx.helper.make_node(op, names, ["Y"])], op, [
            onnx.helper.make_tensor_value_info("X", onnx_dtype, (4, 4))], [
            onnx.helper.make_tensor_value_info("Y", onnx_dtype, (4, 4))], initializers)
          model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
          runner = OnnxRunner(Tensor(model.SerializeToString(), device="PYTHON"))
          out = runner({"X": Tensor(np.arange(16, dtype=dtype).reshape(4, 4))})["Y"]
          self.assertEqual(out.dtype, dtypes.float16 if dtype is np.float16 else dtypes.float32)
          expected = [[6, 6, 6, 6], [22, 22, 22, 22], [38, 38, 38, 38], [54, 54, 54, 54]] if op == "MatMul" else [
            [6.5, 6.5, 6.5, 6.5], [22.5, 22.5, 22.5, 22.5], [38.5, 38.5, 38.5, 38.5], [54.5, 54.5, 54.5, 54.5]]
          np.testing.assert_array_equal(out.numpy(), np.array(expected, dtype=dtype))

  @unittest.skip("slow")
  def test_efficientnet(self):
    input_name, input_new = "images:0", True
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"),
      input_name, input_new)

  @unittest.skip("TODO: FIX THIS IT CAUSES SEGFAULT")
  def test_shufflenet(self):
    input_name, input_new = "gpu_0/data_0", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx"),
      input_name, input_new)

  @unittest.skip("test is very slow")
  def test_resnet(self):
    # NOTE: many onnx models can't be run right now due to max pool with strides != kernel_size
    input_name, input_new = "data", False
    self._test_model(
      fetch("https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx"),
      input_name, input_new)

  def _test_model(self, fn, input_name, input_new, debug=False):
    run_onnx = OnnxRunner(fn)
    print("onnx loaded")
    from test.models.test_efficientnet import chicken_img, car_img, preprocess, _LABELS

    def run(img):
      inputs = {input_name: preprocess(img, new=input_new)}
      tinygrad_out = list(run_onnx(inputs, debug=debug).values())[0].numpy()
      return tinygrad_out.argmax()

    cls = run(chicken_img)
    print(cls, _LABELS[cls])
    assert _LABELS[cls] == "hen" or _LABELS[cls] == "cock"
    cls = run(car_img)
    print(cls, _LABELS[cls])
    assert "car" in _LABELS[cls] or _LABELS[cls] == "convertible"

@unittest.skipUnless(Device.DEFAULT == "METAL", "only run on METAL")
class TestHuggingFaceOnnxModels(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls._ctx = Context(MAX_BUFFER_SIZE=0)
    cls._ctx.__enter__()

  @classmethod
  def tearDownClass(cls):
    cls._ctx.__exit__()

  def _validate(self, repo_id, model_file, custom_inputs, rtol=1e-4, atol=1e-4):
    onnx_model_path = snapshot_download_with_retry(
      repo_id=repo_id,
      allow_patterns=["*.onnx", "*.onnx_data"],
      local_dir=DOWNLOADS_DIR / repo_id
    )
    onnx_model_path = onnx_model_path / model_file
    file_size = onnx_model_path.stat().st_size
    print(f"Validating model: {repo_id}/{model_file} ({file_size/1e6:.2f}M)")
    validate(onnx_model_path, custom_inputs, rtol=rtol, atol=atol)

  def test_xlm_roberta_large(self):
    repo_id = "FacebookAI/xlm-roberta-large"
    model_file = "onnx/model.onnx"
    custom_inputs = {
      "input_ids": np.random.randint(0, 250002, (1, 11), dtype=np.int64),
      "attention_mask": np.ones((1, 11), dtype=np.int64),
    }
    self._validate(repo_id, model_file, custom_inputs, atol=1e-3)

if __name__ == "__main__":
  unittest.main()
