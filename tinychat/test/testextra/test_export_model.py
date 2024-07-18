import unittest
from extra.export_model import export_model, EXPORT_SUPPORTED_DEVICE
from tinygrad.tensor import Tensor, Device
import json

class MockMultiInputModel:
  def forward(self, x1, x2, x3):
    return x1 + x2 + x3

class MockMultiOutputModel:
  def __call__(self, x1):
    return x1 + 2.0, x1.pad(((0, 0), (0, 1))) + 1.0

# TODO: move compile_efficientnet tests here
@unittest.skipUnless(Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, f"Model export is not supported on {Device.DEFAULT}")
class TextModelExport(unittest.TestCase):
  def test_multi_input_model_export(self):
    model = MockMultiInputModel()
    inputs = [Tensor.rand(2,2), Tensor.rand(2,2), Tensor.rand(2,2)]
    prg, inp_sizes, _, _ = export_model(model, "", *inputs)
    prg = json.loads(prg)

    assert len(inputs) == len(prg["inputs"]) == len(inp_sizes), f"Model and exported inputs don't match: mdl={len(inputs)}, prg={len(prg['inputs'])}, inp_sizes={len(inp_sizes)}"  # noqa: E501

    for i in range(len(inputs)):
      assert f"input{i}" in inp_sizes, f"input{i} not captured in inp_sizes"
      assert f"input{i}" in prg["buffers"], f"input{i} not captured in exported buffers"

    for i, exported_input in enumerate(prg["inputs"]):
      assert inputs[i].dtype.name == exported_input["dtype"], f"Model and exported input dtype don't match: mdl={inputs[i].dtype.name}, prg={exported_input['dtype']}"  # noqa: E501

  def test_multi_output_model_export(self):
    model = MockMultiOutputModel()
    input_tensor = Tensor.rand(2,2)
    outputs = model(input_tensor)
    prg, _, out_sizes, _ = export_model(model, "", input_tensor)
    prg = json.loads(prg)

    assert len(outputs) == len(prg["outputs"]) == len(out_sizes), f"Model and exported outputs don't match: mdl={len(outputs)}, prg={len(prg['outputs'])}, inp_sizes={len(out_sizes)}"  # noqa: E501

    for i in range(len(outputs)):
      assert f"output{i}" in out_sizes, f"output{i} not captured in out_sizes"
      assert f"output{i}" in prg["buffers"], f"output{i} not captured in exported buffers"

    for i, exported_output in enumerate(prg["outputs"]):
      assert outputs[i].dtype.name == exported_output["dtype"], f"Model and exported output dtype don't match: mdl={outputs[i].dtype.name}, prg={exported_output['dtype']}"  # noqa: E501


if __name__ == '__main__':
  unittest.main()
