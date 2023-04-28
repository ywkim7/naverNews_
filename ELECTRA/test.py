import onnx
import onnx.numpy_helper as numpy_helper
from onnx import helper, shape_inference

onnx_path = '/app/PoC/ELECTRA/electra.onnx'

onnx_model = onnx.load(onnx_path)

inferred_model = shape_inference.infer_shapes(onnx_model)
print(inferred_model.graph.value_info)