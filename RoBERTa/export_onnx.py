import torch
import onnx
import collections
from model import naverModel
from onnx import shape_inference



def main(model):
    dummy_ids = torch.ones(1, 514, dtype=torch.long)
    dummy_masks = torch.zeros(1, 514, dtype=torch.long)
    torch.onnx.export(model, (dummy_ids, dummy_masks), "roberta.onnx", input_names=['input_ids', 'attention_mask'])

    path = "roberta.onnx"
    onnx.save(shape_inference.infer_shapes(onnx.load(path)), path)



if __name__=='__main__':
    model = naverModel(num_classes=5)
    params = torch.load("roberta.prm", map_location="cpu")

    new_state_params = collections.OrderedDict()

    for n, v in params.items():
        name = n.replace("module.", "")
        new_state_params[name] = v

    model.load_state_dict(new_state_params)

    model.eval()

    main(model)