import torch
import CRNN

model = CRNN.CRNN()
model.load_state_dict(torch.load("OnlineWeights.pth",map_location='cpu'))

x = torch.randn(1, 1, 129, 501, requires_grad=True)
torch_out = torch.onnx._export(model,x,"online.onnx",export_params=True)
