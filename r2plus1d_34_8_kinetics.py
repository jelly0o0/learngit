import torch
model = torch.hub.load("moabitcoin/ig65m-pytorch", 'r2plus1d_34_8_kinetics', num_classes=400, pretrained=True)
dummy_data = torch.Tensor(1, 3, 8, 112, 112)
torch.onnx.export(model, dummy_data, 'r2plus1d_34_8_kinetics.onnx', export_params=True, input_names = ['input'], output_names = ['output'])
