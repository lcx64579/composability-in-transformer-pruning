import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="model file (.pth)")
args = parser.parse_args()

PATH_TO_MODEL = args.model
assert os.path.exists(PATH_TO_MODEL), "model file not found"

# Load model
model = torch.load(PATH_TO_MODEL)

# Count parameters
total_parameters = 0
total_nonzero_parameters = 0
for parameter in model.parameters():
    total_parameters += parameter.numel()

for layer in model.modules():
    if hasattr(layer, 'weight'):
        nonzero = layer.weight.data.nonzero().shape[0]
        total_nonzero_parameters += nonzero
    if hasattr(layer, 'bias') and layer.bias is not None:
        nonzero = layer.bias.data.nonzero().shape[0]
        total_nonzero_parameters += nonzero

print(f"Total parameters: {total_parameters}")
print(f"Total nonzero parameters: {total_nonzero_parameters}")
