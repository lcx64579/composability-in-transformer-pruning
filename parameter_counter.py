import torch
import argparse
import os
from utils import type_of_module

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
total_zero_parameters = 0
for parameter in model.parameters():
    total_parameters += parameter.numel()

for name, layer in model.named_modules():
    if type_of_module('t5', name, layer) == "MultiheadAttention":
        nonzero_parameters = layer.q.weight.nonzero().size(0) + layer.k.weight.nonzero().size(0) + layer.v.weight.nonzero().size(0) + layer.o.weight.nonzero().size(0)
        layer_parameters = layer.q.weight.numel() + layer.k.weight.numel() + layer.v.weight.numel() + layer.o.weight.numel()
        total_zero_parameters += layer_parameters - nonzero_parameters
        print(f'{name} has {layer_parameters} parameters, {nonzero_parameters} nonzero parameters, {layer_parameters - nonzero_parameters} zero parameters')
    elif type_of_module('t5', name, layer) == "Linear":
        total_zero_parameters += layer.weight.numel() - layer.weight.nonzero().size(0)
        print(f'{name} has {layer.weight.numel()} parameters, {layer.weight.nonzero().size(0)} nonzero parameters, {layer.weight.numel() - layer.weight.nonzero().size(0)} zero parameters')

total_nonzero_parameters = total_parameters - total_zero_parameters

print(f"Total parameters: {total_parameters}")
print(f"Total nonzero parameters: {total_nonzero_parameters}")
