# Argument parser
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from utils import set_module


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--origin_model", type=str, default='./model/t5-small.pth', help="original (unpruned) model file (.pth)")
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/t5-small.pth", help='tokenizer file')
parser.add_argument("-p", "--finetuned_modules", type=str, default='./numpy/module_finetuned.npy', help="file of pruned modules")
parser.add_argument("-c", "--conf_file", type=str, default='./conf.json', help="configuration file")
parser.add_argument('-n', '--use_nth_config', type=int, default=0, help="Compose a model based on `n`-th config in configuration file. `n` starts at 0.")
parser.add_argument('-o', '--output', type=str, default="./model/t5-small_composed.pth", help='output .pth')
args = parser.parse_args()

PATH_TO_ORIGINAL_MODEL = args.origin_model      # input
PATH_TO_TOKENIZER = args.tokenizer              # input
PATH_TO_FINETUNED_MODULES = args.finetuned_modules   # input
PATH_TO_CONFIG = args.conf_file                 # input
assert os.path.exists(PATH_TO_ORIGINAL_MODEL), "Model file (.pth) not found!"
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file (.pth) not found!"
assert os.path.exists(PATH_TO_FINETUNED_MODULES), "Finetuned modules file (.npy) not found!"
assert os.path.exists(PATH_TO_CONFIG), "Configuration file (.json) not found!"
PATH_TO_OUTPUT_MODEL = args.output      # output


NTH_CONFIG = args.use_nth_config


MANUAL_SEED = 42


# Set random seed
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)

# Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load everything
tokenizer = torch.load(PATH_TO_TOKENIZER)
model = torch.load(PATH_TO_ORIGINAL_MODEL)
finetuned_modules = np.load(PATH_TO_FINETUNED_MODULES, allow_pickle=True).item()
conf_file = open(PATH_TO_CONFIG, 'r')
conf = json.load(conf_file)
model_total_number = len(conf)
assert NTH_CONFIG < model_total_number, "Nth config is out of range!"
conf_nth = conf[NTH_CONFIG]


def find_finetuned_module_by_name_and_ratio(finetuned_modules: dict, name: str, ratio: float) -> tuple[nn.Module, list]:
    assert name in finetuned_modules, f"MODULE {name} NOT FOUND"
    info_module_list = finetuned_modules[name]      # [{"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}, ...]
    for info_module in info_module_list:            # {"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}
        if info_module["ratio"] == ratio:
            if hasattr(info_module, "head_mask"):       # is a MultiheadAttention layer
                return info_module["layer"], info_module["head_mask"]
            else:
                return info_module["layer"], None
    return None, None


# Compose model
multihead_attn_modules = []     # Save the MultiHeadAttention modules for later grad zero
for name in conf_nth:
    ratio = conf_nth[name]
    if ratio == 1:      # Skip if ratio is 1, which is the original version of module
        continue
    assert name in finetuned_modules, f"Module {name} not found in finetuned_modules"
    module, head_mask = find_finetuned_module_by_name_and_ratio(finetuned_modules, name, ratio)
    assert module is not None, f"Module {name} by ratio {ratio} is not found in finetuned_modules"
    if head_mask is not None:   # If not None, it is a MultiHeadAttention module
        assert not hasattr(module, 'head_mask'), f"Module {name} already has head_mask"
        module.head_mask = head_mask
        multihead_attn_modules.append(module)       # Save the module for later grad zero
    set_module(model, name, module)


# Save model
torch.save(model, PATH_TO_OUTPUT_MODEL)
print(f"Model saved to {PATH_TO_OUTPUT_MODEL}")
PATH_TO_SAVE_ARGS = "./model/checkpoint/compose_model/args.json"
with open(PATH_TO_SAVE_ARGS, 'w') as f:
    json.dump(vars(args), f)
print(f"Args saved to {PATH_TO_SAVE_ARGS}")
