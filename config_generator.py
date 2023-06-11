r"""
Generate a default config file for pruning.
Output file default: `conf.json`
`conf.json`:
[
    {$module_name1$: $prune_rate1$, $module_name2$: $prune_rate2$, ...},   # config 1
    {$module_name1$: $prune_rate3$, $module_name3$: $prune_rate4$, ...},   # config 2
    ...
]
"""

import torch
import json
import argparse
import os
from utils import type_of_module


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the original model directory.")
parser.add_argument("-o", "--output", type=str, default="conf.json", help="Filename of generated config. Default 'conf.json'")
parser.add_argument("--attention", nargs='*', type=float, default=[0.25, 0.5, 0.25], help="List of prune rate of attention layers. Default 0.25 0.5 0.25")
parser.add_argument("--linear", nargs='*', type=float, default=[0.3, 0.3, 0.5], help="List of prune rate of linear layers. Default 0.3 0.3 0.5")
parser.add_argument("-n", "--conf_number", type=int, default=3, help="Config numbers. Default 3")
args = parser.parse_args()

PATH_TO_MODEL = args.model
PATH_TO_CONF = args.output
CONF_NUMBER = args.conf_number

assert os.path.exists(PATH_TO_MODEL), "MODEL FILE DOES NOT EXIST"
assert len(args.attention) == len(args.linear) == CONF_NUMBER, "LENGTH OF PRUNE RATE LISTS MUST BE EQUAL TO CONF NUMBER"
TYPE_OF_MODEL = ''
if 't5' in PATH_TO_MODEL:
    TYPE_OF_MODEL = 't5'
elif 'distilbert' in PATH_TO_MODEL:
    TYPE_OF_MODEL = 'distilbert'
print(f"Model type: {TYPE_OF_MODEL}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(PATH_TO_MODEL).to(device)

module_dict = {}
for name, value in model.named_modules():
    if type_of_module(TYPE_OF_MODEL, name, value) is not None:
        module_dict[name] = value


conf = []
for i in range(CONF_NUMBER):
    conf_i = {}
    for key in module_dict:
        if type_of_module(TYPE_OF_MODEL, key, module_dict[key]) == "MultiheadAttention":
            conf_i[key] = args.attention[i]
        elif type_of_module(TYPE_OF_MODEL, key, module_dict[key]) == "Linear":
            conf_i[key] = args.linear[i]
        else:
            # Raise an error and print the module name
            raise ValueError(f"Unknown module type: {key}")
    conf.append(conf_i)

conf_json = json.dumps(conf, indent=4)
conf_file = open(PATH_TO_CONF, 'w')
conf_file.write(conf_json)
conf_file.close()

print(f"Configurations saved to: {PATH_TO_CONF}")
