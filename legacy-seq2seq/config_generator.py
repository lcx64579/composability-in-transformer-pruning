"""
配置文件生成。配置文件名：conf.json
内容：
[
    {$module_name1$: $prune_rate1$, $module_name2$: $prune_rate2$, ...},   # config 1
    {$module_name1$: $prune_rate3$, $module_name3$: $prune_rate4$, ...},   # config 2
    ...
]
"""


import torch
import torch.nn as nn
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_file", type=str, required=True, help="File to the original model.")
parser.add_argument("-o", "--output", type=str, default="conf.json", help="Filename of generated config. Default 'conf.json'")
parser.add_argument("-n", "--conf_number", type=int, default=3, help="Config numbers. Default 3")
args = parser.parse_args()

MODEL_BASELINE_FILE = args.model_file
CONFIG_FILE = args.output
CONFIG_NUMBER = args.conf_number

assert os.path.exists(MODEL_BASELINE_FILE), "FILE NOT EXIST"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_BASELINE_FILE).to(device)
module_dict = {}
for name, value in model.named_modules():
    if (isinstance(value, nn.Linear) or isinstance(value, nn.MultiheadAttention)) \
            and not isinstance(value, nn.modules.linear.NonDynamicallyQuantizableLinear):
        # print(name)
        module_dict[name] = value

conf = []
for i in range(CONFIG_NUMBER):
    conf_i = {}
    for key in module_dict:
        conf_i[key] = 0.5
    conf.append(conf_i)

conf_json = json.dumps(conf, indent=4)
conf_file = open(CONFIG_FILE, 'w')
conf_file.write(conf_json)
conf_file.close()
