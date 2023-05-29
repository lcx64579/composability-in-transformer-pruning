"""
由配置文件生成剪枝用配置文件。配置文件名：conf.json。剪枝配置文件名：conf_prune.json
conf.json内容：
[
    {$module_name$: $prune_rate$, $module_name$: $prune_rate$, ...},   # config 1
    {$module_name$: $prune_rate$, $module_name$: $prune_rate$, ...},   # config 2
    ...
]
conf_prune.json内容：
{
    $module_name$: [$prune_rate$, $prune_rate$, ...],
    $module_name$: [$prune_rate$, $prune_rate$, ...],
}
"""


import torch
import torch.nn as nn
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--conf_file", type=str, required=True, help="File to the config.")
# parser.add_argument("-o", "--output", type=str, default="conf_prune.json", help="Filename of generated pruning config. Default 'conf_prune.json'")
args = parser.parse_args()

CONFIG_FILE = args.conf_file
CONFIG_PRUNE_FILE = "conf_prune.json"

assert os.path.exists(CONFIG_FILE), "FILE NOT EXIST"

conf_file = open(CONFIG_FILE, 'r')
conf = json.load(conf_file)
CONFIG_NUMBER = len(conf)

conf_prune = {}
for conf_i in conf:
    for key in conf_i:
        if conf_i[key] == 1:       # 1 means no pruning on this module
            continue
        if key not in conf_prune:       # if no config for this block
            conf_prune[key] = []
        if conf_i[key] not in conf_prune[key]:      # if same ratio for a block in multiple configs, only records one of them
            conf_prune[key].append(conf_i[key])

conf_prune_json = json.dumps(conf_prune, indent=4)
conf_prune_file = open(CONFIG_PRUNE_FILE, 'w')
conf_prune_file.write(conf_prune_json)
conf_prune_file.close()
