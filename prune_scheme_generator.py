r"""
Generate pruning scheme from config.
Input file default: `conf.json`
Output file default: `conf_prune.json`
`conf.json`:
[
    {$module_name$: $prune_rate$, $module_name$: $prune_rate$, ...},   # config 1
    {$module_name$: $prune_rate$, $module_name$: $prune_rate$, ...},   # config 2
    ...
]
`conf_prune.json`:
{
    $module_name$: [$prune_rate$, $prune_rate$, ...],
    $module_name$: [$prune_rate$, $prune_rate$, ...],
}
"""

import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--conf_file", type=str, default="conf.json", help="Path to the config file.")
parser.add_argument("-o", "--output", type=str, default="conf_prune.json", help="Filename of generated pruning scheme. Default 'conf_prune.json'")
args = parser.parse_args()

PATH_TO_CONF = args.conf_file
PATH_TO_OUTPUT = args.output

assert os.path.exists(PATH_TO_CONF), "FILE NOT EXIST"

# Load config file as a json object
file_conf = open(PATH_TO_CONF, 'r')
conf = json.load(file_conf)
CONFIG_NUMBER = len(conf)

prune_scheme = {}
for conf_i in conf:
    for key in conf_i:
        if conf_i[key] == 1:       # 1 means no pruning performed on this module
            continue
        if key not in prune_scheme:       # if no config for this block
            prune_scheme[key] = []
        if conf_i[key] not in prune_scheme[key]:      # if same pruning rate for a block in multiple configs, only records one of them
            prune_scheme[key].append(conf_i[key])

json_prune_scheme = json.dumps(prune_scheme, indent=4)
file_prune_scheme = open(PATH_TO_OUTPUT, 'w')
file_prune_scheme.write(json_prune_scheme)
file_prune_scheme.close()

print(f"Pruning scheme saved to: {PATH_TO_OUTPUT}")
