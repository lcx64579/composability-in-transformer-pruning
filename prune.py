import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import random
import copy
import json
from utils import type_of_t5_module
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="./model/t5-small.pth", help='model file')
parser.add_argument('-c', '--conf', type=str, default="./conf_prune.json", help='pruning scheme file')
parser.add_argument('-o', '--output', type=str, default="./numpy/module_pruned.npy", help='output pruned modules')
# parser.add_argument('-p', '--output_pruned_model', type=str, default="./model/t5-small_pruned.pth", help='output pruned model file (PyTorch .pth)')
args = parser.parse_args()

assert os.path.exists(args.model), "Model file not found!"
assert os.path.exists(args.conf), "Pruning scheme file not found!"

PATH_TO_ORIGINAL_MODEL = args.model
PATH_TO_CONF = args.conf
PATH_TO_MODULE_PRUNED = args.output
# PATH_TO_MODEL_PRUNED = args.output_pruned_model
GENERATE_DICT = False   # Only set to True if the file at PATH_TO_MODULE_DICT is never generated or changed. Otherwise, set to False.
PATH_TO_MODULE_DICT = "./numpy/module_dict.npy"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = T5ForConditionalGeneration.from_pretrained(PATH_TO_ORIGINAL_MODEL)
model = torch.load(PATH_TO_ORIGINAL_MODEL)
conf = json.load(open(args.conf, 'r'))

EMB_SIZE = model.config.d_model
NHEAD = model.config.num_heads


if GENERATE_DICT:
    module_dict = {}
    # For each module in the model, if it is a Conv1D layer and its name is in the conf file, add it to module_dict
    for name, value in model.named_modules():
        if type_of_t5_module(name, value) is not None and name in conf:
            module_dict[name] = value
    np.save(PATH_TO_MODULE_DICT, module_dict)

module_dict = np.load(PATH_TO_MODULE_DICT, allow_pickle=True).item()


def prune_linear(name: str, layer: nn.Module, ratio: float) -> nn.Module:
    """Prune a Conv1D linear layer to a certain ratio by L1-norm.

    Args:
        name (str): name of the layer
        layer (Conv1D): the layer to be pruned
        ratio (float): the ratio of nonzero weights to be pruned
    Returns:
        layer (Conv1D): the pruned layer
    """
    total = layer.weight.data.numel()       # total number of weights
    prune.l1_unstructured(layer, 'weight', amount=1.0-ratio)    # prune
    total_nonzero = layer.weight.data.nonzero().shape[0]    # number of nonzero weights
    # An alternative way to prune:
    # mask = layer.weight.data.abs().clone().gt(0).float().cuda()   # mask of nonzero weights
    # mask = layer.weight.data.abs().gt(0)  # mask of nonzero weights
    # total_nonzero = torch.sum(mask).int().item()    # number of nonzero weights
    # prune.remove(layer, 'weight')      # apply pruning
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}).")  # print pruning ratio
    return layer


def prune_attention(name: str, layer: nn.Module, ratio: float, embed_dim: int, num_heads: int) -> tuple[nn.Module, torch.Tensor]:
    """
    Prunes a Conv1D attention layer by removing the least confident heads. Returns the pruned layer and a binary mask indicating which heads were pruned.

    Args:
        name (str): The name of the layer being pruned.
        layer (Conv1D): The convolutional layer to be pruned.
        ratio (float): The ratio of heads to prune.
        embed_dim (int): The size of the embedding dimension.
        num_heads (int): The number of attention heads in the layer.

    Returns:
        tuple[Conv1D, torch.Tensor]: A tuple containing the pruned layer and a binary mask indicating which heads were pruned.
    """
    # print(name, m, n)
    # assert isinstance(layer, t5.T5Attention), "Pruning attention layer requires a T5Attention layer."

    total = layer.weight.numel()        # total number of weights

    head_dim = embed_dim // num_heads       # Example: 512 dimensions with 8 heads => 64 dimensions per head. The loaded model has this shape.
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    head_scores = [0] * num_heads           # head_scores[] is used to record the "confidence" of each head
    # Perform operations on Q, K, and V separately
    for head in range(num_heads):
        dim_from = head * head_dim
        dim_to = dim_from + head_dim
        head_weight = layer.weight[dim_from:dim_to]
        # Determine whether to prune: Use the sum of the weights of one head as the confidence score.
        head_score = head_weight.abs().sum().item()
        head_scores[head] += head_score                 # Accumulate the confidence of this QKV head

    _, head_scores_rank = torch.Tensor(head_scores).sort()  # Sort by confidence score to obtain the ranking of each head: rank[0] is the least confident, rank[7] is the most confident.

    threshold = int(num_heads * (1 - ratio) + 0.5)      # Number of heads to prune, rounded up
    prune_heads_index = head_scores_rank[:threshold]    # The top `threshold` heads, i.e., the least confident ones
    head_mask = torch.ones(num_heads)                   # Create a mask, 0 means prune, 1 means keep
    head_mask[prune_heads_index] = 0                    # The mask for these unconfident heads is 0, which will be pruned.

    for head in range(num_heads):
        dim_from = head * head_dim
        dim_to = dim_from + head_dim
        head_weight = layer.weight[dim_from:dim_to]
        with torch.no_grad():
            head_weight.data.mul_(head_mask[head])      # Apply the mask to each head. (only weight. There is no bias in these layers.)

    # nonzero = layer.in_proj_weight.data.abs().clone().gt(0).float().cuda()      # Counting
    # total_nonzero = torch.sum(nonzero).int().item()         # Counting: number of remaining parameters after pruning
    total_nonzero = layer.weight.nonzero().shape[0]         # Counting: number of remaining parameters after pruning
    head_nonzero = head_mask.nonzero().shape[0]             # Counting: number of remaining heads after pruning
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}). {head_nonzero} heads left.")  # Counting

    return layer, head_mask


pruned_modules = {}

for name_module in conf:
    ratio_list = conf[name_module]
    layer = module_dict[name_module]
    module_list = []
    module_type = type_of_t5_module(name_module, layer)
    if module_type == "Linear":
        for ratio in ratio_list:
            this_layer = copy.deepcopy(layer)
            pruned_layer = prune_linear(name_module, this_layer, ratio)
            module_list.append({"layer": pruned_layer, "ratio": ratio})
    elif module_type == "MultiheadAttention":
        for ratio in ratio_list:
            this_layer = copy.deepcopy(layer)
            pruned_layer, head_mask = prune_attention(name_module, this_layer, ratio, EMB_SIZE, NHEAD)
            module_list.append({"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask})
    else:
        raise ValueError(f"Unknown module type: {module_type} of {name_module}")
    pruned_modules[name_module] = module_list
    # `pruned_modules` is a dictionary {"module_name": [{}, {}, ...]},
    # the key is the name of the module, and the value is a list of dictionaries,
    # each dictionary contains:
    # {"layer", "ratio", ["head_mask"]}


np.save(PATH_TO_MODULE_PRUNED, pruned_modules, allow_pickle=True)
print(f"Pruned modules saved to {PATH_TO_MODULE_PRUNED}.")

# # model = T5ForConditionalGeneration.from_pretrained(PATH_TO_ORIGINAL_MODEL)   # Load the original model again
# model = torch.load(PATH_TO_ORIGINAL_MODEL)
# for module_name in pruned_modules:
#     module = pruned_modules[module_name][0]["layer"]
#     set_module(model, module_name, module)

# torch.save(model, PATH_TO_MODEL_PRUNED)
# # model.save_pretrained(PATH_TO_MODEL_PRUNED)
# print(f"A test pruned model (pruned with every first module in pruning scheme) saved to {PATH_TO_MODEL_PRUNED}.")
