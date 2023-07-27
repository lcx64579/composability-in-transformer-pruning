import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import random
import copy
import json
from utils import type_of_module, type_of_model, get_embed_dim, get_num_heads
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="./model/t5-small.pth", help='model file')
parser.add_argument('-c', '--conf', type=str, default="./conf_prune.json", help='pruning scheme file')
parser.add_argument('-o', '--output', type=str, default="./numpy/module_pruned.npy", help='output pruned modules')
args = parser.parse_args()

assert os.path.exists(args.model), "Model file not found!"
assert os.path.exists(args.conf), "Pruning scheme file not found!"

PATH_TO_ORIGINAL_MODEL = args.model
PATH_TO_CONF = args.conf
PATH_TO_MODULE_PRUNED = args.output
GENERATE_DICT = True
PATH_TO_MODULE_DICT = "./numpy/module_dict.npy"
TYPE_OF_MODEL = type_of_model(PATH_TO_ORIGINAL_MODEL)
assert TYPE_OF_MODEL is not None, "Model type not supported!"
print(f"Model type: {TYPE_OF_MODEL}")

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load(PATH_TO_ORIGINAL_MODEL)
conf = json.load(open(args.conf, 'r'))

EMB_SIZE = get_embed_dim(TYPE_OF_MODEL, model)
NHEAD = get_num_heads(TYPE_OF_MODEL, model)


if GENERATE_DICT:
    module_dict = {}
    for name, value in model.named_modules():
        if type_of_module(TYPE_OF_MODEL, name, value) is not None and name in conf:
            module_dict[name] = value
    np.save(PATH_TO_MODULE_DICT, module_dict)

module_dict = np.load(PATH_TO_MODULE_DICT, allow_pickle=True).item()


def prune_linear(name: str, layer: nn.Module, ratio: float) -> nn.Module:
    """Prune a Linear layer to a certain ratio by L1-norm.

    Args:
        name (str): name of the layer
        layer (nn.Module): the layer to be pruned
        ratio (float): the ratio of nonzero weights to be pruned
    Returns:
        layer (nn.Module): the pruned layer
    """
    # Determine whether the layer has bias. If so, prune the bias together.
    HAS_BIAS = False
    if hasattr(layer, 'bias') and layer.bias is not None:
        HAS_BIAS = True

    total = layer.weight.data.numel()       # total number of weights
    if HAS_BIAS:
        total += layer.bias.data.numel()

    prune.l1_unstructured(layer, 'weight', amount=1.0-ratio)    # prune
    if HAS_BIAS:
        prune.l1_unstructured(layer, 'bias', amount=1.0-ratio)  # prune biases

    total_nonzero = layer.weight.data.nonzero().shape[0]    # number of nonzero weights
    if HAS_BIAS:
        total_nonzero += layer.bias.data.nonzero().shape[0]
    # An alternative way to prune:
    # mask = layer.weight.data.abs().clone().gt(0).float().cuda()   # mask of nonzero weights
    # mask = layer.weight.data.abs().gt(0)  # mask of nonzero weights
    # total_nonzero = torch.sum(mask).int().item()    # number of nonzero weights
    # prune.remove(layer, 'weight')      # apply pruning
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}).")  # print pruning ratio
    return layer


def prune_attention(name: str, layer: nn.Module, ratio: float, embed_dim: int, num_heads: int) -> tuple[nn.Module, torch.Tensor]:
    """Prunes a Conv1D attention layer by removing the least confident heads.
    Returns the pruned layer and a binary mask indicating which heads were pruned.

    Args:
        name (str): The name of the layer being pruned.
        layer (nn.Module): The convolutional layer to be pruned.
        ratio (float): The ratio of heads to prune.
        embed_dim (int): The size of the embedding dimension.
        num_heads (int): The number of attention heads in the layer.

    Returns:
        tuple[nn.Module, torch.Tensor]: A tuple containing the pruned layer and a binary mask indicating which heads were pruned.
    """
    # Determine whether the layer has bias. If so, prune the bias together.
    HAS_BIAS = False
    if hasattr(layer, 'bias') and layer.bias is not None:
        HAS_BIAS = True

    if TYPE_OF_MODEL == "t5":
        q = layer.q
        k = layer.k
        v = layer.v
        o = layer.o
    elif TYPE_OF_MODEL == "distilbert":
        q = layer.q_lin
        k = layer.k_lin
        v = layer.v_lin
        o = layer.out_lin

    # Code follows is only for counting. Bias is not considered.
    total = q.weight.numel() + k.weight.numel() + v.weight.numel() + o.weight.numel()

    head_dim = embed_dim // num_heads       # Example: 512 dimensions with 8 heads => 64 dimensions per head. The loaded model has this shape.
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    head_scores = [0] * num_heads           # head_scores[] is used to record the "confidence" of each head
    # Perform operations on only one of Q, K, or V.
    for head in range(num_heads):
        dim_from = head * head_dim
        dim_to = dim_from + head_dim
        # Determine whether to prune: Use the sum of the weights of one head as the confidence score.
        # Use only weight. Bias is not used.
        q_weight = q.weight[dim_from:dim_to]
        k_weight = k.weight[dim_from:dim_to]
        v_weight = v.weight[dim_from:dim_to]
        head_score = q_weight.abs().sum().item() + k_weight.abs().sum().item() + v_weight.abs().sum().item()
        head_scores[head] += head_score                 # Accumulate the confidence of this QKV head

    _, head_scores_rank = torch.Tensor(head_scores).sort()  # Sort by confidence score to obtain the ranking of each head: rank[0] is the least confident, rank[7] is the most confident.

    threshold = int(num_heads * (1 - ratio) + 0.5)      # Number of heads to prune, rounded up
    prune_heads_index = head_scores_rank[:threshold]    # The top `threshold` heads, i.e., the least confident ones
    head_mask = torch.ones(num_heads)                   # Create a mask, 0 means prune, 1 means keep
    head_mask[prune_heads_index] = 0                    # The mask for these unconfident heads is 0, which will be pruned.

    #### Option 1: PYTORCH CUSTOM FROM MASK. Automatically handles gradiant clipping ####
    # Create a mask which is the same size as the weights
    # custom_mask = torch.ones_like(layer.weight.data)    # Create a mask, 0 means prune, 1 means keep
    # for head in range(num_heads):
    #     dim_from = head * head_dim
    #     dim_to = dim_from + head_dim
    #     custom_mask[dim_from:dim_to].mul_(head_mask[head])

    # prune.CustomFromMask.apply(q, 'weight', custom_mask)    # Apply the mask to the layer
    # if HAS_BIAS:
    #     custom_mask_bias = torch.ones_like(layer.bias.data)     # Create a mask, 0 means prune, 1 means keep
    #     for head in range(num_heads):
    #         dim_from = head * head_dim
    #         dim_to = dim_from + head_dim
    #         custom_mask_bias[dim_from:dim_to].mul_(head_mask[head])
    #     prune.CustomFromMask.apply(layer, 'bias', custom_mask_bias)  # Apply the mask to the layer


    #### Option 2: PRUNE MANUALLY. Handle gradiant clipping by yourself ####
    # for head in range(num_heads):
    #     dim_from = head * head_dim
    #     dim_to = dim_from + head_dim
    #     head_weight = layer.weight[dim_from:dim_to]
    #     with torch.no_grad():
    #         head_weight.data.mul_(head_mask[head])      # Apply the mask to each head. (only weight. There is no bias in these layers.)
    #     if HAS_BIAS:
    #         head_bias = layer.bias[dim_from:dim_to]
    #         with torch.no_grad():
    #             head_bias.data.mul_(head_mask[head])

    #### Option 3: REMOVE THE HEADS. ####
    # This will cause a bug in the inference process (transformers 4.29.0). See https://github.com/huggingface/transformers/issues/19625#issuecomment-1296708659 for details.
    # To solve this bug, follow this PR: https://github.com/huggingface/transformers/pull/20106/commits/1c036908365b9ea448498d5a9f63a89bb58b2aa2
    prune_heads_index = prune_heads_index.tolist()
    layer.prune_heads(prune_heads_index)

    print(f'prune_heads_index: {prune_heads_index}')
    # print(f'shape of q.weight: {layer.q.weight.shape}')

    # nonzero = layer.in_proj_weight.data.abs().clone().gt(0).float().cuda()      # Counting
    # total_nonzero = torch.sum(nonzero).int().item()         # Counting: number of remaining parameters after pruning
    # Code follows is only for counting. Bias is not considered

    # a simple `q` does not work. Has to use `layer.q`
    if TYPE_OF_MODEL == "t5":
        total_nonzero = layer.q.weight.nonzero().shape[0] + layer.k.weight.nonzero().shape[0] + layer.v.weight.nonzero().shape[0] + layer.o.weight.nonzero().shape[0]
    elif TYPE_OF_MODEL == "distilbert":
        total_nonzero = layer.q_lin.weight.nonzero().shape[0] + layer.k_lin.weight.nonzero().shape[0] + layer.v_lin.weight.nonzero().shape[0] + layer.out_lin.weight.nonzero().shape[0]
    head_nonzero = head_mask.nonzero().shape[0]             # Counting: number of remaining heads after pruning
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}). {head_nonzero} heads left.")  # Counting

    return layer, head_mask


pruned_modules = {}

for name_module in conf:
    ratio_list = conf[name_module]
    layer = module_dict[name_module]
    module_list = []
    module_type = type_of_module(TYPE_OF_MODEL, name_module, layer)
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
