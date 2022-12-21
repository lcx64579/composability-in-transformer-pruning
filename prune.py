import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np
import random
import copy
import json
from typing import Iterable, List
from model import *
from utils import set_module
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--pruning_rate", type=float, required=True, help="e.g. 0.5")
parser.add_argument("--generate_config", dest="generate_config", action="store_true")
args = parser.parse_args()
assert args.pruning_rate > 0 and args.pruning_rate <= 1, "Illegal pruning rate reported"
pruning_rate_underline = str(args.pruning_rate).replace(".", "_")

MODEL_BASELINE_FILE = "./model/baseline.pth"
MODEL_PRUNED_FILE = "./model/pruned_all_" + pruning_rate_underline + ".pth"
CONFIG_FILE = "./conf_prune.json"
MODULE_PRUNED_FILE = "./numpy/modules_pruned_all_" + pruning_rate_underline + ".npy"
MODULE_DICT_FILE = "./numpy/module_dict.npy"
GENERATE_DICT = False       # 除非1)从未生成过module_dict.npy这个文件2)换了新模型，否则不用打开
GENERATE_CONFIG = args.generate_config
GENERATE_CONFIG_PRUNE_RATE = args.pruning_rate


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}


# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

model = torch.load(MODEL_BASELINE_FILE).to(device)

if GENERATE_DICT:
    module_dict = {}
    for name, value in model.named_modules():
        if (isinstance(value, nn.Linear) or isinstance(value, nn.MultiheadAttention)) \
                and not isinstance(value, nn.modules.linear.NonDynamicallyQuantizableLinear):
            # `nn.modules.linear.NonDynamicallyQuantizableLinear` is a linear layer located in `nn.MultiheadAttention`.
            # 就是里面的out_proj对象.
            # `nn.modules.activation.MultiheadAttention` equals to `nn.MultiheadAttention`.
            # print(f"{name}\t{type(value)}")
            print(name)
            module_dict[name] = value
    np.save(MODULE_DICT_FILE, module_dict)
module_dict = np.load(MODULE_DICT_FILE, allow_pickle=True).item()


if GENERATE_CONFIG:
    conf = {}
    for key in module_dict:
        conf[key] = [GENERATE_CONFIG_PRUNE_RATE]

    conf_json = json.dumps(conf, indent=4)
    conf_file = open(CONFIG_FILE, 'w')
    conf_file.write(conf_json)
    conf_file.close()
conf_file = open(CONFIG_FILE, 'r')
conf = json.load(conf_file)


# Pruning
def prune_linear(name: str, layer: nn.Linear, ratio: float) -> nn.Linear:
    r"""
    剪枝全连接层。按 L1-Unstructured 方法剪枝。TODO: 将来可以通过参数传入自定义剪枝方法。

    Args:
        `name`: 层名，用于输出
        `layer`: 层
        `ratio`: 剪枝比，留下该比例的参数，而将`1 - ratio`比例的参数置为0
    Outputs:
        `layer`:  剪枝完成的层
    """
    total = layer.weight.data.numel()       # 统计用：剪枝前总参数量
    prune.l1_unstructured(layer, 'weight', amount=1.0-ratio)    # 剪枝只有这一行。PyTorch API
    # mask = layer.weight.data.abs().clone().gt(0).float().cuda()  # 统计用
    # mask = layer.weight.data.abs().gt(0)  # 统计用 - alternative
    # total_nonzero = torch.sum(mask).int().item()                # 统计用：剪枝后剩余参数量
    total_nonzero = layer.weight.data.nonzero().shape[0]
    # prune.remove(layer, 'weight')           # 应用剪枝。但留着mask可以在训练时对梯度自动应用mask
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}).")  # 统计
    return layer


def prune_attention(name: str, layer: nn.MultiheadAttention, ratio: float, embed_dim: int, num_heads: int) -> tuple[nn.MultiheadAttention, torch.Tensor]:
    r"""
    按头剪枝多头自注意力层的`in_proj_weight`。必须是自注意力 (Self-Attention)。

    将一个头的权重和作为其「自信度」，将不自信的头（即权重小的头）剪掉。

    Args:
        `name`: 层名。用于输出
        `layer`: 层
        `ratio`: 剪枝比。留下该比例的头，而将`1 - ratio`比例的头置为0
        `embed_dim`: 输入的 Embedding 的维度数
        `num_heads`: 多头注意力的总头数。应当能整除`embed_dim`，即`embed_dim // num_heads`为每头维度数
    Outputs:
        `layer`:  剪枝完成的层
        `head_mask`: 该层的头剪枝掩模，是一个长度为`num_heads`的一维张量
    """
    total = layer.in_proj_weight.data.numel()       # 统计用

    head_dim = embed_dim // num_heads       # 例：512维，8头 => 每头64维。这份代码加载的模型就是这个形状
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    head_scores = [0] * num_heads           # head_scores[] 用于记录每个头的「自信度」
    for threetimes in range(3):             # 分别对Q，K，V进行操作
        start_dim = embed_dim * threetimes  # 对于例，Q，K，V分别是从0、512、1024开始的512维。记下从哪维开始
        for head in range(num_heads):
            dim_from = start_dim + head * head_dim
            dim_to = dim_from + head_dim
            head_weight = layer.in_proj_weight[dim_from:dim_to]     # Q或K或V的这一头的权重
            head_score = head_weight.sum().abs().item()     # 决定是否要剪：将一头权重的和作为自信度。TODO: 这个标准可能要改
            head_scores[head] += head_score                 # 这一头的QKV自信度累加

    _, head_scores_rank = torch.Tensor(head_scores).sort()  # 按自信度排序，得到各头排名：rank[0]最不自信，rank[7]最自信

    threshold = int(num_heads * (1 - ratio) + 0.5)      # 剪掉头的数量，四舍五入
    prune_heads_index = head_scores_rank[:threshold]    # 排名的前`threshold`名，即最不自信的那些头
    head_mask = torch.ones(num_heads)                   # 作一个掩模，0为剪掉，1为留下
    head_mask[prune_heads_index] = 0                    # 这些不自信头的掩模为0，将被剪掉

    for threetimes in range(3):
        start_dim = embed_dim * threetimes  # 对于例，Q，K，V分别是从0、512、1024开始的512维。记下从哪维开始
        for head in range(num_heads):
            dim_from = start_dim + head * head_dim
            dim_to = dim_from + head_dim
            head_weight = layer.in_proj_weight[dim_from:dim_to]
            head_weight.data.mul_(head_mask[head])      # 对每个头应用掩模

    # nonzero = layer.in_proj_weight.data.abs().clone().gt(0).float().cuda()      # 统计用
    # total_nonzero = torch.sum(nonzero).int().item()         # 统计用：剪枝后剩余参数量
    total_nonzero = layer.in_proj_weight.data.nonzero().shape[0]
    head_nonzero = head_mask.nonzero().shape[0]             # 统计用：剪枝后剩余头数
    print(f"{name}: pruned to {total_nonzero / total:.6f} ({ratio}). {head_nonzero} heads left.")  # 统计

    return layer, head_mask


pruned_modules = {}

for m_name in conf:
    ratio_list = conf[m_name]
    layer = module_dict[m_name]
    module_list = []
    if isinstance(layer, nn.Linear):
        for ratio in ratio_list:
            this_layer = copy.deepcopy(layer)
            pruned_layer = prune_linear(m_name, this_layer, ratio)
            module_list.append({"layer": pruned_layer, "ratio": ratio})
    elif isinstance(layer, nn.MultiheadAttention):
        for ratio in ratio_list:
            this_layer = copy.deepcopy(layer)
            pruned_layer, head_mask = prune_attention(m_name, this_layer, ratio, EMB_SIZE, NHEAD)
            module_list.append({"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask})
    pruned_modules[m_name] = module_list


np.save(MODULE_PRUNED_FILE, pruned_modules, allow_pickle=True)


model = torch.load(MODEL_BASELINE_FILE).to(device)      # 虽然上面没改过model，但还是初始化一下放心

for module_name in pruned_modules:
    module = pruned_modules[module_name][0]["layer"]
    set_module(model, module_name, module)

torch.save(model, MODEL_PRUNED_FILE)
