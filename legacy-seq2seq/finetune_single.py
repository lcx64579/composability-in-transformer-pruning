import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from model import *
from utils import set_module
from parallel_block import *
import argparse
import os
# from evaluate_function import evaluate_BLEU

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("-r", "--pruning_rate", type=float, required=True, help="e.g. 0.5")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()
assert args.pruning_rate > 0 and args.pruning_rate <= 1, "Illegal pruning rate reported"
pruning_rate_underline = str(args.pruning_rate).replace(".", "_")

filename_tail = "_all_" + pruning_rate_underline + "_epoch" + str(args.epochs)

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
MODEL_BASELINE_FILE = "./model/baseline.pth"
MODEL_FINETUNED_PARALLEL_FILE = "./model/finetuned_parallel" + filename_tail + ".pth"
MODEL_FINETUNED_FILE = "./model/finetuned" + filename_tail + ".pth"
# MODEL_BEST_PREFIX = "./model/finetuned_all_" + pruning_rate_underline
MODULE_PRUNED_FILE = "./numpy/modules_pruned_all_" + pruning_rate_underline + ".npy"
MODULE_FINETUNED_FILE = "./numpy/modules_finetuned" + filename_tail + ".npy"
# MODULE_BEST_PREFIX = "./numpy/modules_finetuned_all_" + pruning_rate_underline
assert os.path.exists(MODULE_PRUNED_FILE), "FILE NOT EXIST"


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


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


model = torch.load(MODEL_BASELINE_FILE).to(device)
pruned_modules = np.load(MODULE_PRUNED_FILE, allow_pickle=True).item()
for param in model.parameters():
    param.requires_grad = False

param_to_finetune = []      # optimizer 使用
block_list = []             # loss function 使用

for name_pruned in pruned_modules:
    module_pruned_info = pruned_modules[name_pruned][0]     # 暂时先只用第一个块，后面再改全训练
    assert isinstance(module_pruned_info['layer'], nn.MultiheadAttention) or isinstance(module_pruned_info['layer'], nn.Linear), "Unexpected type of module"
    if isinstance(module_pruned_info['layer'], nn.MultiheadAttention):
        parallel_block = construct_parallel_block(
            model,
            name_pruned,
            module_pruned_info['layer'],
            module_type="MultiheadAttention",
            head_mask=module_pruned_info['head_mask']
        )
        set_module(model, name_pruned, parallel_block)
        param_to_finetune.append({"params": parallel_block.pruned_module.parameters()})
        block_list.append(parallel_block)
    elif isinstance(module_pruned_info['layer'], nn.Linear):
        parallel_block = construct_parallel_block(
            model,
            name_pruned,
            module_pruned_info['layer'],
            module_type="Linear"
        )
        set_module(model, name_pruned, parallel_block)
        param_to_finetune.append({"params": parallel_block.pruned_module.parameters()})
        block_list.append(parallel_block)


optimizer = torch.optim.Adam(param_to_finetune, lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
lossfn_block = nn.MSELoss()     # 剪枝块与原块之间的loss。因为形式与输出不同，所以也不能用同样的Loss函数。
train_loss_block_list = []
# eval_loss_list = []
eval_BLEU_list = []
# best_val = 999999999.0
best_BLEU = 0
best_epoch = 0

# for epoch in range(1, NUM_EPOCHS+1):
for epoch in range(1, 3):
    start_time = timer()

    model.train()
    losses = 0
    losses_block = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        # 计算所有的「剪枝块与原块之间的Loss」之和
        for block in block_list:
            loss = lossfn_block(block.out_pruned, block.out_original)
            losses_block += loss
            loss.backward()
            # 现在梯度已经产生。但剪枝掉的 Attention 头也产生了梯度，如果就此令 optimizer 更新，
            # 这会使剪枝失效。因此，把这些剪掉的头的梯度置为零。
            # 首先，判断这个block是不是MHA。如果是个Linear就不用管了。
            if block.module_type == "MultiheadAttention":
                num_heads = NHEAD
                embed_dim = EMB_SIZE
                head_dim = embed_dim // num_heads
                module = block.pruned_module
                head_mask = block.head_mask
                for threetimes in range(3):
                    start_dim = embed_dim * threetimes
                    for head in range(num_heads):
                        dim_from = start_dim + head * head_dim
                        dim_to = dim_from + head_dim
                        module.in_proj_weight.grad.data[dim_from:dim_to].mul_(head_mask[head])

        optimizer.step()

    train_loss_block = losses_block
    train_loss_block_list.append(train_loss_block.cpu())

    end_time = timer()
    print(f"Epoch: {epoch}, Train Block Loss: {train_loss_block:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")

    # 计算 val loss
    # 每隔数个epochs，做一次eval。方法：当场组装一个模型，送去eval函数里。
    # if epoch % 5 == 1:        # 每5个epochs
    # if True:                    # 每个epoch
    #     model_eval = torch.load(MODEL_BASELINE_FILE).to(device)
    #     for module_name in pruned_modules:
    #         module = pruned_modules[module_name][0]["layer"]
    #         set_module(model_eval, module_name, module)
    #     val_loss = evaluate(model_eval)
    #     eval_loss_list.append(val_loss)
    #     print(f"Evaluate Loss: {val_loss: .3f}")
    #     # save best model
    #     if val_loss < best_val:
    #         best_val = val_loss
    #         best_epoch = epoch
    #         # save best finetuned modules
    #         MODULE_BEST_FILE = MODULE_BEST_PREFIX + ".npy"
    #         np.save(MODULE_BEST_FILE, pruned_modules, allow_pickle=True)
    #         # save best finetuned model
    #         MODEL_BEST_FILE = MODEL_BEST_PREFIX + ".pth"
    #         model2 = torch.load(MODEL_BASELINE_FILE).to(device)
    #         for module_name in pruned_modules:
    #             module = pruned_modules[module_name][0]["layer"]
    #             set_module(model2, module_name, module)
    #         torch.save(model2, MODEL_BEST_FILE)
    #         print("Best model updated")

    # 剪枝块的目的是尽可能模仿原块的信息，而非得到最高的BLEU。因此只要Train Loss最低即可，下面的BLEU部分不需要了。
#     # 计算 BLEU
#     if True:
#         model_eval = torch.load(MODEL_BASELINE_FILE).to(device)
#         for module_name in pruned_modules:
#             module = pruned_modules[module_name][0]["layer"]
#             set_module(model_eval, module_name, module)
#         BLEU_score = evaluate_BLEU(model_eval)
#         eval_BLEU_list.append(BLEU_score)
#         print(f"BLEU score: {BLEU_score: .4f}")
#         if BLEU_score > best_BLEU:
#             best_BLEU = BLEU_score
#             best_epoch = epoch
#             MODULE_BEST_FILE = MODULE_BEST_PREFIX + ".npy"
#             np.save(MODULE_BEST_FILE, pruned_modules, allow_pickle=True)
#             MODEL_BEST_FILE = MODEL_BEST_PREFIX + ".pth"
#             # model2 = torch.load(MODEL_BASELINE_FILE).to(device)
#             # for module_name in pruned_modules:
#             #     module = pruned_modules[module_name][0]["layer"]
#             #     set_module(model2, module_name, module)
#             torch.save(model_eval, MODEL_BEST_FILE)
#             print("Best model updated")

# # print(f"Best model at Epoch: {best_epoch}, Evaluate loss: {best_val:.3f}")
# print(f"Best model at Epoch: {best_epoch}, BLEU score: {best_BLEU:.4f}")
# MODULE_BEST_FILE_RENAME = MODULE_BEST_PREFIX + "_epoch" + str(best_epoch) + ".npy"
# MODEL_BEST_FILE_RENAME = MODEL_BEST_PREFIX + "_epoch" + str(best_epoch) + ".pth"
# if os.path.exists(MODULE_BEST_FILE_RENAME):
#     os.remove(MODULE_BEST_FILE_RENAME)
# if os.path.exists(MODEL_BEST_FILE_RENAME):
#     os.remove(MODEL_BEST_FILE_RENAME)
# os.rename(MODULE_BEST_FILE, MODULE_BEST_FILE_RENAME)
# os.rename(MODEL_BEST_FILE, MODEL_BEST_FILE_RENAME)

# Check the result
for i in range(0, len(train_loss_block_list)):
    train_loss_block_list[i] = train_loss_block_list[i].cpu().item()

x = range(1, NUM_EPOCHS + 1)
plt.figure(num=1)
plt.plot(x, train_loss_block_list, label="train_block")
plt.xlabel("Epochs")
plt.ylabel("Block Loss Sum")
plt.savefig("finetune_train_loss.png")

# plt.figure(num=2)
# plt.plot(x, eval_loss_list, label="eval")
# plt.xlabel("Epochs")
# plt.ylabel("Val Loss")
# plt.savefig("loss_finetune_val.png")

# plt.figure(num=2)
# plt.plot(x, eval_BLEU_list, label="eval BLEU")
# plt.xlabel("Epochs")
# plt.ylabel("BLEU score")
# plt.savefig("finetune_BLEU.png")

SAVE_PARALLEL_MODEL = False
if SAVE_PARALLEL_MODEL:
    torch.save(model, MODEL_FINETUNED_PARALLEL_FILE)
    # np.save("losses_block.npy", train_loss_block_list)
    print("Model (paralleled) saved.")

SAVE_BLOCK = True
if SAVE_BLOCK:
    np.save(MODULE_FINETUNED_FILE, pruned_modules, allow_pickle=True)
    print("Pruned blocks saved.")

SAVE_EVALUATE_MODEL = True
if SAVE_EVALUATE_MODEL:
    model2 = torch.load(MODEL_BASELINE_FILE).to(device)

    for module_name in pruned_modules:
        module = pruned_modules[module_name][0]["layer"]
        set_module(model2, module_name, module)

    torch.save(model2, MODEL_FINETUNED_FILE)
    print("Model for evaluation saved.")
