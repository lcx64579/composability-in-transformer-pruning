import json
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
from evaluate_function import evaluate_BLEU
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modules_file", type=str, required=True, help="Path to file of pruned modules.")
parser.add_argument("-o", "--original_model_file", type=str, required=True, default="./model/baseline.pth", help="Path to file of original model.")
parser.add_argument("-c", "--conf_file", type=str, required=True, default="./conf.json", help="Path to configuration file.")
parser.add_argument("-n", "--finetune_nth_config", type=int, required=True, help="This program finetunes ONLY ONE config at each running. Specify the `n`-th config in conf.json to be worked on. `n` starts at 0.")
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

filename_tail = "_epoch" + str(args.epochs)


BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
MODEL_BASELINE_FILE = args.original_model_file
MODEL_RETRAINED_FILE = "./model/retrained" + filename_tail + ".pth"
MODULE_FINETUNED_FILE = args.modules_file
MODEL_BEST_PREFIX = "./model/retrained_best"
MODEL_BEST_FILE = "./model/retrained_best_tmp.pth"
CONFIG_FILE = args.conf_file
NTH_CONFIG = args.finetune_nth_config

assert os.path.exists(MODEL_BASELINE_FILE), "ORIGINAL MODEL FILE NOT EXISTED"
assert os.path.exists(MODULE_FINETUNED_FILE), "FINETUNED MODULES FILE NOT EXISTED"
assert os.path.exists(CONFIG_FILE), "CONFIG FILE NOT EXISTED"


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
finetuned_modules = np.load(MODULE_FINETUNED_FILE, allow_pickle=True).item()
conf_file = open(CONFIG_FILE, 'r')
conf = json.load(conf_file)
model_number = len(conf)


def find_finetuned_module_by_name_and_ratio(finetuned_modules: dict, name: str, ratio: float) -> tuple[nn.Module, list]:
    assert name in finetuned_modules, f"MODULE {name} NOT FOUND"
    info_module_list = finetuned_modules[name]      # [{"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}, ...]
    for info_module in info_module_list:            # {"layer": pruned_layer, "ratio": ratio, "head_mask": head_mask}
        if info_module["ratio"] == ratio:
            if isinstance(info_module["layer"], nn.MultiheadAttention):
                return info_module["layer"], info_module["head_mask"]
            else:
                return info_module["layer"], None
    return None, None


# for name in finetuned_modules:
#     module = finetuned_modules[name][0]['layer']
#     if isinstance(module, nn.MultiheadAttention):
#         module.head_mask = finetuned_modules[name][0]['head_mask']
#     set_module(model, name, module)

# for conf_model_ith in conf:
#     for name in conf_model_ith:
#         if conf_model_ith[name] == 1:       # 若为1则该块使用原版，也即不需要对模型上的该块做任何事
#             continue
#         else:
#             for name in finetuned_modules:

conf_nth = conf[NTH_CONFIG]
print("-------------------------")
print("Working on configuration:")
print(conf_nth)
print("-------------------------")

for name in conf_nth:
    ratio = conf_nth[name]
    if ratio == 1:      # 若为1则该块使用原版，也即不需要对模型上的该块做任何事
        continue
    assert name in finetuned_modules, "A SPECIFIED PRUNED RATIO OF MODULE DOES NOT EXIST"
    module, head_mask = find_finetuned_module_by_name_and_ratio(finetuned_modules, name, ratio)
    assert module is not None, "A SPECIFIED PRUNED RATIO OF MODULE DOES NOT EXIST"
    if isinstance(module, nn.MultiheadAttention):
        module.head_mask = head_mask
    set_module(model, name, module)


loss_train_history = []
loss_val_history = []
BLEU_history = []
best_BLEU = 0
best_epoch = 0

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


BLEU_without_retrain = evaluate_BLEU(model)
print(f"BLEU score without retraining: {BLEU_without_retrain:.4f}")
BLEU_history.append(BLEU_without_retrain)


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()

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

        # 清空Attention的梯度
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention) and hasattr(module, "head_mask"):
                num_heads = NHEAD
                embed_dim = EMB_SIZE
                head_dim = embed_dim // num_heads
                head_mask = module.head_mask
                for threetimes in range(3):
                    start_dim = embed_dim * threetimes
                    for head in range(num_heads):
                        dim_from = start_dim + head * head_dim
                        dim_to = dim_from + head_dim
                        module.in_proj_weight.grad.data[dim_from:dim_to].mul_(head_mask[head])

        optimizer.step()
        losses += loss.item()

    train_loss = losses / len(list(train_dataloader))

    end_time = timer()
    val_loss = evaluate(model)
    BLEU_score = evaluate_BLEU(model)
    loss_train_history.append(train_loss)
    loss_val_history.append(val_loss)
    BLEU_history.append(BLEU_score)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, BLEU: {BLEU_score:.4f} "f"Epoch time = {(end_time - start_time):.3f}s"))
    if BLEU_score > best_BLEU:
        best_BLEU = BLEU_score
        best_epoch = epoch
        torch.save(model, MODEL_BEST_FILE)
        print("Best model updated")


print(f"Best model at Epoch: {best_epoch}, BLEU: {best_BLEU:.4f}")
MODEL_BEST_FILE_RENAME = MODEL_BEST_PREFIX + "_epoch" + str(best_epoch) + ".pth"
if os.path.exists(MODEL_BEST_FILE_RENAME):
    os.remove(MODEL_BEST_FILE_RENAME)
os.rename(MODEL_BEST_FILE, MODEL_BEST_FILE_RENAME)

# Show the result
x = range(1, len(loss_train_history) + 1)
plt.figure(1)
plt.plot(x, loss_train_history, label='train')
plt.plot(x, loss_val_history, label='val')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("retrain_loss.png")

x2 = range(0, len(BLEU_history))
plt.figure(2)
plt.plot(x2, BLEU_history, label='BLEU')
plt.xlabel("Epochs")
plt.ylabel("BLEU")
plt.savefig("retrain_BLEU.png")

torch.save(model, MODEL_RETRAINED_FILE)
print(f"Finetuned module saved to {MODEL_RETRAINED_FILE}")
