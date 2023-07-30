import torch
import torch.nn.utils.prune as prune
from utils import type_of_module, type_of_model, get_embed_dim, get_num_heads
import numpy as np
import random
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="model file (.pth)")
parser.add_argument("-t", "--tokenizer", type=str, default="./tokenizer/t5-small.pth", help="tokenizer file (.pt)")
parser.add_argument("--warmup", type=int, default=100, help="number of warmup iterations")
parser.add_argument("--inference", type=int, default=500, help="number of inference iterations")
args = parser.parse_args()

PATH_TO_MODEL = args.model
assert os.path.exists(PATH_TO_MODEL), "model file not found"
PATH_TO_TOKENIZER = args.tokenizer
assert os.path.exists(PATH_TO_TOKENIZER), "tokenizer file not found"
NUM_WARMUP = args.warmup
NUM_INFERENCES = args.inference
assert NUM_WARMUP > 0, "number of warmup iterations must be greater than 0"
assert NUM_INFERENCES > 0, "number of inference iterations must be greater than 0"

TYPE_OF_MODEL = type_of_model(PATH_TO_MODEL)
MANUAL_SEED = 42

# Set random seed
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = torch.load(PATH_TO_MODEL).to(device)
tokenizer = torch.load(PATH_TO_TOKENIZER)

# print some related information
EMBED_DIM = get_embed_dim(TYPE_OF_MODEL, model)
NUM_HEADS = get_num_heads(TYPE_OF_MODEL, model)
print(f"Embedding dimension: {EMBED_DIM}")
print(f"Number of heads: {NUM_HEADS}")

# Inference memory overhead
# But first, let's remove all the buffers and apply the pruning mask
for name, module in model.named_modules():
    if type_of_module(TYPE_OF_MODEL, name, module) == "Linear" and hasattr(module, 'weight_orig'):
        prune.remove(module, 'weight')
        if hasattr(module, 'bias') and module.bias is not None:
            prune.remove(module, 'bias')
# Then we calculate the memory footprint
memory_footprint = model.get_memory_footprint(return_buffers=True)
print(f'Memory footprint (Bytes): {memory_footprint}')

# Inference time
sentence = ""
if TYPE_OF_MODEL == "t5":
    sentence = "translate English to German: A group of people are standing in front of an auditorium."
elif TYPE_OF_MODEL == "distilbert":
    sentence = "I've been waiting for this movie my whole life. It was so good I cried."
input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
times = []
model.eval()

if TYPE_OF_MODEL == "t5":
    # Warm up
    for i in range(NUM_WARMUP):
        model.generate(input_ids, max_new_tokens=100)
    # Actual inference
    for i in range(NUM_INFERENCES):
        start = datetime.now()
        model.generate(input_ids, max_new_tokens=100)
        end = datetime.now()
        deltatime_by_microseconds = (end - start).microseconds
        times.append(deltatime_by_microseconds)
elif TYPE_OF_MODEL == "distilbert":
    # Warm up
    for i in range(NUM_WARMUP):
        model(input_ids)
    # Actual inference
    for i in range(NUM_INFERENCES):
        start = datetime.now()
        model(input_ids)
        end = datetime.now()
        deltatime_by_microseconds = (end - start).microseconds
        times.append(deltatime_by_microseconds)
average_time = np.mean(times)
print(f'Average inference time (seconds): {average_time}')
