import torch
import torch.nn.utils.prune as prune
from utils import type_of_module, type_of_model
import numpy as np
import random
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="model file (.pth)")
parser.add_argument("-t", "--tokenizer", type=str, default="./tokenizer/t5-small.pth", help="tokenizer file (.pt)")
args = parser.parse_args()

PATH_TO_MODEL = args.model
assert os.path.exists(PATH_TO_MODEL), "model file not found"
PATH_TO_TOKENIZER = args.tokenizer
assert os.path.exists(PATH_TO_TOKENIZER), "tokenizer file not found"

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
sentence = "translate English to German: A group of people are standing in front of an auditorium."
input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
times = []
model.eval()
# Warm up
for i in range(100):
    model.generate(input_ids, max_new_tokens=100)
# Actual inference
for i in range(500):
    start = datetime.now()
    model.generate(input_ids, max_new_tokens=100)
    end = datetime.now()
    deltatime_by_microseconds = (end - start).microseconds
    times.append(deltatime_by_microseconds)
average_time = np.mean(times)
print(f'Average inference time (seconds): {average_time}')
