import torch
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
memory_footprint = model.get_memory_footprint()
print(f'Memory footprint (Bytes): {memory_footprint}')

# Inference time
sentence = "translate English to German: A group of people are standing in front of an auditorium."
input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
times = []
for i in range(10):
    start = datetime.now()
    model.generate(input_ids)
    end = datetime.now()
    deltatime_by_microseconds = (end - start).microseconds
    times.append(deltatime_by_microseconds)
average_time = np.mean(times)
print(f'Average inference time (seconds): {average_time}')
