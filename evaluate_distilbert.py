import argparse
import os
import re
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='model name')
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/distilbert-imdb.pth", help='tokenizer file')
args = parser.parse_args()

PATH_TO_MODEL = args.model
PATH_TO_TOKENIZER = args.tokenizer
assert os.path.exists(PATH_TO_MODEL), "Model file does not exist."
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file does not exist."
BATCH_SIZE = 64


def predict(model, tokenizer, text):
    # encoding = tokenizer.batch_encode_plus(sentence_batch, return_tensors='pt', padding=True, truncation=True).to(device)       # For batch
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)       # For batch
    output = model(**encoding)
    prediction = [np.argmax(logit, axis=-1) for logit in output.logits.detach().cpu().numpy()]
    return prediction


# Load model
model = torch.load(PATH_TO_MODEL).to(device)
tokenizer = torch.load(PATH_TO_TOKENIZER)

# Load dataset
testset = load_dataset('imdb', split='test')
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
testloader_length = len(testloader)


correct = 0
progress_bar = tqdm(enumerate(testloader, 0), total=testloader_length, desc=f"Evaluating: ")
for i, batch in progress_bar:
    src, tgt = batch['text'], batch['label']

    prediction = predict(model, tokenizer, src)

    correct += np.sum(prediction == tgt.numpy())
    progress_bar.set_postfix({'accuracy': correct / ((i+1) * BATCH_SIZE)})


accuracy = correct / len(testset)
print(f"Accuracy: {accuracy:.4f}")
