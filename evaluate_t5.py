import argparse
import os
import re
import torch
from tqdm import tqdm
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='model name')
parser.add_argument('-t', '--tokenizer', type=str, default="./tokenizer/t5-small.pth", help='tokenizer file')
args = parser.parse_args()

PATH_TO_MODEL = args.model
PATH_TO_TOKENIZER = args.tokenizer
assert os.path.exists(PATH_TO_MODEL), "Model file does not exist."
assert os.path.exists(PATH_TO_TOKENIZER), "Tokenizer file does not exist."
BATCH_SIZE = 64


def translate(model, tokenizer, sentence_batch):
    # input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)        # For single sentence
    encoding = tokenizer.batch_encode_plus(sentence_batch, return_tensors='pt', padding=True, truncation=True).to(device)       # For batch
    input_ids = encoding['input_ids']
    # attention_mask = encoding['attention_mask']
    output = model.generate(
        input_ids=input_ids,
        # attention_mask=attention_mask,
        max_new_tokens=50,
    )
    result = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return result


# Load model
model = torch.load(PATH_TO_MODEL).to(device)
tokenizer = torch.load(PATH_TO_TOKENIZER)

# Load dataset
test_iter = Multi30k(split='test', language_pair=('en', 'de'))
testset = list(test_iter)
# Preprocess dataset. T5 do machine translation by attaching this prefix to the source sentence.
prefix = 'translate English to German: '
testset = [{'src': prefix + en, 'tgt': de} for en, de in testset]

testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


example_result = translate(model, tokenizer, ["translate English to German: A group of people stand in front of an auditorium."])
print("====================================")
print("Source | A group of people stand in front of an auditorium.")
print("Target | Eine Gruppe von Menschen steht vor einem Iglu.")
print(f"Result | {example_result[0]}")
print("====================================")


references = []
candidates = []
testloader_length = len(testloader)

progress_bar = tqdm(enumerate(testloader, 0), total=testloader_length, desc=f"Evaluating: ")
for i, batch in progress_bar:
    src, tgt = batch['src'], batch['tgt']

    candidate = translate(model, tokenizer, src)

    for c, t in zip(candidate, tgt):
        c = re.sub(r"^\s*", r"", c)  # Remove leading spaces
        c = re.sub(r"\s*$", r"", c)  # Remove trailing spaces
        t = t.split(' ')
        c = c.split(' ')
        references.append([t])      # Must append a list because the bleu_score function expects a list of references. We have only 1 reference.
        candidates.append(c)


print("====================================")
example_number = 5
example_cand = [' '.join(token) for token in candidates[:example_number]]
example_ref = [' '.join(token[0]) for token in references[:example_number]]
for i in range(example_number):
    print("Target |" + example_ref[i])
    print("Result |" + example_cand[i])
    print()
print("====================================")


score = bleu_score(candidates, references)
score_100 = score * 100
print(f"{score} ({score_100:.4f})")


# Examples：
# "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche. = Two young, White males are outside near many bushes."
# "Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem. = Several men in hard hats are operating a giant pulley system."
# "Ein kleines Mädchen klettert in ein Spielhaus aus Holz. = A little girl climbing into a wooden playhouse."
# "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster. = A man in a blue shirt is standing on a ladder cleaning a window."
# "Zwei Männer stehen am Herd und bereiten Essen zu. = Two men are at the stove preparing food."
