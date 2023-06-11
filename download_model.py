# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('model_type', type=str, help='model type')
args = parser.parse_args()

assert args.model_type in ['t5', 'distilbert'], "Model type not supported!"

OUTPUT_MODEL_PTH = ''
OUTPUT_TOKENIZER_PTH = ''
model = None
tokenizer = None


if args.model_type == 't5':
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    # OUTPUT_DIR = "./model/t5-small"
    OUTPUT_MODEL_PTH = "./model/t5-small.pth"
    OUTPUT_TOKENIZER_PTH = "./tokenizer/t5-small.pth"

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # model.save_pretrained(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)
    # print(f"Huggingface Model & Tokenizer saved to: {OUTPUT_DIR}")

    torch.save(model, OUTPUT_MODEL_PTH)
    torch.save(tokenizer, OUTPUT_TOKENIZER_PTH)

elif args.model_type == 'distilbert':
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    OUTPUT_MODEL_PTH = './model/distilbert-imdb.pth'
    OUTPUT_TOKENIZER_PTH = './tokenizer/distilbert-imdb.pth'
    model = DistilBertForSequenceClassification.from_pretrained('lvwerra/distilbert-imdb')
    tokenizer = DistilBertTokenizer.from_pretrained('lvwerra/distilbert-imdb')
    torch.save(model, OUTPUT_MODEL_PTH)
    torch.save(tokenizer, OUTPUT_TOKENIZER_PTH)


print(f'Model type: {type(model)}')
print(f".pth Model saved to: {OUTPUT_MODEL_PTH}")
print(f'.pth Tokenizer saved to: {OUTPUT_TOKENIZER_PTH}')

# print(model)
