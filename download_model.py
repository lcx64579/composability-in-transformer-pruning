# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
import torch

OUTPUT_DIR = "./model/t5-small"
OUTPUT_MODEL_PTH = "./model/t5-small.pth"

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
# print(f"Huggingface Model & Tokenizer saved to: {OUTPUT_DIR}")

torch.save(model, OUTPUT_MODEL_PTH)
print(f".pth Model saved to: {OUTPUT_MODEL_PTH}")

print(type(model))
# print(model)
