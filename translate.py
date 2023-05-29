from transformers import T5Tokenizer, T5ForConditionalGeneration

PATH_TO_MODEL = "./model/t5-small"
tokenizer = T5Tokenizer.from_pretrained(PATH_TO_MODEL)
model = T5ForConditionalGeneration.from_pretrained(PATH_TO_MODEL)

# Google T5 only support translation from English to French, German and Romanian
# Format: "translate English to French: How old are you?". The output should be: "Wie alt sind Sie?"
sample_text = "translate English to German: A group of people stand in front of an auditorium."

input_ids = tokenizer.encode(sample_text, return_tensors='pt')       # `return_tensors='pt'` returns a tensor, not a list
outputs = model.generate(input_ids=input_ids, max_new_tokens=50)     # `max_new_tokens=50` controls the length of the output. This eliminates an unwanted warning.
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)     # `skip_special_tokens=Tru`e will not output special tokens (e.g. <pad>, <unk>, <eos>, <sos>, etc.)
print(decoded)
