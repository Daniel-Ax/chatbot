from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name=r"C:\Users\finyw\chat_bot\model_train\tuned_gpt_on_books_v2"
# Modell betöltése
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Pad token beállítása
tokenizer.pad_token = tokenizer.eos_token

# Tesztkérdés küldése
question = "What are the three primary colors?"
inputs = tokenizer.encode(question, return_tensors="pt", padding=True, truncation=True)

# attention_mask és pad_token_id beállítása
attention_mask = inputs.ne(tokenizer.pad_token_id).float()

# Modell válasz generálása
outputs = model.generate(inputs, max_length=50, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
