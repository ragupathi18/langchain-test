from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"  # You can choose the model size (e.g., "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set up prompt text
prompt = "Once upon a time"

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
