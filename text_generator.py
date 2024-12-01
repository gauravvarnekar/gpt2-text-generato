from transformers import pipeline

# Load pre-trained GPT model
generator = pipeline(
    'text-generation',
    model='gpt2',
    truncation=True,  # Explicitly enable truncation
    pad_token_id=50256  # Set pad_token_id to GPT-2's EOS token
)

# Generate text based on a prompt
prompt = "Once upon a time in a world powered by AI"
results = generator(prompt, max_length=50, num_return_sequences=1)

# Beautify and display the result
print("\n" + "=" * 50)
print("🌟 Prompt:")
print(f"   {prompt}")
print("\n" + "-" * 50)
print("💡 Generated Text:")
print(f"   {results[0]['generated_text']}")
print("=" * 50 + "\n")





