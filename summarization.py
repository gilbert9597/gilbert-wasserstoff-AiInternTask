from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to summarize the entire text of a PDF as a single summary
def summarize_text(text, tokenizer, model, max_token_length=350, temperature=0.07):
    # Ensure text length does not exceed model's input size
    max_input_length = 512  # Default max length for most transformer models
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_input_length)
    
    # Generate the summary with custom parameters
    summary_ids = model.generate(
        inputs['input_ids'],  # Pass the input_ids for the model to generate summary
        max_length=max_token_length,
        min_length=50,
        do_sample=True,
        temperature=temperature,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Summary: {summary}")
    return summary
