import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Disable OneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to summarize text with FLAN-T5-small, custom max token length, and temperature
def summarize_text(text_chunks, max_token_length=350, temperature=0.07):
    # Define the FLAN-T5-small model
    model_name = "google/flan-t5-small"
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summaries = []
    for text_segment in text_chunks:
        # Tokenize the input text
        inputs = tokenizer(text_segment, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate the summary with custom parameters
        summary_ids = model.generate(
            inputs.input_ids,
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
        summaries.append(summary)
    
    return summaries
