import os
import time
import pickle
import concurrent.futures
from parsing import parse_pdf
from summarization import summarize_text
from keyword_utils import extract_keywords
from docUpdation import update_mongo
import psutil
import random
import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Constants and configuration
CHECKPOINT_FILE = 'checkpoint.pkl'  # Pickle file to store progress/checkpoints
folder_path = r'E:\Project\PYCHARM\VS\PDF'  # Folder path containing the PDFs
model_name = "google/flan-t5-small"  # FLAN-T5-small model name
METRICS_FILE = "resource_utilization.txt"  # File to save resource utilization metrics

# Resource utilization metrics initialization
cpu_usage_per_pdf = []
memory_usage_per_pdf = []
concurrent_tasks = []

# Load checkpoint progress
def load_checkpoint():
    """Load the checkpoint progress."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return set()

# Save checkpoint progress
def save_checkpoint(processed_files):
    """Save the checkpoint progress by pickling the processed files."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(processed_files, f)

# Function to write resource utilization metrics to a file
def save_metrics_to_file(file_path, time_taken, cpu_usage, memory_usage):
    """Save the performance metrics in the new report format."""
    with open(METRICS_FILE, 'a') as f:
        f.write("Performance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"File: {file_path}\n")
        f.write(f"Time Taken: {time_taken:.2f} seconds\n")
        f.write(f"CPU Usage: {cpu_usage:.2f} %\n")
        f.write(f"Memory Usage: {memory_usage:.2f} %\n")
        f.write("-" * 40 + "\n")

# Function to process a single PDF file
def process_pdf(file_path, processed_files, tokenizer, model):
    try:
        # Check if the PDF has already been processed
        if file_path in processed_files:
            print(f"{file_path} already processed.")
            return
        
        # Parse PDF into chunks
        print(f"Parsing PDF: {file_path}")
        text_chunks = parse_pdf(file_path)
        print(f"PDF parsed: {file_path}")

        # Concatenate all text chunks into one long string for summarization
        entire_text = " ".join(text_chunks)
        
        # Record start time for speed metric
        start_time = time.time()

        # Record CPU and memory usage before processing
        cpu_before = psutil.cpu_percent(interval=1)
        memory_before = psutil.virtual_memory().percent
        
        # Run keyword extraction and summarization concurrently
        summary_future = concurrent.futures.ThreadPoolExecutor().submit(summarize_text, entire_text, tokenizer, model)
        keyword_future = concurrent.futures.ThreadPoolExecutor().submit(extract_keywords, text_chunks)
        
        summary = summary_future.result()
        keywords = keyword_future.result()

        # Calculate time taken for processing
        time_taken = time.time() - start_time
        
        # Record CPU and memory usage after processing
        cpu_after = psutil.cpu_percent(interval=1)
        memory_after = psutil.virtual_memory().percent
        
        # Store metrics for the current PDF
        cpu_usage_per_pdf.append(cpu_after - cpu_before)
        memory_usage_per_pdf.append(memory_after - memory_before)

        # Save the resource utilization metrics to a file in the new format
        save_metrics_to_file(file_path, time_taken, cpu_after - cpu_before, memory_after - memory_before)

        print(f"Summary and Keywords extracted for {file_path}.")
        
        # Update MongoDB with summarized data and keywords
        update_mongo(os.path.basename(file_path), summary, keywords, time_taken)
        print(f"MongoDB updated for {file_path}")

        # Add the file to processed files set
        processed_files.add(file_path)

        # Periodically save checkpoint after processing a PDF
        save_checkpoint(processed_files)
        print(f"Processed {file_path}.")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Main function to start processing
def main():
    # Load checkpoint and get processed files
    processed_files = load_checkpoint()
    
    # List all PDFs in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    print(f"Starting PDF processing for {len(pdf_files)} files.")
    
    # Initialize tokenizer and model for summarization
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Use ThreadPoolExecutor for concurrent PDF processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Add all files for processing to the executor
        for pdf_file in pdf_files:
            file_path = os.path.join(folder_path, pdf_file)
            executor.submit(process_pdf, file_path, processed_files, tokenizer, model)

    print(f"Processing completed for {len(pdf_files)} PDFs.")

if __name__ == "__main__":
    main()
