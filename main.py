import os
import time
import pickle
import concurrent.futures
from parsing import parse_pdf
from summarization import summarize_text
from keyword_utils import extract_keywords
from docUpdation import update_mongo

CHECKPOINT_FILE = 'checkpoint.pkl'  # Pickle file to store progress/checkpoints

def load_checkpoint():
    """Load the checkpoint progress."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return set()

def save_checkpoint(processed_files):
    """Save the checkpoint progress by pickling the processed files."""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(processed_files, f)

def process_pdf(file_path, processed_files):
    try:
        # Check if the PDF has already been processed
        if file_path in processed_files:
            print(f"{file_path} already processed.")
            return
        
        # Parse PDF into chunks
        text_chunks = parse_pdf(file_path)
        
        # Summarize text in chunks
        start_time = time.time()
        summaries = summarize_text(text_chunks)
        
        # Extract keywords from chunks
        keywords = extract_keywords(text_chunks)
        
        # Update MongoDB with summarized data and keywords
        time_taken = time.time() - start_time
        update_mongo(os.path.basename(file_path), summaries, keywords, time_taken)

        # Add the file to processed files set
        processed_files.add(file_path)

        # Periodically save checkpoint after processing a PDF
        save_checkpoint(processed_files)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_pdfs_in_folder(folder_path):
    processed_files = load_checkpoint()  # Load checkpoint progress

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(folder_path, file_name)
                # Submit tasks to process PDFs in parallel
                futures.append(executor.submit(process_pdf, file_path, processed_files))

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensures exceptions in threads are raised

if __name__ == '__main__':
    folder_path = r'E:\Project\PYCHARM\VS\PDF'  # Folder path containing the PDFs
    process_pdfs_in_folder(folder_path)
