# Domain-Specific PDF Summarization & Keyword Extraction Pipeline

## Overview
    The **Domain-Specific PDF Summarization & Keyword Extraction Pipeline** is designed to process multiple PDF documents from a single domain. 
    The pipeline generates domain-specific summaries and extracts relevant keywords from each document. After processing, the data (summary and keywords) 
    is stored in a MongoDB database. The system efficiently handles documents of varying lengths, from short to long, ensuring that summaries and keyword extraction 
    are accurate and concise. The process is designed to handle documents in parallel, improving efficiency and throughput.

## System Requirements
    - Python 3.10
    - MongoDB (locally installed or via cloud like MongoDB Atlas)
    - Required Python libraries:
      - `pymongo`
      - `pdfplumber`
      - `requests`
      - `Pillow`
      - `pytesseract`
      - `transformers`
      - `scikit-learn`
      - `tensorflow`
      - `nltk`

## Installation
    1. **Set up Python 3.10 environment**:
       - Install Python 3.10 from [python.org](https://www.python.org/downloads/).
    
    2. **Install required dependencies**:
       Use `pip` to install the required libraries:
       ```bash
       pip install pymongo pdfplumber requests Pillow pytesseract transformers scikit-learn tensorflow nltk
       ```

    3. **MongoDB Setup**:
       - Install MongoDB locally or use [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) for a cloud-based solution.
       - If using locally, make sure MongoDB is running on `localhost:27017` or adjust the connection string in the code as needed.

## How It Works
    1. **Ingestion and Parsing**:
       - The pipeline starts by reading all PDF files from a specified folder on your desktop.
       - It handles both text-based and image-based PDFs by using OCR (Optical Character Recognition) for the latter. 
         The PDFs are parsed into chunks to make processing more efficient.

    2. **Summarization**:
       - The parsed text is summarized dynamically based on the document's length. The summarization uses a pre-trained **FLAN-T5** model for abstractive summarization, 
         which generates relevant summaries for each document.

    3. **Keyword Extraction**:
       - Keywords are extracted using the **RAKE** (Rapid Automatic Keyword Extraction) algorithm, which identifies key phrases that are domain-specific 
         and relevant to the content of the document.

    4. **MongoDB Storage**:
       - The processed data, including the document's summary, extracted keywords, and the time taken for processing, is stored in a MongoDB database.
       - The MongoDB collection is updated with this information after each document is processed.

    5. **Concurrency**:
       - The pipeline uses multithreading to process multiple PDFs concurrently, improving efficiency and ensuring that large volumes of documents are handled without crashing.

## Running the Application
    1. **MongoDB Setup**:
       - Ensure MongoDB is running on your system (or use MongoDB Atlas if you're using a cloud-based solution).

    2. **Run the script**:
       - To run the PDF processing pipeline, simply execute the Python script. You can specify the folder path containing the PDF documents in the `folder_path` variable within the script.
       ```bash
       python process_pdfs.py
       ```

    3. **Check MongoDB**:
       - After running the script, check the MongoDB database to see the processed documents, summaries, and keywords.

