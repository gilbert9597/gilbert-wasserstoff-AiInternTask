import pdfplumber
import pytesseract
from PIL import Image

# Function to parse the PDF file and extract text chunks
def parse_pdf(file_path, chunk_size=500, overlap=100):
    try:
        # Open the PDF file
        c = 0
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                print("page ", str(c))
                c += 1
                if page_text:
                    text += page_text
                else:
                    # Perform OCR if page is image-based
                    image = page.to_image()
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text

        # If no text is extracted after OCR
        if not text.strip():
            print("Warning: No text found in PDF, even after OCR.")
            return []

        # Chunk the text into manageable parts with overlap
        text_chunks = chunk_text(text, chunk_size, overlap)
        return text_chunks
    
    except Exception as e:
        print(f"An error occurred while parsing the PDF: {e}")
        return []

# Function to chunk text into smaller pieces with overlap
def chunk_text(text, chunk_size, overlap):
    sentences = text.split('. ')
    
    # Create chunks with sliding window
    chunks = []
    current_chunk = ''
    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            # Start the next chunk with overlap
            current_chunk = '. '.join(sentences[max(i - overlap, 0):i]) + '. '
    
    # Add any remaining text as a final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
