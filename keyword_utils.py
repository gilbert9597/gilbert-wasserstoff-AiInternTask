from rake_nltk import Rake

# Function to perform keyword extraction using RAKE
def extract_keywords(text_chunks, num_keywords=5):
    rake = Rake()  # Initialize RAKE for keyword extraction
    # Join the text chunks into a single string
    combined_text = ' '.join(text_chunks)
    rake.extract_keywords_from_text(combined_text)
    
    # Get the ranked keywords
    ranked_keywords = rake.get_ranked_phrases_with_scores()
    
    # Return top 'num_keywords' keywords
    return ranked_keywords[:num_keywords]
