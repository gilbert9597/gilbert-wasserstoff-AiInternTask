import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Disable OneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load FLAN-T5-small model and tokenizer
model_name = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings for text chunks
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get the embeddings (mean pooling of the last hidden state)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return np.array(embeddings)

# Function to perform keyword extraction using clustering
def extract_keywords(text_chunks, num_keywords=5, num_clusters=5):
    # Step 1: Generate embeddings for each chunk
    embeddings = generate_embeddings(text_chunks)
    
    # Step 2: Perform KMeans clustering on the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    
    # Step 3: Extract top keywords for each cluster
    cluster_keywords = []
    for i in range(num_clusters):
        # Find the chunks belonging to the current cluster
        cluster_indices = [index for index, label in enumerate(labels) if label == i]
        
        # Collect the corresponding text for each chunk in the cluster
        cluster_text = [text_chunks[index] for index in cluster_indices]
        
        # Combine the text in the cluster into a single string
        combined_text = ' '.join(cluster_text)
        
        # Use TF-IDF to find the top keywords within the cluster
        vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
        X = vectorizer.fit_transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Add the top keywords for the cluster
        cluster_keywords.append(feature_names.tolist())
    
    return cluster_keywords
