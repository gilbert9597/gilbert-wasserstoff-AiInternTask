from pymongo import MongoClient

# Connect to MongoDB (default localhost and port)
client = MongoClient("mongodb://localhost:27017/")

# Create or use a database
db = client["pdf_documents"]

# Create a collection for storing documents
collection = db["documents"]

