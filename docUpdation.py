from pymongo import MongoClient
import datetime

def connect_to_mongo():
    """Establish connection to MongoDB."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pdf_database']
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def update_mongo(file_name, summary, keywords, time_taken):
    """Update the MongoDB collection with the document metadata."""
    db = connect_to_mongo()
    
    if db is None:
        return  # Exit if MongoDB connection fails
    
    collection = db['documents']
    
    metadata = {
        'file_name': file_name,
        'summary': summary,
        'keywords': keywords,
        'time_taken': time_taken,
        'timestamp': datetime.datetime.now()
    }
    
    try:
        collection.update_one({'file_name': file_name}, {'$set': metadata}, upsert=True)
        print(f"Document {file_name} successfully updated in MongoDB.")
    except Exception as e:
        print(f"Error updating MongoDB: {e}")
