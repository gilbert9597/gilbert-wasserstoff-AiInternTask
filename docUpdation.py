import datetime
from pymongo import MongoClient

# Function to connect to MongoDB
def connect_to_mongo():
    """Establish connection to MongoDB."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pdf_database']
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# Function to update MongoDB with metadata
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
        # Retry logic for MongoDB updates
        for attempt in range(3):
            try:
                collection.update_one({'file_name': file_name}, {'$set': metadata}, upsert=True)
                print(f"Document {file_name} successfully updated in MongoDB.")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Error updating MongoDB, retrying... ({attempt + 1}/3)")
                    time.sleep(random.uniform(1, 3))  # Exponential backoff
                else:
                    print(f"Failed to update MongoDB after 3 attempts: {e}")
    except Exception as e:
        print(f"Error updating MongoDB: {e}")
