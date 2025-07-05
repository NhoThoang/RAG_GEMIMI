from pymongo import MongoClient as PyMongoClient
from config.settings import settings

class MongoClient:
    def __init__(self):
        self.client = PyMongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.MONGODB_DATABASE]
        self.collection = self.db[settings.MONGODB_COLLECTION]
        print("▶ MongoDB connected")
    
    def insert_one(self, document):
        """Insert a single document"""
        return self.collection.insert_one(document)
    
    def find(self, filter_dict):
        """Find documents"""
        return self.collection.find(filter_dict)
    
    def find_one(self, filter_dict):
        """Find a single document"""
        return self.collection.find_one(filter_dict)
    
    def update_one(self, filter_dict, update_dict):
        """Update a single document"""
        return self.collection.update_one(filter_dict, update_dict)
    
    def delete_one(self, filter_dict):
        """Delete a single document"""
        return self.collection.delete_one(filter_dict) 