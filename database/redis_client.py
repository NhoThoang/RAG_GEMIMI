import redis
import json
from config.settings import settings

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        print("▶ Redis connected")
    
    def get(self, key):
        """Get value by key"""
        return self.client.get(key)
    
    def set(self, key, value, ex=None):
        """Set key-value pair"""
        return self.client.set(key, value, ex=ex)
    
    def setex(self, key, time, value):
        """Set key-value pair with expiration"""
        return self.client.setex(key, time, value)
    
    def delete(self, key):
        """Delete key"""
        return self.client.delete(key)
    
    def exists(self, key):
        """Check if key exists"""
        return self.client.exists(key)
    
    def get_json(self, key):
        """Get JSON value by key"""
        value = self.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_json(self, key, value, ex=None):
        """Set JSON value by key"""
        return self.set(key, json.dumps(value), ex=ex) 