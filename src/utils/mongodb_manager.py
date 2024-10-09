# /src/utils/mongodb_manager.py

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


class MongoDBManager:
    """Class to manage MongoDB operations."""

    def __init__(self):
        # Initialize config
        self.config = ConfigLoader()
        # Initialize logger
        self.logger = setup_logger(filename=__name__)
        # Get MongoDB configuration
        mongodb_config = self.config.get('mongodb', {})
        hosts = mongodb_config.get('HOST', ['localhost'])
        port = mongodb_config.get('PORT', 27017)
        db_name = mongodb_config.get('DB_NAME', 'test')
        self.collection_name = mongodb_config.get('COLLECTION', 'collection')
        # Try to connect to one of the hosts
        self.client = None
        self.connected_host = None
        for host in hosts:
            try:
                self.logger.info(f"Attempting to connect to MongoDB at {host}:{port}")
                self.client = MongoClient(
                    host=host,
                    port=port,
                    serverSelectionTimeoutMS=5000  # 5 seconds timeout
                )
                # Ping the server to check if it's available
                self.client.admin.command('ping')
                self.logger.info(f"Successfully connected to MongoDB at {host}:{port}")
                self.connected_host = host
                break
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                self.logger.error(f"Failed to connect to MongoDB at {host}:{port} - {e}")
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred while connecting to MongoDB at {host}:{port} - {e}")
        if self.client is None:
            self.logger.critical("Could not connect to any MongoDB host.")
            raise ConnectionError("Could not connect to any MongoDB host.")
        self.db = self.client[db_name]
        self.logger.debug(f"Using database: {db_name}")

    def insert_documents(self, collection_name, documents):
        """Insert multiple documents into a collection."""
        try:
            collection = self.db[collection_name]
            result = collection.insert_many(documents)
            self.logger.debug(f"Inserted document IDs: {result.inserted_ids}")
            self.logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
            return result.inserted_ids
        except Exception as e:
            self.logger.exception(f"An error occurred while inserting documents into {collection_name}: {e}")
            raise

    def insert_document(self, collection_name, document):
        """Insert a single document into a collection."""
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            self.logger.debug(f"Inserted document ID: {result.inserted_id}")
            self.logger.info(f"Inserted document with _id: {result.inserted_id} into {collection_name}")
            return result.inserted_id
        except Exception as e:
            self.logger.exception(f"An error occurred while inserting a document into {collection_name}: {e}")
            raise

    def find_documents(self, collection_name, query, projection=None):
        """Find documents in a collection based on a query."""
        try:
            collection = self.db[collection_name]
            documents = list(collection.find(query, projection))
            self.logger.debug(f"Query: {query}")
            self.logger.info(f"Found {len(documents)} documents in {collection_name}")
            return documents
        except Exception as e:
            self.logger.exception(f"An error occurred while finding documents in {collection_name}: {e}")
            raise

    def update_documents(self, collection_name, query, update_values):
        """Update documents in a collection."""
        try:
            collection = self.db[collection_name]
            result = collection.update_many(query, {'$set': update_values})
            self.logger.debug(f"Update Values: {update_values}")
            self.logger.info(f"Updated {result.modified_count} documents in {collection_name}")
            return result.modified_count
        except Exception as e:
            self.logger.exception(f"An error occurred while updating documents in {collection_name}: {e}")
            raise

    def delete_documents(self, collection_name, query):
        """Delete documents from a collection."""
        try:
            collection = self.db[collection_name]
            result = collection.delete_many(query)
            self.logger.debug(f"Delete Query: {query}")
            self.logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")
            return result.deleted_count
        except Exception as e:
            self.logger.exception(f"An error occurred while deleting documents from {collection_name}: {e}")
            raise
