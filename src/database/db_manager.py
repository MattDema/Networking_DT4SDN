# src/database/db_manager.py
import sqlite3
import pandas as pd
from pathlib import Path

class TrafficDatabase:
    def __init__(self, db_path='data/traffic.db'):
        # Initialize SQLite connection
        # Create tables if not exist
        
    def insert_traffic(self, timestamp, link_id, bytes_sent, packets_sent):
        # INSERT INTO traffic VALUES (...)
        
    def get_recent_traffic(self, link_id, duration_seconds=60):
        # SELECT * FROM traffic 
        # WHERE link_id = ? AND timestamp > ?
        # ORDER BY timestamp
        # Return as pandas DataFrame
