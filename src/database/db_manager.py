# db_manager.py - SQLite database manager for Digital Twin
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Default database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'digital_twin.db')


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database with schema."""
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        
        with self._get_connection() as conn:
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    conn.executescript(f.read())
            conn.commit()
    
    # --- TRAFFIC STATS ---
    def save_port_stats(self, dpid: int, port_no: int, 
                        rx_packets: int, tx_packets: int,
                        rx_bytes: int, tx_bytes: int):
        """Save port statistics from a switch."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO traffic_stats 
                (dpid, port_no, rx_packets, tx_packets, rx_bytes, tx_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dpid, port_no, rx_packets, tx_packets, rx_bytes, tx_bytes))
            conn.commit()
    
    def save_flow_stats(self, dpid: int, priority: int, 
                        match_rules: dict, packet_count: int,
                        byte_count: int, actions: list):
        """Save flow statistics from a switch."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO flow_stats 
                (dpid, priority, match_rules, packet_count, byte_count, actions)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dpid, priority, json.dumps(match_rules), 
                  packet_count, byte_count, json.dumps(actions)))
            conn.commit()
    
    def save_bulk_flow_stats(self, flows: List[Dict]):
        """Save multiple flow stats at once."""
        with self._get_connection() as conn:
            conn.executemany('''
                INSERT INTO flow_stats 
                (dpid, priority, match_rules, packet_count, byte_count, actions)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', [(f['dpid'], f['priority'], json.dumps(f.get('match', {})),
                   f['packet_count'], f['byte_count'], 
                   json.dumps(f.get('actions', []))) for f in flows])
            conn.commit()
    
    # --- HOSTS ---
    def save_host(self, mac_address: str, dpid: int, port: int):
        """Save or update a discovered host."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO hosts (mac_address, dpid, port, last_seen)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(mac_address) DO UPDATE SET
                    dpid = excluded.dpid,
                    port = excluded.port,
                    last_seen = CURRENT_TIMESTAMP
            ''', (mac_address, dpid, port))
            conn.commit()
    
    def get_all_hosts(self) -> List[Dict]:
        """Get all discovered hosts."""
        with self._get_connection() as conn:
            rows = conn.execute('SELECT * FROM hosts ORDER BY last_seen DESC').fetchall()
            return [dict(row) for row in rows]
    
    # --- QUERIES FOR ML ---
    def get_traffic_history(self, dpid: int = None, 
                            minutes: int = 60) -> List[Dict]:
        """Get traffic history for ML training."""
        with self._get_connection() as conn:
            if dpid:
                rows = conn.execute('''
                    SELECT * FROM traffic_stats 
                    WHERE dpid = ? AND timestamp > datetime('now', ?)
                    ORDER BY timestamp ASC
                ''', (dpid, f'-{minutes} minutes')).fetchall()
            else:
                rows = conn.execute('''
                    SELECT * FROM traffic_stats 
                    WHERE timestamp > datetime('now', ?)
                    ORDER BY timestamp ASC
                ''', (f'-{minutes} minutes',)).fetchall()
            return [dict(row) for row in rows]
    
    def get_flow_history(self, dpid: int = None,
                         minutes: int = 60) -> List[Dict]:
        """Get flow stats history."""
        with self._get_connection() as conn:
            if dpid:
                rows = conn.execute('''
                    SELECT * FROM flow_stats 
                    WHERE dpid = ? AND timestamp > datetime('now', ?)
                    ORDER BY timestamp ASC
                ''', (dpid, f'-{minutes} minutes')).fetchall()
            else:
                rows = conn.execute('''
                    SELECT * FROM flow_stats 
                    WHERE timestamp > datetime('now', ?)
                    ORDER BY timestamp ASC
                ''', (f'-{minutes} minutes',)).fetchall()
            return [dict(row) for row in rows]
    
    # --- PREDICTIONS ---
    def save_prediction(self, dpid: int, predicted_packets: int,
                        predicted_bytes: int, horizon: int = 30):
        """Save a traffic prediction."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO predictions 
                (dpid, predicted_packets, predicted_bytes, prediction_horizon)
                VALUES (?, ?, ?, ?)
            ''', (dpid, predicted_packets, predicted_bytes, horizon))
            conn.commit()
    
    def update_prediction_actual(self, prediction_id: int,
                                  actual_packets: int, actual_bytes: int):
        """Update prediction with actual values for validation."""
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE predictions 
                SET actual_packets = ?, actual_bytes = ?
                WHERE id = ?
            ''', (actual_packets, actual_bytes, prediction_id))
            conn.commit()
    
    # --- CLEANUP ---
    def cleanup_old_data(self, days: int = 7):
        """Remove data older than specified days."""
        with self._get_connection() as conn:
            cutoff = f'-{days} days'
            conn.execute('DELETE FROM traffic_stats WHERE timestamp < datetime("now", ?)', (cutoff,))
            conn.execute('DELETE FROM flow_stats WHERE timestamp < datetime("now", ?)', (cutoff,))
            conn.commit()
    
    # --- STATS ---
    def get_db_stats(self) -> Dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            stats = {}
            for table in ['traffic_stats', 'flow_stats', 'hosts', 'predictions']:
                count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                stats[table] = count
            return stats


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance