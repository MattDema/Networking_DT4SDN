# src/database/db_manager.py
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
            # If schema.sql file exists, load it. 
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    conn.executescript(f.read())
            conn.commit()
    
    # traffic stats
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
    
    # hosts
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
    
    # queries for ml
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
    
    # predictions
    def save_prediction(self, dpid: int, port: int, predicted_packets: int,
                        predicted_bytes: int, horizon: int = 30):
        """Save a traffic prediction."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO predictions 
                (dpid, port_no, predicted_bytes, prediction_horizon)
                VALUES (?, ?, ?, ?)
            ''', (dpid, port, predicted_bytes, horizon))
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
    
    def store_prediction(self, dpid: int, port_no: int,
                         predicted_bytes: float, timestamp=None):
        """
        Saves a single prediction entry to the database.
        """
        # Format timestamp if provided as a number (Unix time)
        ts_val = timestamp
        if isinstance(timestamp, (int, float)):
            ts_val = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO predictions 
                (dpid, port_no, predicted_bytes, timestamp, prediction_horizon)
                VALUES (?, ?, ?, ?, ?)
            ''', (dpid, port_no, float(predicted_bytes), ts_val, 5))
            conn.commit()

    def get_recent_traffic(self, link_id: str, duration_seconds: int = 60):
        """
        Fetches recent traffic history for a specific link ID.
        Calculates the delta (speed) between rows.
        """
        import pandas as pd

        # parse id ("s1-eth1" -> dpid=1, port=1)
        try:
            clean_id = link_id.replace('s', '')
            parts = clean_id.split('-eth')
            dpid = int(parts[0])
            port_no = int(parts[1])
        except Exception:
            # return empty df if parsing fails
            return pd.DataFrame()

        with self._get_connection() as conn:
            # fetch data, more rows than needed (duration * 3) to ensure we have enough 
            # points to calculate differences even if some polls were missed.
            limit = duration_seconds * 3
            query = '''
                SELECT tx_bytes, timestamp 
                FROM traffic_stats 
                WHERE dpid = ? AND port_no = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(dpid, port_no, limit))

        if not df.empty and len(df) > 1:
            # sort oldest -> newest (required for diff calculation)
            df = df.sort_values('timestamp', ascending=True)

            # critical: calculate speed (delta)
            # this converts "total cumulative bytes" into "bytes per interval"
            df['bytes_sent'] = df['tx_bytes'].diff().fillna(0)

            # remove negative spikes (which happen if switch resets counters)
            df['bytes_sent'] = df['bytes_sent'].clip(lower=0)

            # return only the relevant columns
            return df[['bytes_sent', 'timestamp']]

        return pd.DataFrame()  # return empty if no data or not enough data

    # cleanup
    def cleanup_old_data(self, days: int = 7):
        """Remove data older than specified days."""
        with self._get_connection() as conn:
            cutoff = f'-{days} days'
            conn.execute('DELETE FROM traffic_stats WHERE timestamp < datetime("now", ?)', (cutoff,))
            conn.execute('DELETE FROM flow_stats WHERE timestamp < datetime("now", ?)', (cutoff,))
            conn.commit()
    
    # stats
    def get_db_stats(self) -> Dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            stats = {}
            for table in ['traffic_stats', 'flow_stats', 'hosts', 'predictions']:
                try:
                    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                    stats[table] = count
                except sqlite3.OperationalError:
                    stats[table] = 0 # Table might not exist yet
            return stats

    def get_active_links(self) -> List[Dict]:
        """
        Returns a list of links (dpid, port) that have reported traffic
        in the last 5 minutes.
        """
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT DISTINCT dpid, port_no 
                FROM traffic_stats 
                WHERE timestamp >= datetime('now', '-5 minutes')
            ''').fetchall()

            return [{'dpid': row['dpid'], 'port': row['port_no']} for row in rows]

# helper function
def get_db():
    """
    Returns a NEW DatabaseManager instance.
    Essential for multi-threading (Collector vs Predictor vs Web).
    """
    db = DatabaseManager()

    return db
