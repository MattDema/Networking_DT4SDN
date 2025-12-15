-- src/database/schema.sql
CREATE TABLE IF NOT EXISTS traffic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    link_id TEXT NOT NULL,
    bytes_sent INTEGER NOT NULL,
    packets_sent INTEGER NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_link_id (link_id)
);
