-- Schema for Digital Twin traffic data storage

-- Traffic statistics collected from switches
CREATE TABLE IF NOT EXISTS traffic_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    dpid INTEGER NOT NULL,              -- Switch ID
    port_no INTEGER,                    -- Port number
    rx_packets INTEGER DEFAULT 0,       -- Received packets
    tx_packets INTEGER DEFAULT 0,       -- Transmitted packets
    rx_bytes INTEGER DEFAULT 0,         -- Received bytes
    tx_bytes INTEGER DEFAULT 0          -- Transmitted bytes
);

-- Flow rules snapshot
CREATE TABLE IF NOT EXISTS flow_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    dpid INTEGER NOT NULL,              -- Switch ID
    priority INTEGER,
    match_rules TEXT,                   -- JSON of match conditions
    packet_count INTEGER DEFAULT 0,
    byte_count INTEGER DEFAULT 0,
    actions TEXT                        -- JSON of actions
);

-- Discovered hosts
CREATE TABLE IF NOT EXISTS hosts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mac_address TEXT UNIQUE NOT NULL,
    dpid INTEGER,                       -- Connected switch
    port INTEGER,                       -- Connected port
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ML Predictions (for validation)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    dpid INTEGER NOT NULL,
    port INTEGER,                       
    predicted_packets INTEGER,
    predicted_bytes INTEGER,
    actual_packets INTEGER,             -- Filled later for validation
    actual_bytes INTEGER,
    prediction_horizon INTEGER          -- Seconds into future (30, 60)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_traffic_time ON traffic_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_dpid ON traffic_stats(dpid);
CREATE INDEX IF NOT EXISTS idx_flow_time ON flow_stats(timestamp);