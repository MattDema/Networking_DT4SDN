# Architecture Documentation

## Project Overview

This project implements a **predictive Digital Twin** for Software-Defined Networks (SDN). The Digital Twin uses Machine Learning to forecast network traffic 30-60 seconds into the future, enabling proactive network management.

---

## System Architecture

### High-Level Overview


                ╔══════════════════════════════════╗
                ║   Physical Twin (Mininet)        ║
                ║                                  ║
                ║  [h1]───[s1]───[s2]───[h2]       ║
                ║         Real Traffic             ║
                ╚════════════════╦═════════════════╝
                                 ║
                     RYU Northbound REST API
                  (Topology + Traffic Statistics)
                                 ║
                ╔════════════════╩═════════════════╗
                ║  RYU Controller (Physical)       ║
                ║  ┌──────────┐  ┌──────────┐      ║
                ║  │Topology  │→ │Traffic   │      ║
                ║  │Discovery │  │Monitor   │      ║
                ║  └──────────┘  └────┬─────┘      ║
                ╚════════════════╦════╩════════════╝
                                 ║
                          Store Historical Data
                                 ║
                ╔════════════════╩═════════════════╗
                ║         Database (SQLite)        ║
                ║      -  Timestamps               ║
                ║      -  Link utilization         ║
                ║      -  Packet/byte counts       ║
                ╚════════════════╦═════════════════╝
                                 ║
                          Training & Prediction
                                 ║
                ╔════════════════╩═════════════════╗
                ║  ML Prediction Module            ║
                ║  ┌────────┐ ┌──────┐ ┌────────┐  ║
                ║  │Preproc │→│LSTM/ │→│Traffic ║  ║
                ║  │        │ │GRU   │ │Predict ║  ║
                ║  └────────┘ └──────┘ └───┬────┘  ║
                ╚════════════════╦═════════╩═══════╝
                                 ║
                         Predicted Traffic
                                 ║
                ╔════════════════╩═════════════════╗
                ║       Digital Twin (Mininet)     ║
                ║                                  ║
                ║    [h1]───[s1]───[s2]───[h2]     ║
                ║    Simulated Future (30-60s)     ║
                ╚════════════════╦═════════════════╝
                                 ║
                        Real-time Monitoring
                                 ║
                ╔════════════════╩═════════════════╗
                ║  Web Interface (Flask + D3.js)   ║
                ║  ┌──────────┐  ┌──────────┐      ║
                ║  │Physical  │  │Digital   │      ║
                ║  │Twin View │  │Twin View │      ║
                ║  │(Current) │  │(Future)  │      ║
                ║  └──────────┘  └──────────┘      ║
                ║  ┌─────────────────────────┐     ║
                ║  │ Accuracy Metrics        │     ║
                ║  └─────────────────────────┘     ║
                ╚══════════════════════════════════╝


### Component Descriptions

#### 1. Physical Twin Layer
- **Technology**: Mininet + Open vSwitch
- **Purpose**: Real SDN network emulation
- **Components**: 
  - Hosts (h1, h2, ...) - Generate actual traffic
  - Switches (s1, s2, ...) - OpenFlow-enabled switches
  - Links - Network connections with configurable bandwidth

#### 2. RYU Controller (Physical Twin)
- **Topology Discovery**: Detects switches, links, and hosts
- **Traffic Monitoring**: Collects port statistics every 1-2 seconds
- **Data Logger**: Stores metrics to database
- **API**: Exposes REST endpoints for data access

#### 3. Database Layer
- **Technology**: SQLite3
- **Storage**: Time-series traffic data
- **Schema**: 
  - Primary: `traffic_data` table
  - Indexes: Optimized for time-range queries
- **Retention**: Configurable (default: 7 days)

#### 4. ML Prediction Module
- **Framework**: TensorFlow/Keras
- **Models**: LSTM or GRU neural networks
- **Input**: Last 60 seconds of traffic data
- **Output**: Predicted traffic for next 30-60 seconds
- **Training**: Offline on historical data
- **Inference**: Real-time predictions

#### 5. Digital Twin Layer
- **Technology**: Mininet + RYU Controller
- **Purpose**: Simulate predicted network state
- **Traffic Generation**: Based on ML predictions
- **Topology**: Mirrors Physical Twin exactly

#### 6. Web Interface
- **Backend**: Flask (Python)
- **Frontend**: HTML5 + D3.js
- **Features**:
  - Dual-view topology visualization
  - Real-time traffic graphs
  - Prediction accuracy dashboard
  - Interactive controls

---

---

## Directory Structure Explained

### `/src/controllers/`
**Purpose**: RYU controller applications for Physical and Digital Twins.

#### `physical_twin_controller.py`
- Monitors Physical Twin network
- Collects topology via RYU REST API
- Records traffic statistics
- Saves data to database
- **Based on**: MaxMichelutti's `digital_twin_ryu_async.py`

#### `digital_twin_controller.py`
- Controls Digital Twin network
- Receives predictions from ML module
- Simulates predicted traffic
- Logs actual vs predicted for validation

---

### `/src/ml_models/`
**Purpose**: Machine Learning components for traffic prediction.

#### `traffic_predictor.py`
Main prediction interface

```python
class TrafficPredictor:
def init(self, model_path):
"""Load trained model"""

text
def predict(self, historical_data, horizon=30):
    """
    Predict traffic for next 'horizon' seconds
    
    Args:
        historical_data: Last N seconds of traffic
        horizon: Seconds to predict ahead
        
    Returns:
        Predicted traffic values
    """
```

#### `model_trainer.py`
Training pipeline

```python
class ModelTrainer:
def train(self, training_data):
"""Train LSTM/GRU model on historical data"""

def evaluate(self, test_data):
    """Calculate accuracy metrics"""
    
def save_model(self, path):
    """Save trained model"""
```

#### `/models/`
- Stores trained model files (.h5, .pkl)
- Version controlled separately (not in Git due to size)

---

### `/src/web_interface/`
**Purpose**: Flask web application for visualization.

#### `app.py`
Flask routes

```python
@app.route('/')
def home():
"""Main dashboard"""

@app.route('/api/topology')
def get_topology():
"""Get current Physical Twin topology"""

@app.route('/api/predictions')
def get_predictions():
"""Get Digital Twin predictions"""

@app.route('/api/accuracy')
def get_accuracy():
"""Get prediction accuracy metrics"""
```

#### `/templates/index.html`
- Main dashboard HTML
- Dual-view layout (Physical vs Digital)
- D3.js integration

#### `/static/js/`
- `topology.js`: D3.js network visualization
- `charts.js`: Traffic and accuracy charts
- `api.js`: API calls to Flask backend

---

### `/src/database/`
**Purpose**: Database management for historical data.

#### `db_manager.py`

```python
class DatabaseManager:
def init(self, db_path):
"""Initialize SQLite connection"""

def store_traffic(self, timestamp, link_id, stats):
    """Store traffic statistics"""
    
def get_historical_data(self, link_id, duration):
    """Retrieve historical data for ML training"""
    
def get_latest(self, link_id, seconds=60):
    """Get recent data for prediction input"""
```

#### `schema.sql`

```sql
CREATE TABLE traffic_data (
timestamp REAL PRIMARY KEY,
link_id TEXT NOT NULL,
bytes_sent INTEGER,
bytes_received INTEGER,
packets_sent INTEGER,
packets_received INTEGER,
bandwidth_utilization REAL
);

CREATE INDEX idx_link_time ON traffic_data(link_id, timestamp);
```

---

### `/src/utils/`
**Purpose**: Helper functions and utilities.

#### `data_collector.py`
- Interfaces with RYU REST API
- Formats raw data for database
- Handles data validation

#### `metrics.py`
```python
def calculate_accuracy(predicted, actual):
"""Calculate RMSE, MAE, R² scores"""

def plot_comparison(predicted, actual, save_path):
"""Generate prediction vs actual plots"""
```

---

### `/data/`
**Purpose**: Data storage (excluded from Git).

- `/training/`: Training datasets
- `/logs/`: RYU controller logs
- `/models/`: Large ML model files

---

### `/tests/`
**Purpose**: Unit and integration tests.

#### `test_predictor.py`
```python
def test_prediction_format():
"""Test predictor output format"""

def test_accuracy_calculation():
"""Test metric calculations"""
```

---

## Data Flow

### 1. Data Collection (Physical Twin → Database)
Mininet → RYU Controller → data_collector.py → db_manager.py → SQLite

### 2. Model Training (Database → ML Model)
SQLite → model_trainer.py → Trained Model (.h5)

### 3. Prediction (ML Model → Digital Twin)
Recent Data → traffic_predictor.py → Predictions → digital_twin_controller.py

### 4. Visualization (All Components → Web Interface)
Physical Twin Stats ↘
→ Flask API → D3.js Dashboard
Predictions ↗

---

## Key Design Decisions

### Why LSTM/GRU for Prediction?
- Sequential data (time-series traffic)
- Captures temporal dependencies
- Proven for network traffic forecasting

### Why SQLite?
- File-based (no server needed)
- Fast enough for project scale
- Easy deployment
- Built into Python

### Why Flask?
- Lightweight Python framework
- Easy ML model integration
- RESTful API support
- Works with D3.js

### Why Dual Mininet Instances?
- Physical Twin: Real traffic
- Digital Twin: Simulated predictions
- Visual comparison of predicted vs actual

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- ✅ Project structure setup
- ✅ Environment configuration
- [ ] Reverse engineer MaxMichelutti's code
- [ ] Basic data collection pipeline

### Phase 2: ML Module (Weeks 3-4)
- [ ] Implement data preprocessing
- [ ] Build LSTM/GRU models
- [ ] Training pipeline
- [ ] Validation and metrics

### Phase 3: Web Interface (Weeks 5-6)
- [ ] Flask backend
- [ ] D3.js visualization
- [ ] Dual-view dashboard
- [ ] Real-time updates

### Phase 4: Integration (Weeks 7-8)
- [ ] Connect all components
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation

---

## Performance Requirements

- **Prediction Latency**: < 1 second
- **Prediction Accuracy**: R² > 0.85
- **Data Collection Rate**: Every 1-2 seconds
- **Web Dashboard Update**: Every 5 seconds
- **Prediction Horizon**: 30-60 seconds ahead

---