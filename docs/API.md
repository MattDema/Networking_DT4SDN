# API Documentation

## RYU Northbound REST API

### Topology Endpoints (Physical Twin)

#### Get All Switches

GET http://localhost:8080/v1.0/topology/switches

**Response:**
```json
[
{
"dpid": "0000000000000001",
"ports": [...]
}
]
```

#### Get All Links

GET http://localhost:8080/v1.0/topology/links


#### Get All Hosts

GET http://localhost:8080/v1.0/topology/hosts


### Traffic Statistics

#### Get Port Statistics

GET http://localhost:8080/stats/port/{dpid}


---

## Flask Web API (Our Application)

### Topology API

#### GET /api/topology
Get current Physical Twin network topology.

**Response:**
```json
{
"switches": [...],
"links": [...],
"hosts": [...]
}
```

### Prediction API

#### GET /api/predictions
Get Digital Twin predictions for next 30-60 seconds.

**Response:**
```json
{
"timestamp": "2025-12-08T18:30:00Z",
"horizon": 30,
"predictions": [
{
"link_id": "s1-s2",
"predicted_bandwidth": 0.65,
"confidence": 0.92
}
]
}
```

### Accuracy API

#### GET /api/accuracy
Get prediction accuracy metrics.

**Response:**
```json
{
"overall": {
"rmse": 125.4,
"mae": 98.2,
"r2": 0.87
},
"per_link": [...]
}
```

---

## Database Schema

### traffic_data Table

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
```

---