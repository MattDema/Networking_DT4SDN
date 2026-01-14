# Networking_DT4SDN
Project related to "Networking mod.2" course of University of Trento regarding the creation of a Digital Twin for SDN networks

# Digital Twin for SDN Networks

Predictive Digital Twin implementation for Software-Defined Networks using Machine Learning.

## 🎯 Project Goal

Build an intelligent Digital Twin that predicts network traffic 30-60 seconds into the future, enabling proactive network management and optimization.

## 🏗️ Architecture

- **Physical Twin**: Mininet network with real traffic
- **Digital Twin**: Mininet simulation of predicted traffic
- **ML Engine**: LSTM/GRU models for traffic forecasting
- **Web Dashboard**: Real-time visualization of both twins

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- ComNetsEmu VM (for testing)
- 8GB RAM minimum

### Setup

**Windows:**
```bash
.\setup_env.ps1
```

```bash
python test_setup.py
```

**macOS:**
```bash
./setup_env_mac.sh
python test_setup.py
```

**Linux/VM:**
```bash
./setup_vm.sh
python3 test_setup.py
```

### Running the Project

## 🌐 Multi-VM Orchestration Setup

The system is designed to run across two separate virtual machines to isolate the **Physical Twin (PT)** from the **Digital Twin (DT)**.

### 1. Physical Twin (PT) VM
This VM hosts the actual network and the SDN controller.

1. **Start the Ryu Controller**:
   Run the script to initialize the controller with REST support and link monitoring:
   ```bash
   ./start_ryu.sh
   ```
2. **Start the Mininet Topology**:
   In a separate terminal, create the physical network:
   ```bash
   sudo python3 src/utils/custom_topo.py
   ```

### 2. Digital Twin (DT) VM
This VM hosts the system intelligence (ML) and the control dashboard.

1. **Configure the Physical Twin IP**:
   Set the PT VM's IP address so the orchestrator can reach it:
   ```bash
   export PT_IP=<PHYSICAL_VM_IP>
   ```
2. **Start the Orchestrator**:
   Launch the main process that handles data collection, predictions, and the web server:
   ```bash
   python3 src/controllers/orchestrator.py
   ```

### 📊 Monitoring
Once everything is running, the dashboard will be accessible from the Digital Twin VM (or the host if port-forwarding is configured) at:
`http://<DT_VM_IP>:5000`

## 📁 Project Structure

```bash
Networking_DT4SDN/
├── src/ # Source code
│ ├── controllers/ # RYU controllers
│ ├── ml_models/ # ML prediction
│ ├── web_interface/# Flask + D3.js
│ ├── database/ # SQLite management
│ └── utils/ # Helpers
├── data/ # Training data & logs
├── tests/ # Unit tests
└── docs/ # Documentation
```

## 📚 Documentation

- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) - System design
- [API Reference](docs/API.md) - Endpoints and interfaces

## 🧪 Testing

Run environment test
```bash
python test_setup.py
```

Run unit tests
```bash
pytest tests/
```

## 👥 Team

- De Marco Matthew
- Lo Iacono Andrea
- Revrenna Jago

## 📝 License

Academic project for Networking 2 course, University of Trento.

---

**Course**: Networking 2 (Softwarized and Virtualized Mobile Networks)  
**Professor**: Prof. Fabrizio Granelli  
**Year**: 2025-2026
