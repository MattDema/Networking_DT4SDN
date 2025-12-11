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
.\setup_env.ps1
python test_setup.py

text

**macOS:**
./setup_env_mac.sh
python test_setup.py

text

**Linux/VM:**
./setup_vm.sh
python3 test_setup.py

text

### Running the Project

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

## 📁 Project Structure

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

text

## 📚 Documentation

- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) - System design
- [API Reference](docs/API.md) - Endpoints and interfaces

## 🧪 Testing

Run environment test
python test_setup.py

Run unit tests
pytest tests/

text

## 👥 Team

- Student 1: ML Module Development
- Student 2: Web Interface
- Student 3: Integration & Testing

## 📝 License

Academic project for Networking 2 course, University of Trento.

---

**Course**: Networking 2 (Softwarized and Virtualized Mobile Networks)  
**Professor**: Prof. Fabrizio Granelli  
**Year**: 2024-2025
6. Create Empty init.py Files
Run this to make Python treat directories as packages:

bash
# On Mac/Linux or Git Bash on Windows
touch src/__init__.py
touch src/controllers/__init__.py
touch src/ml_models/__init__.py
touch src/web_interface/__init__.py
touch src/database/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
Or create each file manually with empty content.

7. Git Commit All Documentation
bash
git add .
git commit -m "Add comprehensive setup scripts and documentation

- Added Mac setup script (setup_env_mac.sh)
- Added environment test script (test_setup.py)
- Created complete documentation:
  - SETUP.md: Installation guide for all platforms
  - ARCHITECTURE.md: System design and structure
  - API.md: API endpoints and schemas
- Updated README.md with quick start guide
- Updated requirements.txt with detailed comments
"

git push origin main