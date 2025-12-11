# Setup Guide - Digital Twin for SDN Networks

This guide covers setting up the development environment for all team members.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Windows Setup](#windows-setup)
- [macOS Setup](#macos-setup)
- [Linux/VM Setup](#linuxvm-setup)
- [Testing Setup](#testing-setup)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### All Platforms
- **Python 3.8+** (3.10 recommended)
- **Git** for version control
- **8GB RAM minimum** (16GB recommended for ML training)
- **10GB free disk space**

### For Testing
- **ComNetsEmu VM** or Linux machine for Mininet/RYU testing
- VirtualBox (if using VM)
- SSH client (MobaXterm for Windows, built-in for Mac/Linux)

---

## Windows Setup

### 1. Install Python
Download from [python.org](https://www.python.org/downloads/) and install.
**Important**: Check "Add Python to PATH" during installation.

### 2. Clone Repository
```powershell
cd C:\Users\YourName\Documents
git clone [https://github.com/MattDema/Networking_DT4SDN.git](https://github.com/MattDema/Networking_DT4SDN.git)
cd Networking_DT4SDN
```

### 3. Run Setup Script
```powershell
.\setup_env.ps1
```

If you get execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Activate Environment
Every time you start working:
```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Test Setup
```powershell
python test_setup.py
```

---

## macOS Setup

### 1. Install Python 3
Check if already installed
```bash
python3 --version
```

If not installed, use Homebrew:
```bash
brew install python3
```

Or download from [python.org](https://www.python.org/downloads/) and install.


### 2. Clone Repository
```bash
cd ~/Documents # or your preferred location
git clone [https://github.com/MattDema/Networking_DT4SDN.git](https://github.com/MattDema/Networking_DT4SDN.git)
cd Networking_DT4SDN
```

### 3. Run Setup Script
```bash
chmod +x setup_env_mac.sh
./setup_env_mac.sh
```

### 4. Activate Environment
Every time you start working:
```bash
source venv/bin/activate
```

### 5. Test Setup
```bash
python test_setup.py
```

---

## Linux/VM Setup

### For ComNetsEmu VM

#### 1. SSH to VM
```bash
ssh vagrant@127.0.0.1
```

Password: vagrant (usually)

#### 2. Create Project Directory
```bash
cd ~
mkdir -p digital-twin-project
cd digital-twin-project
```

#### 3. Clone Repository
```bash
git clone [https://github.com/MattDema/Networking_DT4SDN.git](https://github.com/MattDema/Networking_DT4SDN.git)
cd Networking_DT4SDN
```

#### 4. Run Setup Script
```bash
chmod +x setup_vm.sh
./setup_vm.sh
```

#### 5. Activate Environment
Every time you SSH to VM:
```bash
cd ~/digital-twin-project/Networking_DT4SDN
source venv/bin/activate
```

#### 6. Test Setup
```bash
python3 test_setup.py
```

---

## Testing Setup

### Verify Everything Works

Activate virtual environment first
Then run:
```bash
python test_setup.py
```

Expected output:
```text
===========================================================
Digital Twin for SDN - Environment Test
Python Version Check
Python version: 3.10.x
✓ Python version is compatible (3.8+)

Package Import Test
✓ NumPy - OK
✓ Pandas - OK
✓ Scikit-learn - OK
✓ TensorFlow - OK
✓ Flask - OK
...

✓ All tests passed! Environment is ready.
```

---

## Development Workflow

### For Code Development (Windows/Mac)
1. Write code in VS Code or preferred IDE
2. Commit and push to Git:
```bash
git add .
git commit -m "Description of changes"
git push origin main
```

### For Testing with Mininet (VM)
1. SSH to ComNetsEmu VM
2. Pull latest code:
```bash
cd ~/digital-twin-project/Networking_DT4SDN
git pull
```

3. Run tests (see Testing Guide)

---

## Troubleshooting

### TensorFlow Installation Fails
**On Windows:**
```bash
pip install tensorflow --no-cache-dir
```

**On Mac M1/M2:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal # For GPU acceleration
```

### Virtual Environment Issues
Delete and recreate:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
```

### Git Issues
Set up Git credentials:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Permission Denied (Linux/Mac)
```bash
chmod +x setup_env_mac.sh
chmod +x setup_vm.sh
```

---

## Next Steps

After successful setup:
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand project structure
2. Review [API.md](API.md) for module interfaces
3. Check assigned tasks in Notion

---

## Quick Reference

|  Platform  |         Activate venv         |  Deactivate  |
|------------|-------------------------------|--------------|
|   Windows  | `.\venv\Scripts\Activate.ps1` | `deactivate` |
|  Mac/Linux |  `source venv/bin/activate`   | `deactivate` |

---