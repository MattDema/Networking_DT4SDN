# VM Setup Guide for Digital Twin 4 SDN

This guide covers setting up two VMs for the Digital Twin project:
- **VM1: ComNetsEmu** - Mininet + RYU Controller (Physical Twin)
- **VM2: Digital Twin** - Orchestrator + Web + PyTorch ML

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST MACHINE                             │
│     (Your laptop with VirtualBox + MobaXterm)               │
│                                                              │
│  ┌──────────────────────┐    ┌───────────────────────────┐  │
│  │   VM1: ComNetsEmu    │    │   VM2: Digital Twin       │  │
│  │   192.168.56.101     │    │   192.168.56.102          │  │
│  │                      │    │                           │  │
│  │  ┌────────────────┐  │    │  ┌─────────────────────┐  │  │
│  │  │    Mininet     │  │REST│  │    Orchestrator     │  │  │
│  │  │   (topology)   │◄─┼────┼──│  (data collector)   │  │  │
│  │  └───────┬────────┘  │API │  └──────────┬──────────┘  │  │
│  │          │           │:8080│             │             │  │
│  │  ┌───────▼────────┐  │    │  ┌──────────▼──────────┐  │  │
│  │  │      RYU       │  │    │  │     ML Predictor    │  │  │
│  │  │   Controller   │  │    │  │     (PyTorch)       │  │  │
│  │  │  :6633 + :8080 │  │    │  └──────────┬──────────┘  │  │
│  │  └────────────────┘  │    │  ┌──────────▼──────────┐  │  │
│  └──────────────────────┘    │  │   Web Dashboard     │  │  │
│                              │  │      :5000          │  │  │
│                              │  └─────────────────────┘  │  │
│                              └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## VM1: ComNetsEmu (Already Installed)

### Requirements
- **RAM:** 2-4 GB
- **Disk:** 20 GB (default from Vagrant)
- **Network:** NAT + Host-Only Adapter

### Verify Installation
```bash
# SSH into ComNetsEmu via MobaXterm
vagrant ssh

# Test Mininet
sudo mn --test pingall

# Test RYU
ryu-manager --version
```

### Configure Network IP
```bash
# Check current IP
ip addr show eth1

# If not set, configure host-only adapter
sudo ip addr add 192.168.56.101/24 dev eth1
sudo ip link set eth1 up
```

### Make IP Persistent
```bash
# Edit Vagrantfile in comnetsemu directory
# Add this line inside the Vagrant.configure block:
config.vm.network "private_network", ip: "192.168.56.101"

# Then reload the VM
vagrant reload
```

---

## VM2: Digital Twin (New Installation)

### Step 1: Create New VM via Vagrant

Create a new directory and Vagrantfile:

```bash
# On your host machine
mkdir digital-twin-vm
cd digital-twin-vm
```

Create `Vagrantfile`:
```ruby
Vagrant.configure("2") do |config|
  # Use Ubuntu 22.04 LTS
  config.vm.box = "ubuntu/jammy64"
  
  # VM Name
  config.vm.hostname = "digital-twin"
  
  # Resources (20GB disk, 4GB RAM for PyTorch)
  config.vm.provider "virtualbox" do |vb|
    vb.name = "DigitalTwin"
    vb.memory = 4096
    vb.cpus = 2
    
    # Increase disk size to 20GB
    # Note: vagrant-disksize plugin required
    # Install: vagrant plugin install vagrant-disksize
  end
  
  # Networking
  config.vm.network "private_network", ip: "192.168.56.102"
  config.vm.network "forwarded_port", guest: 5000, host: 5000  # Web dashboard
  
  # Provisioning script
  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y python3-pip python3-venv git
  SHELL
end
```

```bash
# Start the VM
vagrant up
vagrant ssh
```

### Step 2: Install Dependencies

```bash
# Inside VM2 (Digital Twin)
cd ~

# Clone your project
git clone https://github.com/MattDema/Networking_DT4SDN.git
cd Networking_DT4SDN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install base requirements
pip install --upgrade pip
pip install flask pandas numpy requests joblib scikit-learn matplotlib

# Install PyTorch (CPU version for VM - no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Verify Installation

```bash
# Test PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# Test Flask
python3 -c "import flask; print(f'Flask {flask.__version__} installed')"

# Test connection to ComNetsEmu
curl http://192.168.56.101:8080/stats/switches
# Should return [] or list of switch IDs if RYU is running
```

---

## Running the Full System

### Terminal Setup (4 terminals via MobaXterm)

#### Terminal 1: RYU Controller (on ComNetsEmu)
```bash
# SSH to ComNetsEmu
vagrant ssh  # or connect via MobaXterm

cd ~/comnetsemu
ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest --observe-links
```

#### Terminal 2: Mininet Topology (on ComNetsEmu)
```bash
# SSH to ComNetsEmu (new session)
cd ~/Networking_DT4SDN/src/utils
sudo python3 custom_topo.py
```

#### Terminal 3: Orchestrator + Web (on Digital Twin)
```bash
# SSH to Digital Twin VM
vagrant ssh  # or connect via MobaXterm

cd ~/Networking_DT4SDN/src/controllers
source ../venv/bin/activate
python3 orchestrator.py
```

#### Terminal 4: Traffic Generation (on ComNetsEmu)
```bash
# SSH to ComNetsEmu (new session)
cd ~/Networking_DT4SDN/src/utils
sudo python3 generate_traffic_scenarios.py --scenario normal --duration 60
```

### View Web Dashboard
Open browser on your host machine:
```
http://localhost:5000
```

---

## Overnight Data Collection

### On ComNetsEmu (Traffic Generation):
```bash
cd ~/Networking_DT4SDN/src/utils
sudo python3 run_overnight_collection.py --rounds 2
```

### On Digital Twin (Data Capture):
```bash
cd ~/Networking_DT4SDN/src/utils
source ../venv/bin/activate

# Capture for the full duration (match overnight collection time)
# 5 scenarios × 30 min × 2 rounds = 300 min = 5 hours
python3 capture_real_traffic.py \
  --scenario all \
  --duration 18000 \
  --controller-ip 192.168.56.101 \
  --output data/real_traffic/overnight_$(date +%Y%m%d).csv
```

---

## Troubleshooting

### Cannot connect to RYU from Digital Twin
```bash
# On ComNetsEmu, check if RYU is listening
netstat -tlnp | grep 8080

# Check firewall
sudo ufw status
sudo ufw allow 8080

# Test from Digital Twin
curl http://192.168.56.101:8080/stats/switches
```

### VMs cannot ping each other
```bash
# Check both VMs are on same host-only network
# On each VM:
ip addr show

# Should see 192.168.56.x on an interface
# If not, add the network in VirtualBox:
# Settings > Network > Adapter 2 > Host-only Adapter
```

### PyTorch installation fails (not enough disk)
```bash
# Check disk space
df -h

# Expand disk using vagrant-disksize plugin
# In Vagrantfile add:
config.disksize.size = '30GB'

# Then: vagrant reload
```

---

## Quick Reference

| Component | VM | IP:Port | Command |
|-----------|-----|---------|---------|
| RYU Controller | ComNetsEmu | 192.168.56.101:6633 | `ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest` |
| RYU REST API | ComNetsEmu | 192.168.56.101:8080 | (same as above) |
| Mininet | ComNetsEmu | - | `sudo python3 custom_topo.py` |
| Orchestrator | Digital Twin | 192.168.56.102 | `python3 orchestrator.py` |
| Web Dashboard | Digital Twin | localhost:5000 | (via orchestrator) |
