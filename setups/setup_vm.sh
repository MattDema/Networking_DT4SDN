#!/bin/bash
# setup_vm.sh - Linux/VM Environment Setup
# Digital Twin for SDN Networks

set -e  # Exit on any error

echo "=========================================="
echo "   Digital Twin Project - VM Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed!"
    echo "Please install Python 3: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "✗ requirements.txt not found!"
    echo "Make sure you're in the project root directory"
    exit 1
fi

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "✗ Failed to create virtual environment!"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo "✗ Failed to activate virtual environment!"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "✓ pip upgraded"
echo ""

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
echo "(This may take 10-15 minutes, especially for TensorFlow)"
echo ""

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed"
else
    echo "✗ Some packages failed to install!"
    echo "Check the errors above"
    exit 1
fi

echo ""
echo "=========================================="
echo "         ✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test your setup: python3 test_setup.py"
echo "2. Read the docs: docs/SETUP.md"
echo ""
echo "To activate environment in future sessions:"
echo "  cd ~/digital-twin-project/Networking_DT4SDN"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
