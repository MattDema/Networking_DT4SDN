#!/bin/bash
# setup_env_mac.sh - macOS Environment Setup

echo "=========================================="
echo     "Digital Twin Project - macOS Setup"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3 from https://www.python.org/downloads/"
    echo "Or use Homebrew: brew install python3"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "           ✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""

#################################################################
# remeber to make it executable with: chmod +x setup_env_mac.sh #
#################################################################