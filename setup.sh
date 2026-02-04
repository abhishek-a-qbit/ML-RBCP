#!/bin/bash

# Convection Solver - Setup Script
# This script sets up the development environment and verifies installation

echo "🌊 Convection Solver - Setup Script"
echo "======================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}✗ Python 3.8 or higher required. Found: $PYTHON_VERSION${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}! Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Install development dependencies (optional)
echo ""
read -p "Install development dependencies (pytest, black, flake8)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pytest black flake8 mypy jupyter
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p experiments/results
mkdir -p experiments/figures
mkdir -p experiments/data
mkdir -p tests
mkdir -p notebooks
mkdir -p docs
echo -e "${GREEN}✓ Directories created${NC}"

# Verify installation
echo ""
echo "Verifying installation..."

# Check critical packages
PACKAGES=("streamlit" "numpy" "torch" "scipy" "plotly")
ALL_OK=true

for package in "${PACKAGES[@]}"; do
    python3 -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓ $package ($VERSION)${NC}"
    else
        echo -e "${RED}✗ $package not found${NC}"
        ALL_OK=false
    fi
done

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print('✓ GPU available:', torch.cuda.is_available())"

# Summary
echo ""
echo "======================================"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run the application:"
    echo "   streamlit run app.py"
    echo ""
    echo "3. Read the documentation:"
    echo "   - QUICKSTART.md for getting started"
    echo "   - README.md for detailed information"
    echo "   - CONTRIBUTING.md if you want to contribute"
else
    echo -e "${RED}⚠️  Setup completed with warnings${NC}"
    echo "Please check the errors above and reinstall missing packages"
fi

echo ""
echo "For help, visit: https://github.com/yourusername/convection-solver"
echo "======================================"
