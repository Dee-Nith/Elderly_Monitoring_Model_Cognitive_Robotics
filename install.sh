#!/bin/bash

# Elderly Monitoring Model - Installation Script

echo "🏥 Elderly Monitoring Model - Installation Script"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models logs

# Test installation
echo "🧪 Testing installation..."
python3 -c "import cv2, mediapipe, sklearn, pandas, numpy; print('✅ All dependencies installed successfully!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "To run the system:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run the system: python src/main_predictor.py"
    echo ""
    echo "For more information, see README.md"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi
