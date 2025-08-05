#!/bin/bash

# Setup script for chatDESI modular refactor
echo "Setting up chatDESI modular project structure..."

# Create main project directory
mkdir -p chatdesi_modular
cd chatdesi_modular

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv chatdesi_venv

# Activate virtual environment (Linux/Mac)
source chatdesi_venv/bin/activate

# For Windows, use instead:
# chatdesi_venv\Scripts\activate

# Create project structure
echo "Creating project structure..."
mkdir -p chatdesi/{config,auth,data,ui,utils,tests}
mkdir -p docs
mkdir -p scripts

# Create __init__.py files
touch chatdesi/__init__.py
touch chatdesi/config/__init__.py
touch chatdesi/auth/__init__.py
touch chatdesi/data/__init__.py
touch chatdesi/ui/__init__.py
touch chatdesi/utils/__init__.py
touch chatdesi/tests/__init__.py

echo "Project structure created successfully!"
echo "Next steps:"
echo "1. Activate the virtual environment: source chatdesi_venv/bin/activate"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Copy your existing files to the appropriate modules"