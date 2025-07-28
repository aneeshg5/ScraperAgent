#!/bin/bash

# Multimodal Web Research Intelligence Agent Startup Script
# NVIDIA AI Stack Integration

echo "🤖 Multimodal Web Research Intelligence Agent"
echo "🎯 NVIDIA AI Stack • FastAPI • Playwright"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "💡 Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -d "app" ]] || [[ ! -f "requirements.txt" ]]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install

# Check for environment file
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        echo "⚙️  Copying environment template..."
        cp .env.example .env
        echo "💡 Please edit .env file with your NVIDIA API credentials"
    else
        echo "⚠️  No environment file found. Please create .env with your API keys."
    fi
fi

# Start the application
echo ""
echo "🚀 Starting Multimodal Web Research Intelligence Agent..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 NVIDIA-Powered AI Research Agent"
echo "🌐 Web Interface: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Tip: Configure your NVIDIA API keys in .env for full functionality"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo ""
echo "👋 Agent stopped. Thank you for using NVIDIA AI Stack!" 