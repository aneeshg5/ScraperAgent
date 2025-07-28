@echo off
cls

REM Multimodal Web Research Intelligence Agent Startup Script
REM NVIDIA AI Stack Integration

echo 🤖 Multimodal Web Research Intelligence Agent
echo 🎯 NVIDIA AI Stack • FastAPI • Playwright
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    echo 💡 Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app\" (
    echo ❌ Please run this script from the project root directory
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt not found in current directory
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

REM Install Playwright browsers
echo 🌐 Installing Playwright browsers...
playwright install

REM Check for environment file
if not exist ".env" (
    if exist ".env.example" (
        echo ⚙️  Copying environment template...
        copy .env.example .env
        echo 💡 Please edit .env file with your NVIDIA API credentials
    ) else (
        echo ⚠️  No environment file found. Please create .env with your API keys.
    )
)

REM Start the application
echo.
echo 🚀 Starting Multimodal Web Research Intelligence Agent...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🤖 NVIDIA-Powered AI Research Agent
echo 🌐 Web Interface: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo ❤️  Health Check: http://localhost:8000/health
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 Tip: Configure your NVIDIA API keys in .env for full functionality
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Run the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo 👋 Agent stopped. Thank you for using NVIDIA AI Stack!
pause 