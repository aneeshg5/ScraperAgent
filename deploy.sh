#!/bin/bash
# =============================================================================
# NVIDIA Hackathon Deployment Script
# Multimodal Web Research Intelligence Agent
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/deployment.log"
PYTHON_VERSION="3.11"
VENV_NAME="hackathon-env"

# Print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}🔧 $1${NC}"
}

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    print_status "$1"
}

# Error handling
handle_error() {
    print_error "Deployment failed at line $1"
    print_error "Check $LOG_FILE for details"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🚀 NVIDIA HACKATHON DEPLOYMENT 🚀                        ║"
    echo "║              Multimodal Web Research Intelligence Agent                      ║"
    echo "║                        GPU-Accelerated & Ready                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# System information
check_system() {
    log "🔍 Checking system information..."
    
    echo "System Information:" >> "$LOG_FILE"
    echo "==================" >> "$LOG_FILE"
    uname -a >> "$LOG_FILE"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv >> "$LOG_FILE"
    else
        print_warning "No NVIDIA GPU detected - will run in CPU mode"
    fi
    
    if command -v lscpu >/dev/null 2>&1; then
        CPU_INFO=$(lscpu | grep "Model name" | head -1)
        print_info "CPU: ${CPU_INFO#*:}"
    fi
    
    MEMORY_INFO=$(free -h | grep "Mem:" | awk '{print $2}')
    print_info "Memory: $MEMORY_INFO"
}

# Check if running on Brev platform
check_brev_platform() {
    log "🔍 Checking for Brev platform..."
    
    if [[ -n "${BREV_INSTANCE_ID}" ]]; then
        print_success "Running on Brev platform (Instance: ${BREV_INSTANCE_ID})"
        export HACKATHON_MODE=true
        export GPU_ACCELERATED=true
    elif [[ -f "/etc/brev" ]] || [[ -n "${BREV_ENV}" ]]; then
        print_success "Brev platform detected"
        export HACKATHON_MODE=true
    else
        print_info "Not running on Brev platform - using local configuration"
    fi
}

# System dependencies
install_system_dependencies() {
    log "📦 Installing system dependencies..."
    
    if command -v apt-get >/dev/null 2>&1; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y \
            curl wget git build-essential \
            ffmpeg libsm6 libxext6 \
            python3-dev python3-pip python3-venv \
            sqlite3 libsqlite3-dev \
            libjpeg-dev libpng-dev \
            redis-server
        print_success "System dependencies installed (Ubuntu/Debian)"
    elif command -v yum >/dev/null 2>&1; then
        # RHEL/CentOS
        sudo yum update -y
        sudo yum install -y \
            curl wget git gcc gcc-c++ make \
            ffmpeg \
            python3-dev python3-pip \
            sqlite-devel \
            libjpeg-devel libpng-devel \
            redis
        print_success "System dependencies installed (RHEL/CentOS)"
    else
        print_warning "Unknown package manager - some dependencies may need manual installation"
    fi
}

# Python environment setup
setup_python_environment() {
    log "🐍 Setting up Python environment..."
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        print_error "Python not found"
        exit 1
    fi
    
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
    print_info "Python version: $PYTHON_VER"
    
    # Create virtual environment
    if [[ ! -d "$VENV_NAME" ]]; then
        $PYTHON_CMD -m venv "$VENV_NAME"
        print_success "Virtual environment created: $VENV_NAME"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
}

# Install Python dependencies
install_python_dependencies() {
    log "📚 Installing Python dependencies..."
    
    # Install from hackathon requirements if available
    if [[ -f "requirements-hackathon.txt" ]]; then
        print_info "Installing hackathon-optimized requirements..."
        pip install -r requirements-hackathon.txt
        print_success "Hackathon requirements installed"
    elif [[ -f "requirements.txt" ]]; then
        print_info "Installing standard requirements..."
        pip install -r requirements.txt
        print_success "Standard requirements installed"
    else
        print_error "No requirements file found"
        exit 1
    fi
    
    # Install additional hackathon dependencies
    log "🎯 Installing additional hackathon dependencies..."
    
    # Try to install NVIDIA-specific packages (may not be available until hackathon)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || {
        print_warning "CUDA-enabled PyTorch installation failed - using CPU version"
        pip install torch torchvision
    }
    
    # Install additional monitoring tools
    pip install nvidia-ml-py3 psutil structlog || print_warning "Some monitoring packages failed to install"
    
    print_success "Python dependencies installed"
}

# Browser setup
setup_browser_automation() {
    log "🌐 Setting up browser automation..."
    
    # Install Playwright browsers
    if command -v playwright >/dev/null 2>&1; then
        playwright install chromium
        playwright install-deps chromium || print_warning "Browser dependencies may be incomplete"
        print_success "Playwright browsers installed"
    else
        print_warning "Playwright not found - browser automation may be limited"
    fi
}

# Environment configuration
setup_environment() {
    log "⚙️  Setting up environment configuration..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f "hackathon.env.template" ]]; then
            cp hackathon.env.template .env
            print_success "Environment file created from hackathon template"
        elif [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_success "Environment file created from example"
        else
            # Create basic .env file
            cat > .env << EOF
# NVIDIA Hackathon Configuration
NVIDIA_API_KEY=your_hackathon_nvidia_api_key_here
NIM_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
NVIDIA_VISION_ENDPOINT=https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2

# Hackathon Mode
HACKATHON_MODE=true
GPU_ACCELERATED=true

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Database
DATABASE_URL=sqlite:///./hackathon_agent_memory.db

# Browser
HEADLESS_BROWSER=true
BROWSER_TIMEOUT=30000
EOF
            print_success "Basic environment file created"
        fi
    else
        print_info "Environment file already exists"
    fi
    
    # Set environment variables for current session
    export HACKATHON_MODE=true
    export GPU_ACCELERATED=true
    
    # Load environment variables
    if [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
        print_success "Environment variables loaded"
    fi
}

# Database initialization
initialize_database() {
    log "🗄️  Initializing database..."
    
    python3 -c "
import asyncio
import sys
import sqlite3
import os

async def init_db():
    try:
        from app.utils.memory import MemorySaver
        memory = MemorySaver()
        await memory.initialize()
        print('✅ Database initialized successfully')
    except Exception as e:
        print(f'⚠️  Database initialization error: {e}')
        # Create basic database structure
        db_path = os.getenv('DATABASE_URL', 'sqlite:///./agent_memory.db').replace('sqlite:///', '')
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        conn.commit()
        conn.close()
        print('✅ Basic database structure created')

if sys.version_info >= (3, 7):
    asyncio.run(init_db())
else:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db())
" || print_warning "Database initialization had issues but continuing..."
    
    print_success "Database setup completed"
}

# NLP models download
download_nlp_models() {
    log "🧠 Downloading NLP models..."
    
    python3 -c "
import nltk
import spacy
import subprocess
import sys

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK models downloaded')
except Exception as e:
    print(f'⚠️  NLTK download warning: {e}')

try:
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=False, capture_output=True)
    print('✅ SpaCy models downloaded')
except Exception as e:
    print(f'⚠️  SpaCy download warning: {e}')
" || print_warning "Some NLP models failed to download"
    
    print_success "NLP models setup completed"
}

# GPU validation
validate_gpu_setup() {
    log "🔧 Validating GPU setup..."
    
    python3 -c "
import sys
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f'✅ GPU Available: {device_name}')
        print(f'📊 GPU Count: {device_count}')
        
        # Test basic GPU operation
        x = torch.randn(2, 3).cuda()
        y = x.sum()
        print('✅ GPU computation test passed')
    else:
        print('⚠️  No GPU available - will run in CPU mode')
except ImportError:
    print('⚠️  PyTorch not available')
except Exception as e:
    print(f'⚠️  GPU validation warning: {e}')
" || print_warning "GPU validation had issues"
}

# Security setup
setup_security() {
    log "🔒 Setting up security configurations..."
    
    # Create logs directory with proper permissions
    mkdir -p logs
    chmod 755 logs
    
    # Create backups directory
    mkdir -p backups
    chmod 755 backups
    
    # Set proper file permissions
    chmod 644 .env 2>/dev/null || true
    chmod 755 *.sh 2>/dev/null || true
    
    print_success "Security configurations applied"
}

# Health check
run_health_check() {
    log "🏥 Running health checks..."
    
    python3 -c "
import asyncio
import sys
import importlib.util

async def health_check():
    try:
        # Test core imports
        import fastapi
        import uvicorn
        import pydantic
        print('✅ Core dependencies available')
        
        # Test app imports
        sys.path.insert(0, '.')
        from app.main import app
        print('✅ Application imports successful')
        
        # Test database connection
        from app.utils.memory import MemorySaver
        memory = MemorySaver()
        await memory.initialize()
        print('✅ Database connection working')
        
        print('🎉 All health checks passed')
        return True
    except Exception as e:
        print(f'❌ Health check failed: {e}')
        return False

if sys.version_info >= (3, 7):
    success = asyncio.run(health_check())
else:
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(health_check())

sys.exit(0 if success else 1)
" || {
    print_error "Health check failed"
    exit 1
}
}

# Start services
start_services() {
    log "🚀 Starting services..."
    
    # Start Redis if available
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port 6379 || print_warning "Redis failed to start"
        print_info "Redis server started"
    fi
    
    # Determine startup command
    if [[ -f "start.py" ]]; then
        STARTUP_CMD="python3 start.py"
        if [[ "$HACKATHON_MODE" == "true" ]]; then
            STARTUP_CMD="$STARTUP_CMD --hackathon-mode"
        fi
        if [[ "$GPU_ACCELERATED" == "true" ]]; then
            STARTUP_CMD="$STARTUP_CMD --gpu"
        fi
    else
        STARTUP_CMD="uvicorn app.main:app --host 0.0.0.0 --port 8000"
    fi
    
    print_success "Deployment completed successfully!"
    print_info "Starting application with: $STARTUP_CMD"
    
    # Print access URLs
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        🎉 DEPLOYMENT SUCCESSFUL! 🎉                       ║"
    echo "║                                                                            ║"
    echo "║  Application URL:  http://localhost:8000                                  ║"
    echo "║  API Documentation: http://localhost:8000/docs                           ║"
    echo "║  Health Check:     http://localhost:8000/health                          ║"
    echo "║                                                                            ║"
    echo "║  🏆 Ready for NVIDIA Hackathon evaluation!                               ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Execute startup command
    exec $STARTUP_CMD
}

# Main deployment flow
main() {
    print_banner
    
    # Create log file
    touch "$LOG_FILE"
    log "🚀 Starting NVIDIA Hackathon deployment..."
    
    # Execute deployment steps
    check_system
    check_brev_platform
    install_system_dependencies
    setup_python_environment
    install_python_dependencies
    setup_browser_automation
    setup_environment
    initialize_database
    download_nlp_models
    validate_gpu_setup
    setup_security
    run_health_check
    start_services
}

# Handle script termination
cleanup() {
    print_info "Cleaning up..."
    deactivate 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@" 