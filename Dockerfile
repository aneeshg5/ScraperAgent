# =============================================================================
# NVIDIA Hackathon - Multi-stage Dockerfile
# Multimodal Web Research Intelligence Agent
# =============================================================================

# =============================================================================
# Stage 1: Base Environment Setup
# =============================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# NVIDIA and CUDA environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# Application environment
ENV HACKATHON_MODE=true
ENV GPU_ACCELERATED=true
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    curl \
    wget \
    git \
    # Python and development
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Media processing
    ffmpeg \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    libpng-dev \
    # Database
    sqlite3 \
    libsqlite3-dev \
    # Browser dependencies
    libglib2.0-0 \
    libnss3-dev \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm-dev \
    libasound2 \
    # Additional utilities
    htop \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM base as python-deps

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-hackathon.txt ./

# Install Python dependencies
RUN pip install -r requirements-hackathon.txt || pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install additional hackathon dependencies
RUN pip install nvidia-ml-py3 psutil structlog

# Download NLP models
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || true
RUN python3 -m spacy download en_core_web_sm || true

# =============================================================================
# Stage 3: Browser Setup
# =============================================================================
FROM python-deps as browser-setup

# Install Playwright and browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Verify browser installation
RUN playwright install --dry-run || echo "Browser installation completed"

# =============================================================================
# Stage 4: Application Setup
# =============================================================================
FROM browser-setup as app-setup

# Create application directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/logs /app/backups /app/cache && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Set proper permissions
RUN chmod +x *.sh 2>/dev/null || true
RUN chmod +x start.py 2>/dev/null || true

# Initialize database structure
RUN python3 -c "
import sqlite3
import os
db_path = 'hackathon_agent_memory.db'
conn = sqlite3.connect(db_path)
conn.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
conn.commit()
conn.close()
print('Database initialized')
" || true

# =============================================================================
# Stage 5: Production Image
# =============================================================================
FROM app-setup as production

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Volume for persistent data
VOLUME ["/app/logs", "/app/backups", "/app/cache"]

# Environment variables for runtime
ENV PYTHONPATH=/app
ENV HACKATHON_MODE=true
ENV GPU_ACCELERATED=true

# Labels for metadata
LABEL org.opencontainers.image.title="NVIDIA Hackathon Multimodal Agent"
LABEL org.opencontainers.image.description="GPU-accelerated multimodal web research intelligence agent"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="hackathon-participant"

# Default command
CMD ["python3", "start.py", "--hackathon-mode", "--gpu"]

# =============================================================================
# Development Stage (optional)
# =============================================================================
FROM production as development

# Install development tools
USER root
RUN pip install pytest pytest-asyncio black flake8 mypy jupyter

# Switch back to app user
USER appuser

# Development command
CMD ["python3", "start.py", "--debug", "--reload"]

# =============================================================================
# GPU Validation Stage (optional)
# =============================================================================
FROM production as gpu-validation

# Run GPU validation
RUN python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU Available: {torch.cuda.get_device_name(0)}')
    print(f'📊 GPU Count: {torch.cuda.device_count()}')
    print(f'🔧 CUDA Version: {torch.version.cuda}')
    x = torch.randn(2, 3).cuda()
    y = x.sum()
    print('✅ GPU computation test passed')
else:
    print('⚠️  No GPU available - will run in CPU mode')
" 