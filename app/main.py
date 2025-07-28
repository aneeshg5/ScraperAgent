"""
FastAPI Backend for Multimodal Web Research Intelligence Agent
NVIDIA AI Stack Integration
"""
import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
import os

from app.agent.core import multimodal_research_agent, autonomous_research_agent
from app.agent.vision import analyze_image_upload
from app.models.schemas import (
    BrowseRequest, 
    AgentResponse, 
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    ErrorResponse,
    HealthResponse
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for health monitoring
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Multimodal Web Research Intelligence Agent")
    logger.info("🔧 Initializing NVIDIA AI stack components...")
    
    # Initialize browser on startup
    try:
        from app.agent.browser_tools import initialize_browser
        await initialize_browser()
        logger.info("✅ Browser automation initialized")
    except Exception as e:
        logger.warning(f"⚠️  Browser initialization warning: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down agent services...")
    try:
        from app.agent.browser_tools import cleanup_browser
        await cleanup_browser()
        logger.info("✅ Browser cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️  Cleanup warning: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Web Research Intelligence Agent",
    description="NVIDIA-powered AI agent for intelligent web research and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving the full frontend interface"""
    try:
        # Try to serve the full frontend interface
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to simple interface if frontend file doesn't exist
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multimodal Web Research Agent</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; }
                .container { max-width: 800px; margin: 0 auto; text-align: center; }
                h1 { color: #76b900; margin-bottom: 20px; }
                .feature { background: rgba(255,255,255,0.1); padding: 20px; margin: 20px 0; border-radius: 10px; }
                .nvidia-green { color: #76b900; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Multimodal Web Research Intelligence Agent</h1>
                <p>Powered by <span class="nvidia-green">NVIDIA AI Stack</span></p>
                
                <div class="feature">
                    <h3>🌐 Web Research & Analysis</h3>
                    <p>Intelligent browsing and content extraction with vision capabilities</p>
                </div>
                
                <div class="feature">
                    <h3>🧠 NVIDIA Nemotron Integration</h3>
                    <p>Advanced reasoning and multimodal understanding</p>
                </div>
                
                <div class="feature">
                    <h3>👁️ Vision Analysis</h3>
                    <p>Screenshot analysis and visual content understanding</p>
                </div>
                
                <p><a href="/docs" style="color: #76b900;">📚 View API Documentation</a></p>
                <p><strong>⚠️ Frontend interface not found. Using fallback UI.</strong></p>
            </div>
        </body>
        </html>
        """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    # Check service dependencies
    services = {
        "nvidia_api": bool(os.getenv("NVIDIA_API_KEY")),
        "browser": True,  # Will be updated based on actual browser status
        "vision_model": bool(os.getenv("NVIDIA_VISION_ENDPOINT"))
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        services=services
    )


@app.post("/agent/research", response_model=AgentResponse)
async def research_with_agent(request: BrowseRequest, background_tasks: BackgroundTasks):
    """
    Enhanced endpoint for multimodal web research with structured data and format flexibility
    
    Supports:
    - Multiple output formats (text, CSV, Excel, JSON, HTML table)
    - Individual media extraction (images, videos, documents)
    - Structured data extraction with AI-powered field detection
    - Full-page screenshots and vision analysis
    - NVIDIA Nemotron reasoning and synthesis
    """
    start_processing_time = time.time()
    
    try:
        logger.info(f"🔍 Starting research for query: {request.query}")
        logger.info(f"📊 Analyzing {len(request.urls)} URLs")
        logger.info(f"🎨 Output format: {request.output_format.value}")
        logger.info(f"🎭 Media extraction: {request.extract_media}")
        
        # Execute the enhanced multimodal research agent
        result = await multimodal_research_agent(
            query=request.query,
            urls=[str(url) for url in request.urls],
            max_tokens=request.max_tokens,
            include_screenshots=request.include_screenshots,
            output_format=request.output_format,
            extract_media=request.extract_media,
            structured_fields=request.structured_fields
        )
        
        processing_time = time.time() - start_processing_time
        logger.info(f"✅ Research completed in {processing_time:.2f} seconds")
        
        if result.get('structured_data') and result['structured_data'].get('total_records', 0) > 0:
            logger.info(f"📊 Extracted {result['structured_data']['total_records']} structured records")
        
        if result.get('media_files'):
            logger.info(f"🎭 Found {len(result['media_files'])} media files")
        
        return AgentResponse(
            result=result["analysis"],
            sources_analyzed=result["sources_analyzed"],
            vision_insights=result.get("vision_insights"),
            screenshots=result.get("screenshots"),
            structured_data=result.get("structured_data"),
            formatted_output=result.get("formatted_output"),
            media_files=result.get("media_files"),
            output_format=result.get("output_format", request.output_format),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Research failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Research agent failed: {str(e)}"
        )


@app.post("/agent/autonomous-research", response_model=AgentResponse)
async def autonomous_research(request: BrowseRequest, background_tasks: BackgroundTasks):
    """
    Autonomous research endpoint with enhanced capabilities
    Combines autonomous decision-making with structured data extraction and format flexibility
    """
    start_processing_time = time.time()
    try:
        logger.info(f"🤖 Starting autonomous research for: {request.query}")
        logger.info(f"🎯 Agent will autonomously analyze {len(request.urls)} URLs")
        logger.info(f"🎨 Output format: {request.output_format.value}")
        
        result = await autonomous_research_agent(
            research_goal=request.query,
            starting_urls=[str(url) for url in request.urls],
            max_iterations=8,
            max_tokens=request.max_tokens,
            output_format=request.output_format,
            extract_media=request.extract_media,
            structured_fields=request.structured_fields
        )
        
        processing_time = time.time() - start_processing_time
        logger.info(f"✅ Autonomous research completed in {processing_time:.2f} seconds")
        logger.info(f"🤖 Agent took {result.get('autonomous_actions_taken', 0)} autonomous actions")
        
        return AgentResponse(
            result=result["analysis"],
            sources_analyzed=[data.get("source_url", "") for data in result.get("collected_data", [])],
            vision_insights=[f"Autonomous action: {action}" for action in result.get("action_log", [])],
            screenshots=result.get("screenshots"),
            structured_data=result.get("structured_data"),
            formatted_output=result.get("formatted_output"),
            media_files=result.get("media_files"),
            output_format=result.get("output_format", request.output_format),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"❌ Autonomous research failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Autonomous research agent failed: {str(e)}")


@app.post("/agent/vision", response_model=VisionAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    query: str = None
):
    """
    Analyze uploaded images using NVIDIA vision models
    
    Accepts image uploads and optional queries for targeted analysis
    """
    try:
        logger.info(f"🖼️  Analyzing uploaded image: {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image content
        image_content = await file.read()
        
        # Analyze with vision model
        analysis_result = await analyze_image_upload(image_content, query)
        
        logger.info("✅ Image analysis completed")
        
        return VisionAnalysisResponse(
            analysis=analysis_result["analysis"],
            confidence=analysis_result.get("confidence"),
            detected_elements=analysis_result.get("detected_elements")
        )
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Vision analysis failed: {str(e)}"
        )


@app.get("/agent/status")
async def agent_status():
    """Get current agent status and capabilities"""
    return {
        "status": "active",
        "capabilities": [
            "Web browsing and content extraction",
            "Screenshot capture and analysis",
            "NVIDIA Nemotron reasoning",
            "Multimodal content synthesis",
            "Real-time research insights"
        ],
        "supported_formats": {
            "input": ["text queries", "URLs", "images"],
            "output": ["structured analysis", "visual insights", "research summaries"]
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "error_type": "NOT_FOUND",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_type": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    ) 