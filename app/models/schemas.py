"""
API schemas for the Multimodal Web Research Intelligence Agent
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ScreenshotData(BaseModel):
    """Model for screenshot data"""
    url: str = Field(..., description="URL where the screenshot was taken")
    image_base64: str = Field(..., description="Base64-encoded screenshot image")
    timestamp: datetime = Field(default_factory=datetime.now, description="When screenshot was captured")
    width: Optional[int] = Field(None, description="Screenshot width in pixels")
    height: Optional[int] = Field(None, description="Screenshot height in pixels")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "timestamp": "2024-01-15T10:30:00Z",
                "width": 1280,
                "height": 720
            }
        }


class BrowseRequest(BaseModel):
    """Request model for web research tasks"""
    query: str = Field(..., description="Research query or question to investigate")
    urls: List[HttpUrl] = Field(..., description="List of URLs to analyze")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens for LLM response")
    include_screenshots: Optional[bool] = Field(True, description="Whether to capture and analyze screenshots")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest trends in AI technology?",
                "urls": [
                    "https://example.com/ai-news",
                    "https://example.com/tech-trends"
                ],
                "max_tokens": 512,
                "include_screenshots": True
            }
        }


class VisionAnalysisRequest(BaseModel):
    """Request model for vision analysis of uploaded images"""
    query: Optional[str] = Field(None, description="Optional query about the image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key elements shown in this screenshot?"
            }
        }


class AgentResponse(BaseModel):
    """Response model for agent research results"""
    result: str = Field(..., description="Analyzed research results and insights")  
    sources_analyzed: List[str] = Field(..., description="List of URLs that were successfully analyzed")
    vision_insights: Optional[List[str]] = Field(None, description="Insights extracted from visual content")
    screenshots: Optional[List[ScreenshotData]] = Field(None, description="Captured screenshots with metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": "Based on the analysis of the provided sources, here are the key insights...",
                "sources_analyzed": ["https://example.com/source1", "https://example.com/source2"],
                "vision_insights": ["Screenshot shows trending AI applications", "Graph indicates 40% growth in AI adoption"],
                "screenshots": [
                    {
                        "url": "https://example.com/source1",
                        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "width": 1280,
                        "height": 720
                    }
                ],
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time": 15.2
            }
        }


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis"""
    analysis: str = Field(..., description="Visual content analysis")
    confidence: Optional[float] = Field(None, description="Confidence score for the analysis")
    detected_elements: Optional[List[str]] = Field(None, description="List of detected visual elements")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis": "The image shows a dashboard with various metrics and charts indicating positive trends in technology adoption.",
                "confidence": 0.92,
                "detected_elements": ["chart", "dashboard", "metrics", "text"]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Failed to analyze the provided URL",
                "error_type": "BROWSER_ERROR",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    services: Dict[str, bool] = Field(..., description="Status of dependent services")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime": 3600.5,
                "services": {
                    "nvidia_api": True,
                    "browser": True,
                    "vision_model": True
                }
            }
        } 