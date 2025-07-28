"""
API schemas for the Multimodal Web Research Intelligence Agent
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class OutputFormat(str, Enum):
    """Supported output formats"""
    TEXT = "text"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    HTML_TABLE = "html_table"


class MediaType(str, Enum):
    """Supported media extraction types"""
    IMAGES = "images"
    VIDEOS = "videos"
    DOCUMENTS = "documents"
    ALL = "all"


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


class MediaData(BaseModel):
    """Model for extracted media files"""
    url: str = Field(..., description="Original URL of the media")
    media_url: str = Field(..., description="Direct URL to the media file")
    media_type: str = Field(..., description="Type of media (image, video, document)")
    filename: Optional[str] = Field(None, description="Original filename if available")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the media")
    alt_text: Optional[str] = Field(None, description="Alt text or description")
    thumbnail_base64: Optional[str] = Field(None, description="Base64 thumbnail for videos/documents")


class StructuredDataField(BaseModel):
    """Model for structured data field definitions"""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (string, number, date, url, etc.)")
    description: Optional[str] = Field(None, description="Field description")


class StructuredData(BaseModel):
    """Model for structured data records"""
    fields: List[StructuredDataField] = Field(..., description="Field definitions")
    records: List[Dict[str, Any]] = Field(..., description="Data records")
    total_records: int = Field(..., description="Total number of records found")


class BrowseRequest(BaseModel):
    """Request model for web research tasks"""
    query: str = Field(..., description="Research query or question to investigate")
    urls: List[HttpUrl] = Field(..., description="List of URLs to analyze")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens for LLM response")
    include_screenshots: Optional[bool] = Field(True, description="Whether to capture and analyze screenshots")
    output_format: Optional[OutputFormat] = Field(OutputFormat.TEXT, description="Desired output format")
    extract_media: Optional[bool] = Field(False, description="Whether to extract individual media files")
    media_types: Optional[List[MediaType]] = Field([MediaType.ALL], description="Types of media to extract")
    structured_fields: Optional[List[str]] = Field(None, description="Specific fields to extract for structured data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Find all Travis Scott concerts this year with dates, locations, and ticket prices",
                "urls": [
                    "https://example.com/concerts",
                    "https://example.com/ticketmaster"
                ],
                "max_tokens": 512,
                "include_screenshots": True,
                "output_format": "csv",
                "extract_media": True,
                "media_types": ["images"],
                "structured_fields": ["date", "location", "venue", "price", "tickets_remaining"]
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
    structured_data: Optional[StructuredData] = Field(None, description="Structured data extracted from sources")
    formatted_output: Optional[str] = Field(None, description="Data formatted in requested format (CSV, HTML, etc.)")
    media_files: Optional[List[MediaData]] = Field(None, description="Extracted media files")
    output_format: Optional[OutputFormat] = Field(OutputFormat.TEXT, description="Format of the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": "Found 5 Travis Scott concerts with complete information",
                "sources_analyzed": ["https://example.com/concerts"],
                "structured_data": {
                    "fields": [
                        {"name": "date", "type": "date", "description": "Concert date"},
                        {"name": "location", "type": "string", "description": "City and venue"}
                    ],
                    "records": [
                        {"date": "2024-03-15", "location": "Los Angeles, CA"}
                    ],
                    "total_records": 5
                },
                "formatted_output": "date,location,price\n2024-03-15,Los Angeles CA,$150",
                "output_format": "csv",
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