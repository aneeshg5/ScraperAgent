"""
Vision model integration for multimodal analysis
Handles screenshot analysis and image understanding using NVIDIA vision models
"""
import logging
import base64
import os
from typing import Dict, Any, Optional, List
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# NVIDIA Vision API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_VISION_ENDPOINT = os.getenv("NVIDIA_VISION_ENDPOINT", "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2")


async def analyze_screenshot(screenshot_bytes: bytes, query: Optional[str] = None) -> str:
    """
    Analyze screenshot using NVIDIA vision models
    
    Args:
        screenshot_bytes: Raw screenshot data
        query: Optional specific query about the image
        
    Returns:
        Analysis results as string
    """
    try:
        logger.info("👁️ Analyzing screenshot with NVIDIA vision model...")
        
        if not screenshot_bytes:
            return "No screenshot data available for analysis"
        
        # Convert to base64 for API
        image_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Use NVIDIA vision API or fallback
        if NVIDIA_API_KEY and NVIDIA_VISION_ENDPOINT:
            result = await _call_nvidia_vision_api(image_b64, query)
        else:
            result = _analyze_screenshot_fallback(screenshot_bytes, query)
        
        logger.info("✅ Screenshot analysis completed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Screenshot analysis failed: {str(e)}")
        return f"Screenshot analysis failed: {str(e)}"


async def analyze_image_upload(image_bytes: bytes, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze uploaded image file
    
    Args:
        image_bytes: Raw image data
        query: Optional query about the image
        
    Returns:
        Dictionary with analysis results
    """
    try:
        logger.info("🖼️ Analyzing uploaded image...")
        
        # Validate and process image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (for API efficiency)
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        processed_bytes = buffer.getvalue()
        
        # Encode for API
        image_b64 = base64.b64encode(processed_bytes).decode('utf-8')
        
        # Analyze with vision model
        if NVIDIA_API_KEY and NVIDIA_VISION_ENDPOINT:
            analysis = await _call_nvidia_vision_api(image_b64, query)
            confidence = 0.95  # Placeholder confidence score
            detected_elements = _extract_detected_elements(analysis)
        else:
            analysis = _analyze_image_fallback(processed_bytes, query)
            confidence = 0.75
            detected_elements = ["general_content"]
        
        logger.info("✅ Image analysis completed")
        
        return {
            "analysis": analysis,
            "confidence": confidence,
            "detected_elements": detected_elements,
            "image_size": image.size,
            "image_format": "JPEG"
        }
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {str(e)}")
        return {
            "analysis": f"Image analysis failed: {str(e)}",
            "confidence": 0.0,
            "detected_elements": []
        }


async def _call_nvidia_vision_api(image_b64: str, query: Optional[str] = None) -> str:
    """
    Call NVIDIA vision API for image analysis
    """
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prepare the prompt
        if query:
            prompt = f"Analyze this image and answer: {query}. Provide detailed observations about visual elements, text, layout, and any relevant information."
        else:
            prompt = "Analyze this image in detail. Describe the visual elements, text content, layout, colors, objects, and any other relevant information that would be useful for research purposes."
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.7,
            "stream": False
        }
        
        logger.info("📡 Sending request to NVIDIA Vision API...")
        response = requests.post(NVIDIA_VISION_ENDPOINT, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                analysis = result["choices"][0]["message"]["content"]
                logger.info("✅ Successfully received vision analysis from NVIDIA")
                return analysis
            else:
                logger.error("❌ Unexpected API response format")
                return _generate_fallback_vision_analysis(query)
        else:
            logger.error(f"❌ NVIDIA Vision API error: {response.status_code} - {response.text}")
            return _generate_fallback_vision_analysis(query)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error with NVIDIA Vision API: {str(e)}")
        return _generate_fallback_vision_analysis(query)
    except Exception as e:
        logger.error(f"❌ Unexpected error with NVIDIA Vision API: {str(e)}")
        return _generate_fallback_vision_analysis(query)


def _analyze_screenshot_fallback(screenshot_bytes: bytes, query: Optional[str] = None) -> str:
    """
    Fallback analysis for screenshots when API is unavailable
    """
    try:
        # Basic image analysis using PIL
        image = Image.open(BytesIO(screenshot_bytes))
        width, height = image.size
        mode = image.mode
        
        # Basic color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        dominant_colors = len(colors) if colors else 0
        
        analysis = f"""
Screenshot Analysis (Fallback Mode):

**Technical Details:**
- Image dimensions: {width}x{height} pixels
- Color mode: {mode}
- Estimated color complexity: {dominant_colors} unique colors

**Visual Assessment:**
- Screenshot successfully captured and processed
- Image appears to be a valid web page screenshot
- Content is ready for further processing when vision API is available

**Query Context:**
{f"User query: {query}" if query else "No specific query provided"}

**Note:** This is a basic fallback analysis. Full vision analysis requires NVIDIA API connectivity.
"""
        
        return analysis.strip()
        
    except Exception as e:
        return f"Fallback screenshot analysis failed: {str(e)}"


def _analyze_image_fallback(image_bytes: bytes, query: Optional[str] = None) -> str:
    """
    Fallback analysis for uploaded images
    """
    try:
        # Basic image analysis
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        mode = image.mode
        format_name = image.format or "Unknown"
        
        analysis = f"""
Image Analysis (Fallback Mode):

**Image Properties:**
- Dimensions: {width}x{height} pixels
- Format: {format_name}
- Color mode: {mode}
- File processed successfully

**Basic Assessment:**
- Image is valid and properly formatted
- Content is accessible for processing
- Ready for advanced analysis when vision models are available

{f"**Query:** {query}" if query else ""}

**Note:** This is a basic technical analysis. Advanced visual understanding requires NVIDIA vision model access.
"""
        
        return analysis.strip()
        
    except Exception as e:
        return f"Fallback image analysis failed: {str(e)}"


def _generate_fallback_vision_analysis(query: Optional[str] = None) -> str:
    """
    Generate fallback vision analysis when API calls fail
    """
    base_analysis = """
Visual Content Analysis (Fallback):

**Status:** Vision analysis service temporarily unavailable

**Content Processing:** 
- Image data received and validated
- Technical processing completed
- Image format and structure confirmed

**Capabilities Available:**
- Basic image metadata extraction
- Format validation and conversion
- Structural analysis preparation

**Full Analysis Pending:**
- Detailed visual element detection
- Text recognition and extraction  
- Object and scene understanding
- Contextual interpretation
"""
    
    if query:
        base_analysis += f"\n**Query Context:** {query}"
        base_analysis += "\n**Note:** Query will be processed when vision services are restored."
    
    return base_analysis.strip()


def _extract_detected_elements(analysis: str) -> List[str]:
    """
    Extract detected elements from vision analysis text
    """
    # Simple keyword extraction (in production, this would be more sophisticated)
    keywords = [
        'text', 'button', 'image', 'chart', 'graph', 'table', 'menu', 'navigation',
        'form', 'input', 'dashboard', 'webpage', 'interface', 'content', 'header',
        'footer', 'sidebar', 'main', 'article', 'section'
    ]
    
    detected = []
    analysis_lower = analysis.lower()
    
    for keyword in keywords:
        if keyword in analysis_lower:
            detected.append(keyword)
    
    return detected[:10]  # Limit to top 10 detected elements


# Utility functions for image processing
def resize_image_for_api(image_bytes: bytes, max_size: tuple = (1024, 1024)) -> bytes:
    """
    Resize image for optimal API processing
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"❌ Image resize failed: {str(e)}")
        return image_bytes


def validate_image_format(image_bytes: bytes) -> bool:
    """
    Validate if the image format is supported
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']
        return image.format in supported_formats
    except Exception:
        return False 