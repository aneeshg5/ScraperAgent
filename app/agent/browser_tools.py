"""
Browser automation tools using Playwright and LangChain
Handles web content extraction and screenshot capture
"""
import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
import os
from urllib.parse import urlparse
import base64
from io import BytesIO
import sys
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "30000"))

# Global browser instance for reuse
_browser: Optional[Browser] = None
_browser_context: Optional[BrowserContext] = None
_playwright_instance = None


async def initialize_browser() -> bool:
    """
    Robust browser initialization with multiple fallback strategies
    Specifically designed to handle Windows Python 3.13 compatibility issues
    """
    global _browser, _browser_context, _playwright_instance
    
    if _browser and _browser_context:
        logger.info("✅ Browser already initialized")
        return True
    
    logger.info("🔧 Initializing browser automation...")
    
    # Get system info
    system = platform.system()
    python_version = sys.version_info
    
    logger.info(f"🖥️  System: {system}, Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Strategy 1: Try standard async initialization first
    if await _try_standard_initialization():
        return True
    
    # Strategy 2: For Windows Python 3.13+, try event loop policy fix
    if system == "Windows" and python_version >= (3, 13):
        logger.info("🔄 Trying Windows-specific event loop policy fix...")
        if await _try_windows_event_loop_fix():
            return True
    
    # Strategy 3: Try thread-based initialization
    logger.info("🔄 Trying thread-based initialization...")
    if await _try_thread_based_initialization():
        return True
    
    # Strategy 4: Try with minimal browser options
    logger.info("🔄 Trying minimal browser configuration...")
    if await _try_minimal_browser_config():
        return True
    
    # All strategies failed - log detailed failure info
    logger.warning("❌ All browser initialization strategies failed")
    logger.warning("🔄 Application will use requests-based fallback for web content")
    logger.warning("⚠️  Screenshots and JavaScript rendering will not be available")
    logger.info("💡 This is expected on Windows Python 3.13 - full functionality available on Linux/Brev platform")
    return False


async def _try_standard_initialization() -> bool:
    """Try standard Playwright async initialization"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        logger.info("🚀 Attempting standard Playwright initialization...")
        
        _playwright_instance = await async_playwright().start()
        _browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                '--disable-extensions',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows'
            ]
        )
        
        _browser_context = await _browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Test the browser works
        await _test_browser_functionality()
        
        logger.info("✅ Standard Playwright initialization successful!")
        return True
        
    except Exception as e:
        logger.info(f"❌ Standard initialization failed: {str(e)}")
        await _cleanup_browser_attempt()
        return False


async def _try_windows_event_loop_fix() -> bool:
    """Try Windows-specific event loop policy fix for Python 3.13+"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        logger.info("🔧 Applying Windows event loop policy fix...")
        
        # Try ProactorEventLoop policy
        original_policy = asyncio.get_event_loop_policy()
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
            _playwright_instance = await async_playwright().start()
            _browser = await _playwright_instance.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--single-process',  # Try single process mode
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            
            _browser_context = await _browser.new_context(
                viewport={'width': 1280, 'height': 720}
            )
            
            await _test_browser_functionality()
            
            logger.info("✅ Windows event loop fix successful!")
            return True
            
        except Exception as inner_e:
            # Restore original policy if this attempt fails
            asyncio.set_event_loop_policy(original_policy)
            raise inner_e
            
    except Exception as e:
        logger.info(f"❌ Windows event loop fix failed: {str(e)}")
        await _cleanup_browser_attempt()
        return False


async def _try_thread_based_initialization() -> bool:
    """Try running Playwright in a separate thread"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        logger.info("🧵 Attempting thread-based initialization...")
        
        def init_browser_sync():
            """Synchronous browser initialization in thread"""
            import asyncio
            from playwright.sync_api import sync_playwright
            
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-dev-shm-usage']
                    )
                    # Return browser instance info for async wrapper
                    return {"success": True, "browser": browser}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Run in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(init_browser_sync)
            result = future.result(timeout=30)
            
        if result.get("success"):
            logger.info("✅ Thread-based initialization successful!")
            # Note: This would need more work to properly integrate async/sync
            # For now, we'll consider this a partial success and fall back
            return False
        else:
            logger.info(f"❌ Thread-based initialization failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.info(f"❌ Thread-based initialization failed: {str(e)}")
        return False


async def _try_minimal_browser_config() -> bool:
    """Try with absolutely minimal browser configuration"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        logger.info("🔧 Trying minimal browser configuration...")
        
        _playwright_instance = await async_playwright().start()
        
        # Try Firefox as alternative
        try:
            _browser = await _playwright_instance.firefox.launch(headless=True)
            logger.info("🦊 Using Firefox as fallback browser")
        except:
            # Fall back to Chromium with minimal args
            _browser = await _playwright_instance.chromium.launch(
                headless=True,
                args=['--no-sandbox']  # Only essential arg
            )
            logger.info("🔧 Using Chromium with minimal configuration")
        
        _browser_context = await _browser.new_context()
        
        await _test_browser_functionality()
        
        logger.info("✅ Minimal browser configuration successful!")
        return True
        
    except Exception as e:
        logger.info(f"❌ Minimal browser configuration failed: {str(e)}")
        await _cleanup_browser_attempt()
        return False


async def _test_browser_functionality() -> None:
    """Test that the browser actually works"""
    if not _browser_context:
        raise Exception("Browser context not available")
    
    page = await _browser_context.new_page()
    try:
        # Test basic navigation
        await page.goto("data:text/html,<html><body><h1>Test</h1></body></html>", timeout=5000)
        
        # Test screenshot capability
        await page.screenshot(type='png')
        
        logger.info("🧪 Browser functionality test passed")
    finally:
        await page.close()


async def _cleanup_browser_attempt() -> None:
    """Clean up failed browser initialization attempt"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        if _browser_context:
            await _browser_context.close()
        if _browser:
            await _browser.close()
        if _playwright_instance:
            await _playwright_instance.stop()
    except:
        pass  # Ignore cleanup errors
    
    _browser = None
    _browser_context = None
    _playwright_instance = None


async def cleanup_browser() -> None:
    """Cleanup browser instances"""
    global _browser, _browser_context, _playwright_instance
    
    try:
        if _browser_context:
            await _browser_context.close()
            logger.info("🔄 Browser context closed")
            
        if _browser:
            await _browser.close()  
            logger.info("🔄 Browser closed")
            
        if _playwright_instance:
            await _playwright_instance.stop()
            logger.info("🔄 Playwright stopped")
            
    except Exception as e:
        logger.error(f"❌ Error during browser cleanup: {str(e)}")
    finally:
        _browser = None
        _browser_context = None
        _playwright_instance = None


async def extract_web_data(urls: List[str], capture_screenshots: bool = True, extract_media: bool = False) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """
    Extract comprehensive web data including text, screenshots, and media
    
    Args:
        urls: List of URLs to process
        capture_screenshots: Whether to capture screenshots
        extract_media: Whether to extract individual media files
        
    Returns:
        Tuple of (web_contents, screenshot_data)
    """
    web_contents = []
    screenshots = []
    
    browser_available = await initialize_browser()
    
    if browser_available and _browser:
        logger.info("🌐 Using Playwright for enhanced data extraction")
        
        try:
            for url in urls:
                try:
                    page = await _browser.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    
                    # Wait for dynamic content
                    await page.wait_for_timeout(2000)
                    
                    # Extract text content
                    text_content = await page.inner_text("body")
                    
                    # Get page HTML for media extraction
                    html_content = await page.content() if extract_media else ""
                    
                    # Extract media files if requested
                    media_files = []
                    if extract_media:
                        media_files = await extract_media_from_page(page, url, html_content)
                    
                    web_content = {
                        "url": url,
                        "text": text_content,
                        "html": html_content if extract_media else "",
                        "media_files": media_files,
                        "title": await page.title(),
                        "extracted_at": datetime.now().isoformat()
                    }
                    
                    web_contents.append(web_content)
                    
                    # Capture screenshot if requested
                    if capture_screenshots:
                        try:
                            screenshot = await page.screenshot(
                                type='png',
                                full_page=True
                            )
                            screenshots.append(screenshot)
                            logger.info(f"📸 Screenshot captured for {url}")
                        except Exception as e:
                            logger.warning(f"⚠️ Screenshot failed for {url}: {str(e)}")
                            screenshots.append(b"")
                    
                    await page.close()
                    logger.info(f"✅ Successfully processed {url}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {url}: {str(e)}")
                    # Add fallback entry
                    fallback_content = await _extract_with_requests(url)
                    if fallback_content:
                        web_contents.append(fallback_content)
                    if capture_screenshots:
                        screenshots.append(b"")
                        
        except Exception as e:
            logger.error(f"❌ Browser processing failed: {str(e)}")
            
    else:
        logger.warning("⚠️ Browser not available, using requests fallback")
        for url in urls:
            content = await _extract_with_requests(url)
            if content:
                # Extract media from HTML if requested
                if extract_media and content.get('html'):
                    from app.utils.formatters import MediaExtractor
                    extractor = MediaExtractor()
                    images = extractor.extract_images_from_html(content['html'], url)
                    videos = extractor.extract_videos_from_html(content['html'], url)
                    content['media_files'] = images + videos
                else:
                    content['media_files'] = []
                
                web_contents.append(content)
            
            if capture_screenshots:
                screenshots.append(b"")  # Empty screenshot for fallback
    
    return web_contents, screenshots


async def extract_media_from_page(page, url: str, html_content: str) -> List[Dict[str, Any]]:
    """
    Extract individual media files from a web page using both Playwright and HTML parsing
    
    Args:
        page: Playwright page object
        url: Page URL
        html_content: Page HTML content
        
    Returns:
        List of media file metadata
    """
    try:
        from app.utils.formatters import MediaExtractor
        media_files = []
        
        # Extract images using HTML parsing
        extractor = MediaExtractor()
        images = extractor.extract_images_from_html(html_content, url)
        videos = extractor.extract_videos_from_html(html_content, url)
        
        # Enhanced image extraction using Playwright for better accuracy
        try:
            img_elements = await page.query_selector_all('img')
            
            for img in img_elements:
                try:
                    src = await img.get_attribute('src')
                    alt = await img.get_attribute('alt') or ''
                    
                    if not src or src.startswith('data:'):
                        continue
                    
                    # Resolve relative URLs
                    from urllib.parse import urljoin
                    full_url = urljoin(url, src)
                    
                    # Skip small icons and placeholder images
                    try:
                        width = await img.get_attribute('width')
                        height = await img.get_attribute('height')
                        
                        if width and height:
                            w, h = int(width), int(height)
                            if w < 50 or h < 50:  # Skip very small images
                                continue
                    except (ValueError, TypeError):
                        pass
                    
                    # Check if already extracted by HTML parser
                    already_exists = any(
                        existing['media_url'] == full_url 
                        for existing in images
                    )
                    
                    if not already_exists:
                        image_data = {
                            'url': url,
                            'media_url': full_url,
                            'media_type': 'image',
                            'filename': full_url.split('/')[-1].split('?')[0],  # Remove query params
                            'alt_text': alt,
                            'mime_type': f"image/{full_url.split('.')[-1].lower()}" if '.' in full_url else 'image/unknown'
                        }
                        
                        # Try to get actual dimensions from the element
                        try:
                            bbox = await img.bounding_box()
                            if bbox:
                                image_data['width'] = int(bbox['width'])
                                image_data['height'] = int(bbox['height'])
                        except:
                            pass
                        
                        images.append(image_data)
                
                except Exception as e:
                    logger.debug(f"Failed to process image element: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"⚠️ Enhanced image extraction failed: {str(e)}")
        
        # Combine all media files
        media_files = images + videos
        
        # Extract document links (PDFs, docs, etc.)
        try:
            doc_links = await page.query_selector_all('a[href*=".pdf"], a[href*=".doc"], a[href*=".docx"], a[href*=".xlsx"], a[href*=".pptx"]')
            
            for link in doc_links:
                try:
                    href = await link.get_attribute('href')
                    text = await link.inner_text()
                    
                    if href:
                        from urllib.parse import urljoin
                        full_url = urljoin(url, href)
                        
                        # Determine file type
                        file_ext = href.split('.')[-1].lower()
                        mime_types = {
                            'pdf': 'application/pdf',
                            'doc': 'application/msword',
                            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                        }
                        
                        media_files.append({
                            'url': url,
                            'media_url': full_url,
                            'media_type': 'document',
                            'filename': href.split('/')[-1].split('?')[0],
                            'alt_text': text.strip(),
                            'mime_type': mime_types.get(file_ext, 'application/octet-stream')
                        })
                
                except Exception as e:
                    logger.debug(f"Failed to process document link: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"⚠️ Document extraction failed: {str(e)}")
        
        logger.info(f"🎭 Extracted {len(media_files)} media files from {url}")
        return media_files
        
    except Exception as e:
        logger.error(f"❌ Media extraction failed for {url}: {str(e)}")
        return []


async def _extract_with_requests(url: str) -> Optional[Dict[str, Any]]:
    """
    Fallback method to extract web content using requests and BeautifulSoup
    Enhanced with media extraction capabilities
    
    Args:
        url: URL to extract content from
        
    Returns:
        Dictionary containing extracted content and media files
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        # Get page title
        title = soup.title.string if soup.title else ""
        
        return {
            "url": url,
            "text": text_content,
            "html": str(soup),
            "media_files": [],  # Will be populated by caller if needed
            "title": title,
            "extracted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Requests fallback failed for {url}: {str(e)}")
        return None


async def _extract_single_url(
    url: str, 
    capture_screenshot: bool = True
) -> Tuple[str, Optional[bytes]]:
    """
    Extract content and optionally capture screenshot from a single URL
    """
    page = None
    try:
        logger.info(f"🔗 Processing URL: {url}")
        
        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError(f"Invalid URL format: {url}")
        
        # Create new page
        page = await _browser_context.new_page()
        
        # Set timeout and navigation
        page.set_default_timeout(BROWSER_TIMEOUT)
        
        # Navigate to URL
        await page.goto(url, wait_until='domcontentloaded', timeout=BROWSER_TIMEOUT)
        
        # Wait for content to load
        await page.wait_for_timeout(2000)  # Allow dynamic content to load
        
        # Extract content
        content = await _extract_page_content(page)
        
        # Capture screenshot if requested
        screenshot = None
        if capture_screenshot:
            screenshot = await _capture_screenshot(page)
        
        logger.info(f"✅ Successfully processed: {url}")
        return content, screenshot
        
    except Exception as e:
        logger.error(f"❌ Error processing {url}: {str(e)}")
        return "", None
        
    finally:
        if page:
            await page.close()


async def _extract_page_content(page: Page) -> str:
    """
    Extract and clean text content from a webpage
    """
    try:
        # Get page HTML
        html_content = await page.content()
        
        # Parse with BeautifulSoup for better text extraction
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit content length to avoid token limits
        if len(text) > 5000:
            text = text[:5000] + "... [content truncated]"
        
        return text
        
    except Exception as e:
        logger.error(f"❌ Content extraction error: {str(e)}")
        return ""


async def _capture_screenshot(page: Page) -> Optional[bytes]:
    """
    Capture screenshot of the current page
    """
    try:
        # Scroll to capture more content
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight/4)")
        await page.wait_for_timeout(1000)
        
        # Capture screenshot - PNG format doesn't support quality parameter
        screenshot = await page.screenshot(
            type='png',
            full_page=True  # Capture full page for better context
        )
        
        return screenshot
        
    except Exception as e:
        logger.error(f"❌ Screenshot capture error: {str(e)}")
        return None


async def search_web_content(query: str, max_results: int = 5) -> List[str]:
    """
    Perform web search and return URLs (placeholder for search integration)
    
    Note: This would integrate with search APIs like Google Custom Search, 
    Bing Search API, or other search services
    """
    logger.info(f"🔍 Searching for: {query}")
    
    # Placeholder implementation - in production, integrate with search APIs
    search_urls = [
        f"https://www.google.com/search?q={query.replace(' ', '+')}",
        f"https://www.bing.com/search?q={query.replace(' ', '+')}",
        f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
    ]
    
    return search_urls[:max_results]


async def get_page_metadata(url: str) -> dict:
    """
    Extract metadata from a webpage (title, description, etc.)
    """
    page = None
    try:
        page = await _browser_context.new_page()
        await page.goto(url, wait_until='domcontentloaded', timeout=BROWSER_TIMEOUT)
        
        # Extract metadata
        title = await page.title()
        
        # Get meta description
        description_element = await page.query_selector('meta[name="description"]')
        description = await description_element.get_attribute('content') if description_element else ""
        
        # Get other useful metadata
        metadata = {
            "title": title,
            "description": description,
            "url": url,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Metadata extraction error for {url}: {str(e)}")
        return {"title": "", "description": "", "url": url}
        
    finally:
        if page:
            await page.close()


# LangChain integration utilities
class PlaywrightBrowserTool:
    """
    LangChain-compatible tool for browser automation
    """
    
    def __init__(self):
        self.name = "playwright_browser"
        self.description = "Browser automation tool for web content extraction and screenshot capture"
    
    async def run(self, urls: List[str], capture_screenshots: bool = True) -> dict:
        """Execute browser automation with LangChain compatibility"""
        contents, screenshots = await extract_web_data(urls, capture_screenshots)
        
        return {
            "contents": contents,
            "screenshots": [base64.b64encode(s).decode() if s else None for s in screenshots],
            "urls_processed": len([c for c in contents if c]),
            "screenshots_captured": len([s for s in screenshots if s])
        }


async def test_browser_capabilities() -> dict:
    """
    Test function to verify browser capabilities
    Returns detailed information about what's working
    """
    logger.info("🧪 Testing browser capabilities...")
    
    results = {
        "browser_available": False,
        "screenshots_working": False,
        "javascript_working": False,
        "error_message": None,
        "browser_type": None,
        "fallback_mode": False
    }
    
    try:
        # Initialize browser if not already done
        await initialize_browser()
        
        if not _browser or not _browser_context:
            results["fallback_mode"] = True
            results["error_message"] = "Browser initialization failed - running in fallback mode"
            logger.info("📝 Browser test result: Fallback mode active")
            return results
        
        results["browser_available"] = True
        results["browser_type"] = _browser.__class__.__name__
        
        # Test basic page load and JavaScript
        page = await _browser_context.new_page()
        
        try:
            # Test JavaScript execution
            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>Test Page</title></head>
            <body>
                <h1 id="title">Original Title</h1>
                <div id="content">Loading...</div>
                <script>
                    document.getElementById('title').textContent = 'JavaScript Works!';
                    document.getElementById('content').innerHTML = '<p>Dynamic content loaded</p>';
                </script>
            </body>
            </html>
            """
            
            await page.goto(f"data:text/html,{html_content}")
            await page.wait_for_timeout(1000)  # Wait for JS to execute
            
            # Check if JavaScript executed
            title_text = await page.text_content('#title')
            if title_text == "JavaScript Works!":
                results["javascript_working"] = True
                logger.info("✅ JavaScript execution test passed")
            else:
                logger.warning("⚠️ JavaScript execution test failed")
            
            # Test screenshot capability
            screenshot_data = await page.screenshot(type='png', full_page=True)
            if screenshot_data and len(screenshot_data) > 1000:  # Valid screenshot should be >1KB
                results["screenshots_working"] = True
                logger.info("✅ Screenshot capture test passed")
            else:
                logger.warning("⚠️ Screenshot capture test failed")
            
            # Test with a real website
            try:
                await page.goto("https://httpbin.org/json", timeout=10000)
                await page.wait_for_load_state("networkidle", timeout=5000)
                content = await page.content()
                if "httpbin" in content.lower():
                    logger.info("✅ Real website navigation test passed")
                else:
                    logger.warning("⚠️ Real website navigation test failed")
            except Exception as e:
                logger.warning(f"⚠️ Real website test failed: {str(e)}")
            
        finally:
            await page.close()
        
        logger.info("🎉 Browser capability testing completed successfully!")
        
    except Exception as e:
        results["error_message"] = str(e)
        logger.error(f"❌ Browser capability test failed: {str(e)}")
    
    # Log summary
    status_emoji = "✅" if results["browser_available"] else "❌"
    screenshot_emoji = "📸" if results["screenshots_working"] else "📵"
    js_emoji = "🚀" if results["javascript_working"] else "⛔"
    
    logger.info(f"📊 Browser Test Summary:")
    logger.info(f"   {status_emoji} Browser Available: {results['browser_available']}")
    logger.info(f"   {screenshot_emoji} Screenshots: {results['screenshots_working']}")
    logger.info(f"   {js_emoji} JavaScript: {results['javascript_working']}")
    
    if results["browser_type"]:
        logger.info(f"   🌐 Browser Type: {results['browser_type']}")
    
    return results


# Export test function for external use
__all__ = [
    'initialize_browser', 
    'extract_web_data', 
    'cleanup_browser', 
    'PlaywrightBrowserTool',
    'test_browser_capabilities'
]


# Export the tool for LangChain integration
browser_tool = PlaywrightBrowserTool() 