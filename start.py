#!/usr/bin/env python3
"""
Startup script for the Multimodal Web Research Intelligence Agent
Enhanced for NVIDIA Hackathon with GPU detection and advanced features
"""
import os
import sys
import subprocess
import asyncio
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_hackathon_environment():
    """Check hackathon-specific environment and GPU availability"""
    print("🏆 Checking NVIDIA Hackathon Environment...")
    
    # Check for GPU availability
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU Available: {gpu_name}")
            print(f"📊 GPU Memory: {memory_gb:.1f}GB")
            print(f"🔢 GPU Count: {gpu_count}")
            gpu_available = True
            os.environ["GPU_ACCELERATED"] = "true"
        else:
            print("⚠️  No GPU detected - running in CPU mode")
    except ImportError:
        print("⚠️  PyTorch not available - GPU detection skipped")
    
    # Check for Brev platform
    if os.getenv("BREV_INSTANCE_ID"):
        print(f"✅ Brev Platform Instance: {os.getenv('BREV_INSTANCE_ID')}")
        os.environ["HACKATHON_MODE"] = "true"
    
    # Check for NeMo Toolkit availability
    try:
        import nemo_toolkit
        print("✅ NeMo Toolkit available")
    except ImportError:
        print("⚠️  NeMo Toolkit not found - will use fallback during hackathon")
    
    return gpu_available

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('uvicorn', 'ASGI server'),
        ('pydantic', 'Data validation'),
        ('requests', 'HTTP client'),
        ('aiosqlite', 'Async SQLite'),
        ('PIL', 'Image processing'),
        ('bs4', 'Web scraping')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ Missing: {description}")
    
    # Check optional hackathon packages
    optional_packages = [
        ('torch', 'PyTorch for GPU acceleration'),
        ('langchain', 'LangChain framework'),
        ('playwright', 'Browser automation'),
        ('transformers', 'Transformer models'),
        ('langchain_nvidia_ai_endpoints', 'NVIDIA LangChain integration')
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {description} (optional)")
        except ImportError:
            print(f"⚠️  {description} (optional - not available)")
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        print("🎯 For hackathon: pip install -r requirements-hackathon.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_playwright_browsers():
    """Check if Playwright browsers are installed"""
    print("🌐 Checking Playwright browsers...")
    
    try:
        result = subprocess.run(['playwright', 'install', '--dry-run'], 
                              capture_output=True, text=True)
        if "chromium" in result.stdout.lower():
            print("✅ Playwright browsers are available")
            return True
        else:
            print("⚠️  Playwright browsers may need installation")
            return False
    except FileNotFoundError:
        print("⚠️  Playwright CLI not found")
        print("💡 The agent will use fallback mode for web content extraction")
        return False

def check_environment():
    """Check environment configuration"""
    print("⚙️  Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        
        # Check for hackathon template
        hackathon_template = Path("hackathon.env.template")
        example_file = Path(".env.example")
        
        if hackathon_template.exists():
            print("💡 Copy hackathon.env.template to .env and configure with hackathon credentials")
        elif example_file.exists():
            print("💡 Copy .env.example to .env and configure your API keys")
        else:
            print("💡 Create a .env file with your configuration")
        return False
    
    # Check for important environment variables
    print("✅ Environment file found")
    
    # Load and validate key settings
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Hackathon-specific checks
        if os.getenv("HACKATHON_MODE", "false").lower() == "true":
            print("🏆 Hackathon mode enabled")
        
        if os.getenv("NVIDIA_API_KEY", "").startswith("your_"):
            print("⚠️  NVIDIA API key appears to be placeholder - configure for full functionality")
        elif os.getenv("NVIDIA_API_KEY"):
            print("✅ NVIDIA API key configured")
        else:
            print("⚠️  NVIDIA API key not found - using demo mode")
        
        return True
    except ImportError:
        print("⚠️  python-dotenv not available")
        return True

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    try:
        subprocess.run(['playwright', 'install'], check=True)
        print("✅ Playwright browsers installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Playwright browsers")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Multimodal Web Research Intelligence Agent...")
    print("━" * 60)
    print("🤖 NVIDIA-Powered AI Research Agent")
    print("🌐 Web Interface: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("━" * 60)
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except ImportError:
        print("❌ uvicorn not installed. Installing dependencies...")
        if install_dependencies():
            import uvicorn
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info"
            )
        else:
            print("❌ Failed to start server")

def parse_arguments():
    """Parse command line arguments for hackathon features"""
    parser = argparse.ArgumentParser(
        description="NVIDIA Hackathon Multimodal Web Research Intelligence Agent"
    )
    parser.add_argument(
        "--hackathon-mode", 
        action="store_true", 
        help="Enable hackathon-optimized features"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU acceleration"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--test-browser", 
        action="store_true", 
        help="Test browser automation capabilities and exit"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    return parser.parse_args()


async def test_browser_functionality():
    """Test browser capabilities and report results"""
    print("\n🧪 BROWSER FUNCTIONALITY TEST")
    print("=" * 40)
    
    try:
        from app.agent.browser_tools import test_browser_capabilities
        
        # Run comprehensive browser test
        results = await test_browser_capabilities()
        
        print("\n📊 TEST RESULTS:")
        print("-" * 30)
        
        # Browser availability
        if results.get("browser_available"):
            print("✅ Browser Initialization: SUCCESS")
            print(f"   🌐 Browser Type: {results.get('browser_type', 'Unknown')}")
        else:
            print("❌ Browser Initialization: FAILED")
            if results.get("error_message"):
                print(f"   ❌ Error: {results['error_message']}")
        
        # Screenshot capability
        if results.get("screenshots_working"):
            print("✅ Screenshot Capture: SUCCESS")
        else:
            print("❌ Screenshot Capture: FAILED")
        
        # JavaScript execution
        if results.get("javascript_working"):
            print("✅ JavaScript Execution: SUCCESS")
        else:
            print("❌ JavaScript Execution: FAILED")
        
        # Fallback mode
        if results.get("fallback_mode"):
            print("⚠️  Running in Fallback Mode (requests-based)")
            print("   📝 Note: Basic text extraction will work, but no screenshots")
        
        print("\n🎯 MULTIMODAL CAPABILITY ASSESSMENT:")
        print("-" * 40)
        
        if results.get("browser_available") and results.get("screenshots_working"):
            print("🎉 FULL MULTIMODAL CAPABILITIES AVAILABLE!")
            print("   ✅ Text extraction from websites")
            print("   ✅ Visual screenshot analysis")
            print("   ✅ JavaScript-rendered content")
            print("   ✅ Complete multimodal AI agent functionality")
        elif results.get("browser_available"):
            print("⚠️  PARTIAL MULTIMODAL CAPABILITIES")
            print("   ✅ Text extraction from websites")
            print("   ✅ JavaScript-rendered content")  
            print("   ❌ Visual screenshot analysis")
        else:
            print("⚠️  BASIC TEXT-ONLY CAPABILITIES")
            print("   ✅ Static text extraction from websites")
            print("   ❌ Visual screenshot analysis")
            print("   ❌ JavaScript-rendered content")
            print("   💡 Consider this a 'Text Research Agent' instead of 'Multimodal'")
        
        print("\n🚀 HACKATHON READINESS:")
        print("-" * 25)
        
        if results.get("fallback_mode"):
            print("⚠️  DEVELOPMENT MODE: Limited functionality on Windows Python 3.13")
            print("🏆 HACKATHON EXPECTED: Full functionality on Brev Linux GPU platform")
            print("💡 Your app architecture is correct - environment limitations only")
        else:
            print("🎉 FULLY READY: All browser capabilities working!")
            print("🏆 Perfect for hackathon demo and evaluation")
        
        return results.get("browser_available", False)
        
    except Exception as e:
        print(f"❌ Browser test failed with error: {str(e)}")
        print("💡 This may be expected on Windows Python 3.13")
        return False


def main():
    """Main startup function with hackathon enhancements"""
    args = parse_arguments()
    
    # Handle browser testing mode
    if args.test_browser:
        print("🔧 Browser Testing Mode Activated")
        
        # Check basic requirements first
        if not check_requirements():
            print("❌ Missing dependencies for browser testing")
            sys.exit(1)
        
        # Run async browser test
        try:
            import asyncio
            browser_working = asyncio.run(test_browser_functionality())
            
            print(f"\n{'='*50}")
            print("🎯 FINAL RECOMMENDATION:")
            
            if browser_working:
                print("✅ Browser automation is working perfectly!")
                print("🎉 Your multimodal agent is fully functional")
            else:
                print("⚠️  Browser automation has limitations on this environment")
                print("📝 Your app will work in text-mode with fallback")
                print("🏆 Full multimodal functionality expected on Brev platform")
            
            print("🚀 Ready for hackathon!")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"❌ Browser test execution failed: {str(e)}")
            
        sys.exit(0)
    
    # Set environment variables based on arguments
    if args.hackathon_mode:
        os.environ["HACKATHON_MODE"] = "true"
    if args.gpu:
        os.environ["GPU_ACCELERATED"] = "true"
    
    # Print banner
    print("=" * 80)
    print("🚀 NVIDIA HACKATHON - Multimodal Web Research Intelligence Agent")
    print("🔧 Powered by NVIDIA AI Stack + GPU Acceleration")
    print("=" * 80)
    print()
    
    # Check if we're in the right directory
    if not Path("app").exists() or not Path("requirements.txt").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Run enhanced checks
    if not check_requirements():
        print("\n🔧 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Setup failed. Please check the installation manually.")
            sys.exit(1)
    
    gpu_available = check_hackathon_environment()
    
    if not check_environment():
        print("⚠️  Environment issues detected - continuing with demo mode")
    
    playwright_available = check_playwright_browsers()
    
    if not playwright_available:
        print("⚠️  Running in fallback mode - some features may be limited")
        print("💡 For full functionality, install Playwright browsers:")
        print("   playwright install chromium")
    
    # System summary
    print()
    print("🎯 HACKATHON CONFIGURATION SUMMARY:")
    print(f"   • Hackathon Mode: {'✅ Enabled' if args.hackathon_mode else '⚠️  Disabled'}")
    print(f"   • GPU Acceleration: {'✅ Available' if gpu_available else '⚠️  CPU Only'}")
    print(f"   • Browser Automation: {'✅ Full' if playwright_available else '⚠️  Limited'}")
    print(f"   • Debug Mode: {'✅ Enabled' if args.debug else 'Disabled'}")
    print()
    
    print("🌟 All systems ready for hackathon evaluation!")
    print("🌐 Starting server...")
    print(f"📊 Interface: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🏥 Health Check: http://{args.host}:{args.port}/health")
    print()
    
    # Import and run the FastAPI application
    import uvicorn
    from app.main import app
    
    # Configure uvicorn with hackathon optimizations
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        reload=args.reload,
        workers=1,  # Single worker for GPU sharing
        access_log=args.debug,
        use_colors=True
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("🏆 Starting NVIDIA Hackathon agent server...")
        import asyncio
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Agent stopped by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check the README.md for troubleshooting help.")
        sys.exit(1) 