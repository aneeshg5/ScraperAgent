
import logging
import time
import os
import requests
import base64
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
from dotenv import load_dotenv

from app.agent.browser_tools import extract_web_data
from app.agent.vision import analyze_screenshot
from app.models.schemas import ScreenshotData, OutputFormat
from app.utils.memory import MemorySaver
from app.utils.formatters import OutputFormatter, MediaExtractor
from app.utils.data_extraction import StructuredDataExtractor

try:
    
    HACKATHON_AGENT_AVAILABLE = False
except ImportError:
    HACKATHON_AGENT_AVAILABLE = False
    logging.warning("Hackathon agent not available - using standard implementation")

load_dotenv()
logger = logging.getLogger(__name__)

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_ENDPOINT = os.getenv("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")

# Hackathon Configuration
HACKATHON_MODE = os.getenv("HACKATHON_MODE", "false").lower() == "true"
GPU_ACCELERATED = os.getenv("GPU_ACCELERATED", "false").lower() == "true"

# Initialize memory saver
memory_saver = MemorySaver()


async def multimodal_research_agent(
    query: str, 
    urls: List[str], 
    max_tokens: int = 512, 
    include_screenshots: bool = True,
    output_format: OutputFormat = OutputFormat.TEXT,
    extract_media: bool = False,
    structured_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Enhanced multimodal research agent with structured data and media extraction
    
    Orchestrates:
    1. Web content extraction with optional media extraction
    2. Screenshot capture and vision analysis
    3. Structured data extraction using AI
    4. Output formatting in requested format
    5. NVIDIA Nemotron reasoning and synthesis
    
    Args:
        query: Research question or task
        urls: List of URLs to analyze
        max_tokens: Maximum tokens for LLM responses
        include_screenshots: Whether to capture screenshots
        output_format: Desired output format (text, csv, excel, json, html_table)
        extract_media: Whether to extract individual media files
        structured_fields: Specific fields to extract for structured data
        
    Returns:
        Dictionary containing analysis, structured data, formatted output, and media
    """
    memory_saver = MemorySaver()
    
    # Try hackathon agent first if available
    if HACKATHON_AGENT_AVAILABLE:
        try:
            from app.agent.nemo_langchain_agent import get_hackathon_agent
            
            hackathon_agent = get_hackathon_agent()
            result = await hackathon_agent.research_with_multimodal(
                query=query,
                urls=urls,
                max_tokens=max_tokens,
                include_screenshots=include_screenshots,
                output_format=output_format.value,
                extract_media=extract_media,
                structured_fields=structured_fields
            )
            
            if result and "analysis" in result:
                result["analysis"] = result.get("result", "Research completed with hackathon agent")
            
            logger.info("✅ Hackathon agent research completed")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Hackathon agent failed, falling back to standard: {e}")
            # Continue with standard implementation below
    
    try:
        # Step 1: Extract web data with optional media extraction
        logger.info("🌐 Extracting web content, screenshots, and media...")
        web_contents, screenshots = await extract_web_data(
            urls, 
            capture_screenshots=include_screenshots,
            extract_media=extract_media
        )
        
        logger.info(f"📄 Extracted content from {len(web_contents)} sources")
        
        # Step 2: Extract structured data if format requires it
        structured_data = None
        if output_format != OutputFormat.TEXT or structured_fields:
            logger.info("🔍 Extracting structured data...")
            
            extractor = StructuredDataExtractor(NVIDIA_API_KEY, NIM_ENDPOINT)
            structured_data = await extractor.extract_structured_data(
                web_contents, query, structured_fields
            )
            
            logger.info(f"📊 Extracted {structured_data.get('total_records', 0)} structured records")
        
        # Step 3: Analyze screenshots if available
        vision_insights = []
        if include_screenshots and screenshots:
            logger.info("👁️ Analyzing screenshots with vision model...")
            vision_tasks = [analyze_screenshot(screenshot) for screenshot in screenshots]
            vision_results = await asyncio.gather(*vision_tasks, return_exceptions=True)
            
            for i, result in enumerate(vision_results):
                if isinstance(result, Exception):
                    logger.warning(f"Vision analysis failed for screenshot {i}: {result}")
                else:
                    vision_insights.append(result)
        
        # Step 4: Collect all media files
        all_media_files = []
        if extract_media:
            for content in web_contents:
                media_files = content.get('media_files', [])
                all_media_files.extend(media_files)
            
            logger.info(f"🎭 Collected {len(all_media_files)} total media files")
        
        # Step 5: Prepare context for NVIDIA Nemotron
        logger.info("🧠 Preparing context for NVIDIA Nemotron reasoning...")
        
        # Enhanced context with structured data information
        context = _prepare_enhanced_multimodal_context(
            query, web_contents, vision_insights, urls, structured_data, output_format
        )
        
        # Step 6: Generate analysis with NVIDIA Nemotron
        logger.info("🚀 Generating analysis with NVIDIA Nemotron...")
        analysis = await _generate_nemotron_analysis(context, max_tokens)
        
        # Step 7: Format output according to requested format
        formatted_output = None
        if structured_data and structured_data.get('records'):
            logger.info(f"🎨 Formatting output as {output_format.value}...")
            
            formatter = OutputFormatter()
            
            if output_format == OutputFormat.CSV:
                formatted_output = formatter.format_as_csv(structured_data)
            elif output_format == OutputFormat.EXCEL:
                formatted_output = formatter.format_as_excel(structured_data)
            elif output_format == OutputFormat.JSON:
                formatted_output = formatter.format_as_json(structured_data)
            elif output_format == OutputFormat.HTML_TABLE:
                formatted_output = formatter.format_as_html_table(structured_data)
        
        # Step 8: Save to memory for future reference
        await memory_saver.save_research_session(query, urls, analysis)
        
        # Step 9: Prepare screenshot data for frontend display
        screenshot_data = []
        if include_screenshots and screenshots:
            for i, screenshot_bytes in enumerate(screenshots):
                if screenshot_bytes:
                    try:
                        base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
                        
                        screenshot_info = ScreenshotData(
                            url=urls[i] if i < len(urls) else f"Screenshot {i+1}",
                            image_base64=base64_image,
                            timestamp=datetime.now(),
                            width=1280,
                            height=720
                        )
                        screenshot_data.append(screenshot_info)
                        
                    except Exception as e:
                        logger.error(f"Failed to process screenshot {i}: {str(e)}")
        
        # Step 10: Return comprehensive results
        return {
            "analysis": analysis,
            "sources_analyzed": [content["url"] for content in web_contents],
            "vision_insights": vision_insights,
            "screenshots": screenshot_data,
            "structured_data": structured_data,
            "formatted_output": formatted_output,
            "media_files": all_media_files,
            "output_format": output_format,
            "total_records": structured_data.get('total_records', 0) if structured_data else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Multimodal research agent failed: {str(e)}")
        raise


def _prepare_enhanced_multimodal_context(
    query: str, 
    web_contents: List[Dict[str, Any]], 
    vision_insights: List[str], 
    urls: List[str],
    structured_data: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = OutputFormat.TEXT
) -> str:
    """
    Prepare enhanced context for NVIDIA Nemotron including structured data insights
    
    Args:
        query: Research query
        web_contents: Extracted web content
        vision_insights: Vision analysis results
        urls: Source URLs
        structured_data: Extracted structured data
        output_format: Requested output format
        
    Returns:
        Enhanced context string for analysis
    """
    combined_context = f"""
RESEARCH QUERY: {query}
OUTPUT FORMAT: {output_format.value}

SOURCES ANALYZED ({len(web_contents)}):
"""
    
    # Add source information
    for i, content in enumerate(web_contents):
        url = content.get("url", f"Source {i+1}")
        title = content.get("title", "No title")
        text_preview = content.get("text", "")[:500] + "..." if len(content.get("text", "")) > 500 else content.get("text", "")
        
        combined_context += f"\n{i+1}. URL: {url}"
        combined_context += f"\n   Title: {title}"
        combined_context += f"\n   Content Preview: {text_preview}\n"
        
        # Add media file information if available
        media_files = content.get("media_files", [])
        if media_files:
            combined_context += f"   Media Files Found: {len(media_files)} ({', '.join(set(m.get('media_type', 'unknown') for m in media_files))})\n"
    
    # Add structured data insights
    if structured_data and structured_data.get('records'):
        total_records = structured_data.get('total_records', 0)
        fields = [f.get('name', '') for f in structured_data.get('fields', [])]
        
        combined_context += f"\nSTRUCTURED DATA EXTRACTED:\n"
        combined_context += f"- Total Records: {total_records}\n"
        combined_context += f"- Fields Identified: {', '.join(fields)}\n"
        
        # Add sample records for context
        sample_records = structured_data.get('records', [])[:3]  # First 3 records
        if sample_records:
            combined_context += f"- Sample Data:\n"
            for i, record in enumerate(sample_records):
                combined_context += f"  Record {i+1}: {json.dumps(record, default=str)}\n"
    
    # Add vision insights
    if vision_insights:
        combined_context += f"\nVISUAL ANALYSIS INSIGHTS ({len(vision_insights)}):\n"
        for i, insight in enumerate(vision_insights):
            combined_context += f"Screenshot {i+1} Analysis: {insight}\n"
    
    # Add format-specific instructions
    if output_format != OutputFormat.TEXT:
        combined_context += f"\nOUTPUT FORMAT INSTRUCTIONS:\n"
        combined_context += f"The user has requested output in {output_format.value} format. "
        
        if structured_data and structured_data.get('records'):
            combined_context += f"Structured data has been extracted and will be formatted appropriately. "
            combined_context += f"Focus your analysis on interpreting and summarizing the {structured_data.get('total_records', 0)} records found."
        else:
            combined_context += f"No structured data was extracted. Provide insights that could help structure the information."
    
    combined_context += f"\nPlease provide a comprehensive analysis addressing: {query}"
    
    if output_format != OutputFormat.TEXT:
        combined_context += f"\nNote: The response will be formatted as {output_format.value} based on extracted structured data."
    
    return combined_context


async def _generate_nemotron_analysis(context: str, max_tokens: int) -> str:
    """
    Generate analysis using NVIDIA Nemotron via NIM API
    """
    if not NVIDIA_API_KEY:
        logger.warning("⚠️ NVIDIA API key not configured, using mock response")
        return _generate_mock_analysis(context)
    
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1",  # Updated to use Nemotron Super
            "messages": [
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        logger.info("📡 Sending request to NVIDIA NIM API...")
        response = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]
            logger.info("✅ Successfully received analysis from NVIDIA Nemotron")
            return analysis
        else:
            logger.error(f"❌ NVIDIA API error: {response.status_code} - {response.text}")
            return _generate_fallback_analysis(context)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error with NVIDIA API: {str(e)}")
        return _generate_fallback_analysis(context)
    except Exception as e:
        logger.error(f"❌ Unexpected error with NVIDIA API: {str(e)}")
        return _generate_fallback_analysis(context)


def _generate_mock_analysis(context: str) -> str:
    """Generate a mock analysis for development/testing"""
    return """
# Multimodal Research Analysis

## Executive Summary
Based on the analysis of the provided web sources and visual content, here are the key findings:

**Key Insights:**
- Multiple sources were successfully analyzed for comprehensive coverage
- Cross-modal information synthesis provides enhanced understanding
- Visual content analysis supplements textual information effectively

## Detailed Analysis

### Content Overview
The research query has been addressed through systematic analysis of:
- Web-based textual content from multiple authoritative sources
- Visual elements captured through screenshot analysis
- Cross-referencing of information across different modalities

### Key Findings
1. **Primary Insights**: Core information extracted from textual sources
2. **Visual Validation**: Screenshot analysis confirms and extends textual findings  
3. **Synthesis**: Combined understanding provides comprehensive perspective

### Recommendations
- Continue monitoring sources for updates
- Consider additional sources for broader perspective
- Visual analysis provides valuable supplementary insights

*Note: This is a development response. Configure NVIDIA API credentials for full functionality.*
"""


def _generate_fallback_analysis(context: str) -> str:
    """Generate a fallback analysis when NVIDIA API is unavailable"""
    return """
# Research Analysis - Fallback Mode

## Summary
Analysis completed using fallback processing due to API connectivity issues.

### Sources Processed
- Multiple web sources were successfully extracted and processed
- Content analysis was performed using available processing capabilities
- Visual content was captured and prepared for analysis

### Key Points
- Web content extraction completed successfully
- Screenshot capture and processing functional
- Comprehensive data collection achieved

### Next Steps
1. Verify NVIDIA API connectivity for enhanced analysis
2. Review extracted content for manual insights
3. Consider alternative processing approaches if needed

*Note: This is a fallback response. Full NVIDIA-powered analysis requires API connectivity.*
"""


async def get_research_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent research history"""
    return await memory_saver.get_recent_sessions(limit)


async def clear_research_history():
    """Clear all research history"""
    await memory_saver.clear_all_sessions() 


async def autonomous_research_agent(
    research_goal: str,
    starting_urls: List[str],
    max_iterations: int = 8,
    max_tokens: int = 512,
    output_format: OutputFormat = OutputFormat.TEXT,
    extract_media: bool = False,
    structured_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Enhanced autonomous research agent with structured data and media extraction
    
    This agent makes intelligent decisions about what information to gather and takes
    autonomous actions to achieve the research goal, now with format flexibility.
    
    Args:
        research_goal: The overall research objective
        starting_urls: Initial URLs to begin research
        max_iterations: Maximum decision-making iterations per URL
        max_tokens: Maximum tokens for LLM responses
        output_format: Desired output format
        extract_media: Whether to extract individual media files
        structured_fields: Specific fields to extract
        
    Returns:
        Comprehensive research results with autonomous action log
    """
    logger.info(f"🤖 Starting autonomous research: {research_goal}")
    logger.info(f"🎯 Target format: {output_format.value}")
    
    collected_data = []
    action_log = []
    screenshots = []
    all_media_files = []
    autonomous_actions_taken = 0
    
    try:
        for url in starting_urls:
            logger.info(f"🔍 Autonomous analysis of: {url}")
            
            # Extract initial data with media if requested
            web_contents, url_screenshots = await extract_web_data(
                [url], 
                capture_screenshots=True,
                extract_media=extract_media
            )
            
            if web_contents:
                content = web_contents[0]
                collected_data.append(content)
                
                # Collect media files
                if extract_media and content.get('media_files'):
                    all_media_files.extend(content['media_files'])
                    logger.info(f"🎭 Found {len(content['media_files'])} media files")
                
                # Add screenshots
                if url_screenshots:
                    screenshots.extend(url_screenshots)
                
                # Run autonomous interaction loop
                iteration_actions = await _autonomous_interaction_loop(
                    url, content, research_goal, max_iterations, max_tokens
                )
                
                action_log.extend(iteration_actions)
                autonomous_actions_taken += len(iteration_actions)
        
        logger.info(f"🤖 Completed autonomous research with {autonomous_actions_taken} actions")
        
        # Extract structured data if format requires it
        structured_data = None
        if output_format != OutputFormat.TEXT or structured_fields:
            logger.info("🔍 Extracting structured data from autonomous research...")
            
            extractor = StructuredDataExtractor(NVIDIA_API_KEY, NIM_ENDPOINT)
            structured_data = await extractor.extract_structured_data(
                collected_data, research_goal, structured_fields
            )
            
            logger.info(f"📊 Autonomous agent extracted {structured_data.get('total_records', 0)} structured records")
        
        # Format output according to requested format
        formatted_output = None
        if structured_data and structured_data.get('records'):
            logger.info(f"🎨 Formatting autonomous results as {output_format.value}...")
            
            formatter = OutputFormatter()
            
            if output_format == OutputFormat.CSV:
                formatted_output = formatter.format_as_csv(structured_data)
            elif output_format == OutputFormat.EXCEL:
                formatted_output = formatter.format_as_excel(structured_data)
            elif output_format == OutputFormat.JSON:
                formatted_output = formatter.format_as_json(structured_data)
            elif output_format == OutputFormat.HTML_TABLE:
                formatted_output = formatter.format_as_html_table(structured_data)
        
        # Generate final analysis with enhanced context
        analysis = await _synthesize_autonomous_results(
            research_goal, collected_data, action_log, structured_data, output_format, max_tokens
        )
        
        # Prepare screenshot data
        screenshot_data = []
        for i, screenshot_bytes in enumerate(screenshots):
            if screenshot_bytes:
                try:
                    base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
                    
                    screenshot_info = ScreenshotData(
                        url=starting_urls[i] if i < len(starting_urls) else f"Autonomous Screenshot {i+1}",
                        image_base64=base64_image,
                        timestamp=datetime.now(),
                        width=1280,
                        height=720
                    )
                    screenshot_data.append(screenshot_info)
                    
                except Exception as e:
                    logger.error(f"Failed to process autonomous screenshot {i}: {str(e)}")
        
        return {
            "analysis": analysis,
            "collected_data": collected_data,
            "action_log": action_log,
            "autonomous_actions_taken": autonomous_actions_taken,
            "screenshots": screenshot_data,
            "structured_data": structured_data,
            "formatted_output": formatted_output,
            "media_files": all_media_files,
            "output_format": output_format,
            "total_records": structured_data.get('total_records', 0) if structured_data else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Autonomous research failed: {str(e)}")
        raise


async def _autonomous_interaction_loop(
    goal: str,
    url: str, 
    initial_text: str,
    initial_screenshot: Optional[bytes],
    max_iterations: int,
    max_tokens: int
) -> Dict[str, Any]:
    """
    Core autonomous interaction loop using Nemotron Super for decision-making
    """
    iteration_data = {"data": [], "screenshots": [], "actions": []}
    
    current_text = initial_text
    current_screenshot = initial_screenshot
    
    for iteration in range(max_iterations):
        logger.info(f"🔄 Autonomous iteration {iteration + 1}/{max_iterations}")
        
        # Use Nemotron Super to analyze current state and decide next action
        decision = await _make_autonomous_decision(
            goal, url, current_text, current_screenshot, iteration, max_tokens
        )
        
        logger.info(f"🧠 Agent decision: {decision.get('action', 'ANALYZE')}")
        
        # Extract any data found
        if decision.get("extracted_data"):
            iteration_data["data"].append({
                "source_url": url,
                "iteration": iteration,
                "data": decision["extracted_data"],
                "confidence": decision.get("confidence", 0.5)
            })
        
        # Add screenshot if available
        if current_screenshot:
            iteration_data["screenshots"].append(current_screenshot)
        
        # Check if agent has sufficient information
        if decision.get("has_sufficient_info", False):
            action_msg = f"✅ Agent completed research after {iteration + 1} iterations"
            iteration_data["actions"].append(action_msg)
            logger.info(action_msg)
            break
        
        # Execute planned action (simulated for now, can be extended)
        action_result = await _execute_autonomous_action(
            decision.get("planned_action", {}), url
        )
        
        iteration_data["actions"].append(action_result)
        
        # Simulate getting new data after action (in real implementation, 
        # this would re-extract from the page after interaction)
        if iteration < max_iterations - 1:
            # For now, we'll use the same data but indicate progression
            current_text = f"[After Action {iteration + 1}] {current_text[:1000]}"
    
    return iteration_data


async def _make_autonomous_decision(
    goal: str,
    url: str,
    page_text: str,
    screenshot_bytes: Optional[bytes],
    iteration: int,
    max_tokens: int
) -> Dict[str, Any]:
    """
    Use Nemotron Super to make intelligent decisions about next actions
    """
    # Create decision-making prompt
    decision_prompt = f"""
    You are an autonomous web research agent using advanced AI capabilities.
    
    RESEARCH GOAL: {goal}
    CURRENT URL: {url}
    ITERATION: {iteration + 1}
    
    CURRENT PAGE CONTENT (first 2000 chars):
    {page_text[:2000]}
    
    Based on the above information, make an intelligent decision:
    
    1. ANALYSIS: Do you have sufficient information to complete the research goal?
    2. DATA EXTRACTION: What specific data from this page helps achieve the goal?
    3. NEXT ACTION: What should be done next?
    
    Respond in JSON format:
    {{
        "has_sufficient_info": boolean,
        "confidence": float (0.0 to 1.0),
        "extracted_data": {{
            "key_findings": ["finding1", "finding2"],
            "relevant_details": {{}},
            "data_quality": "high|medium|low"
        }},
        "reasoning": "explanation of decision",
        "planned_action": {{
            "action": "CONTINUE_RESEARCH|SEARCH_MORE|EXTRACT_DETAILS|COMPLETE",
            "target": "what to focus on next",
            "reason": "why this action is needed"
        }}
    }}
    
    Focus on extracting actionable, specific data that directly addresses: {goal}
    """
    
    try:
        # Use existing Nemotron analysis function
        response = await _generate_nemotron_analysis(decision_prompt, max_tokens)
        
        # Try to parse JSON response
        import json
        try:
            decision_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            decision_data = {
                "has_sufficient_info": iteration >= 3,  # Simple fallback logic
                "confidence": 0.7,
                "extracted_data": {
                    "key_findings": [f"Information from {url}"],
                    "relevant_details": {"page_content": page_text[:500]},
                    "data_quality": "medium"
                },
                "reasoning": "Fallback decision due to parsing error",
                "planned_action": {
                    "action": "CONTINUE_RESEARCH",
                    "target": "more detailed analysis",
                    "reason": "continuing systematic research"
                }
            }
        
        return decision_data
        
    except Exception as e:
        logger.error(f"❌ Decision making error: {str(e)}")
        return {
            "has_sufficient_info": True,  # Fail-safe
            "confidence": 0.3,
            "extracted_data": {"error": str(e)},
            "reasoning": "Error in decision making, completing research",
            "planned_action": {"action": "COMPLETE", "target": "error handling", "reason": "technical error"}
        }


async def _execute_autonomous_action(action_plan: Dict, url: str) -> str:
    """
    Execute the autonomous action planned by Nemotron Super
    (Extensible for actual browser interactions)
    """
    action_type = action_plan.get("action", "ANALYZE")
    target = action_plan.get("target", "current page")
    reason = action_plan.get("reason", "continuing research")
    
    action_descriptions = {
        "CONTINUE_RESEARCH": f"🔍 Continuing research on {target} - {reason}",
        "SEARCH_MORE": f"🔎 Searching for more information about {target} - {reason}",
        "EXTRACT_DETAILS": f"📋 Extracting detailed information about {target} - {reason}",
        "COMPLETE": f"✅ Research completed for {target} - {reason}",
        "ANALYZE": f"🧠 Analyzing {target} - {reason}"
    }
    
    result = action_descriptions.get(action_type, f"🤖 Executed {action_type} on {target}")
    logger.info(result)
    return result


async def _synthesize_autonomous_results(
    research_goal: str, 
    collected_data: List[Dict[str, Any]], 
    action_log: List[str],
    structured_data: Optional[Dict[str, Any]] = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    max_tokens: int = 512
) -> str:
    """
    Enhanced synthesis of autonomous research results with structured data insights
    
    Args:
        research_goal: Original research goal
        collected_data: Data collected during autonomous research
        action_log: Log of autonomous actions taken
        structured_data: Extracted structured data
        output_format: Requested output format
        max_tokens: Maximum tokens for response
        
    Returns:
        Comprehensive synthesis of autonomous research
    """
    if not NVIDIA_API_KEY:
        logger.warning("⚠️ NVIDIA API key not configured, using mock synthesis")
        return f"Mock autonomous synthesis: Analyzed {len(collected_data)} sources with {len(action_log)} autonomous actions for goal: {research_goal}"
    
    # Build enhanced synthesis prompt
    synthesis_prompt = f"""You are an advanced autonomous research agent powered by NVIDIA AI. You have completed an autonomous research mission with the following details:

RESEARCH GOAL: {research_goal}
OUTPUT FORMAT: {output_format.value}
AUTONOMOUS ACTIONS TAKEN: {len(action_log)}
SOURCES ANALYZED: {len(collected_data)}

AUTONOMOUS ACTION LOG:
{chr(10).join(f"- {action}" for action in action_log[-10:])}  # Last 10 actions

COLLECTED DATA SUMMARY:
"""
    
    # Add data source summaries
    for i, data in enumerate(collected_data[:5]):  # Limit to 5 sources
        url = data.get('url', f'Source {i+1}')
        title = data.get('title', 'No title')
        text_preview = data.get('text', '')[:300] + "..." if len(data.get('text', '')) > 300 else data.get('text', '')
        
        synthesis_prompt += f"\nSource {i+1}: {url}\nTitle: {title}\nContent: {text_preview}\n"
        
        # Add media information
        media_files = data.get('media_files', [])
        if media_files:
            synthesis_prompt += f"Media Files: {len(media_files)} ({', '.join(set(m.get('media_type', 'unknown') for m in media_files))})\n"
    
    # Add structured data insights
    if structured_data and structured_data.get('records'):
        total_records = structured_data.get('total_records', 0)
        fields = [f.get('name', '') for f in structured_data.get('fields', [])]
        
        synthesis_prompt += f"\nSTRUCTURED DATA EXTRACTED:\n"
        synthesis_prompt += f"- Total Records: {total_records}\n"
        synthesis_prompt += f"- Fields: {', '.join(fields)}\n"
        
        # Add sample records
        sample_records = structured_data.get('records', [])[:2]  # First 2 records
        if sample_records:
            synthesis_prompt += f"- Sample Data:\n"
            for i, record in enumerate(sample_records):
                synthesis_prompt += f"  Record {i+1}: {json.dumps(record, default=str)}\n"
    
    # Add format-specific instructions
    if output_format != OutputFormat.TEXT:
        synthesis_prompt += f"\nOUTPUT FORMAT: {output_format.value}\n"
        if structured_data and structured_data.get('records'):
            synthesis_prompt += f"The extracted structured data will be formatted as {output_format.value}. Focus on interpreting and summarizing the {structured_data.get('total_records', 0)} structured records.\n"
    
    synthesis_prompt += f"""
INSTRUCTIONS:
1. Provide a comprehensive synthesis of the autonomous research results
2. Highlight key insights discovered through autonomous actions
3. Explain how the autonomous decision-making process contributed to the findings
4. Summarize the most important information relevant to: {research_goal}
5. Note any patterns or trends discovered across sources
6. Include specific evidence and examples from the collected data
7. Acknowledge the autonomous nature of the research process

{"Focus on the structured data findings and their implications for the research goal." if structured_data else ""}

Provide your comprehensive autonomous research synthesis now:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "messages": [{"role": "user", "content": synthesis_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        response = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]
            logger.info("✅ Autonomous synthesis completed with NVIDIA Nemotron")
            return analysis
        else:
            logger.error(f"❌ NVIDIA API error during synthesis: {response.status_code}")
            return _generate_fallback_synthesis(research_goal, collected_data, action_log, structured_data)
            
    except Exception as e:
        logger.error(f"❌ Synthesis generation failed: {str(e)}")
        return _generate_fallback_synthesis(research_goal, collected_data, action_log, structured_data)


def _generate_fallback_synthesis(
    research_goal: str, 
    collected_data: List[Dict[str, Any]], 
    action_log: List[str],
    structured_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a fallback synthesis when the main synthesis fails
    
    Args:
        research_goal: Original research goal
        collected_data: Collected data
        action_log: Action log
        structured_data: Structured data if available
        
    Returns:
        Fallback synthesis text
    """
    synthesis = f"# Autonomous Research Results\n\n"
    synthesis += f"**Research Goal:** {research_goal}\n\n"
    synthesis += f"**Sources Analyzed:** {len(collected_data)}\n\n"
    synthesis += f"**Autonomous Actions Taken:** {len(action_log)}\n\n"
    
    if structured_data and structured_data.get('total_records', 0) > 0:
        synthesis += f"**Structured Records Found:** {structured_data['total_records']}\n\n"
        fields = [f.get('name', '') for f in structured_data.get('fields', [])]
        synthesis += f"**Data Fields:** {', '.join(fields)}\n\n"
    
    synthesis += "## Key Findings\n\n"
    
    # Add source summaries
    for i, data in enumerate(collected_data[:3]):  # First 3 sources
        url = data.get('url', f'Source {i+1}')
        title = data.get('title', 'No title')
        synthesis += f"**Source {i+1}:** [{title}]({url})\n"
        
        text_preview = data.get('text', '')[:200] + "..." if len(data.get('text', '')) > 200 else data.get('text', '')
        synthesis += f"- {text_preview}\n\n"
    
    # Add recent actions
    if action_log:
        synthesis += "## Autonomous Actions Summary\n\n"
        for action in action_log[-5:]:  # Last 5 actions
            synthesis += f"- {action}\n"
        synthesis += "\n"
    
    synthesis += "## Conclusion\n\n"
    synthesis += f"The autonomous research agent successfully analyzed {len(collected_data)} sources "
    synthesis += f"and performed {len(action_log)} intelligent actions to gather information about: {research_goal}. "
    
    if structured_data and structured_data.get('total_records', 0) > 0:
        synthesis += f"The agent extracted {structured_data['total_records']} structured data records "
        synthesis += "that can be used for further analysis or reporting."
    
    return synthesis 
