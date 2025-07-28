"""
Structured data extraction utilities for the Multimodal Web Research Intelligence Agent
Uses AI-powered analysis to extract specific data points from unstructured web content
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class StructuredDataExtractor:
    """Handles extraction of structured data from web content using AI analysis"""
    
    def __init__(self, nvidia_api_key: str, nim_endpoint: str):
        self.nvidia_api_key = nvidia_api_key
        self.nim_endpoint = nim_endpoint
    
    async def extract_structured_data(
        self, 
        web_contents: List[Dict[str, Any]], 
        query: str, 
        structured_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from web content based on the query and specified fields
        
        Args:
            web_contents: List of web content dictionaries
            query: Original research query
            structured_fields: Specific fields to extract
            
        Returns:
            Dictionary containing structured data with fields and records
        """
        try:
            logger.info(f"🔍 Extracting structured data for query: {query}")
            
            # Auto-detect fields if not specified
            if not structured_fields:
                structured_fields = self._auto_detect_fields(query)
            
            logger.info(f"📊 Extracting fields: {structured_fields}")
            
            # Extract data from each source
            all_records = []
            field_definitions = []
            
            for content in web_contents:
                records = await self._extract_from_content(
                    content, query, structured_fields
                )
                all_records.extend(records)
            
            # Generate field definitions
            field_definitions = self._generate_field_definitions(structured_fields, all_records)
            
            # Clean and deduplicate records
            clean_records = self._clean_and_deduplicate(all_records)
            
            structured_data = {
                'fields': field_definitions,
                'records': clean_records,
                'total_records': len(clean_records)
            }
            
            logger.info(f"✅ Extracted {len(clean_records)} structured records")
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Structured data extraction failed: {str(e)}")
            return {
                'fields': [],
                'records': [],
                'total_records': 0,
                'error': str(e)
            }
    
    def _auto_detect_fields(self, query: str) -> List[str]:
        """
        Auto-detect likely fields based on the query
        
        Args:
            query: Research query
            
        Returns:
            List of likely field names
        """
        # Common field patterns
        field_patterns = {
            r'\b(date|dates|when|time|schedule)\b': ['date', 'time'],
            r'\b(location|where|venue|place|city|address)\b': ['location', 'venue', 'address'],
            r'\b(price|cost|ticket|fee|amount|money|\$)\b': ['price', 'cost'],
            r'\b(name|title|event|concert|show)\b': ['name', 'title'],
            r'\b(phone|contact|email)\b': ['phone', 'email', 'contact'],
            r'\b(description|details|info)\b': ['description'],
            r'\b(available|remaining|stock|quantity)\b': ['availability', 'quantity'],
            r'\b(rating|review|score)\b': ['rating', 'reviews'],
            r'\b(website|url|link)\b': ['website', 'url']
        }
        
        detected_fields = []
        query_lower = query.lower()
        
        for pattern, fields in field_patterns.items():
            if re.search(pattern, query_lower):
                detected_fields.extend(fields)
        
        # Remove duplicates while preserving order
        unique_fields = []
        for field in detected_fields:
            if field not in unique_fields:
                unique_fields.append(field)
        
        # Fallback to common fields if nothing detected
        if not unique_fields:
            unique_fields = ['name', 'description', 'url']
        
        logger.info(f"🔍 Auto-detected fields from query: {unique_fields}")
        return unique_fields
    
    async def _extract_from_content(
        self, 
        content: Dict[str, Any], 
        query: str, 
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data from a single web content source
        
        Args:
            content: Web content dictionary
            query: Research query
            fields: Fields to extract
            
        Returns:
            List of extracted records
        """
        try:
            text_content = content.get('text', '')[:8000]  # Limit content length
            source_url = content.get('url', '')
            
            if not text_content.strip():
                return []
            
            # Prepare extraction prompt
            extraction_prompt = self._build_extraction_prompt(
                text_content, query, fields, source_url
            )
            
            # Get structured data from AI
            extracted_data = await self._get_ai_extraction(extraction_prompt)
            
            if extracted_data and isinstance(extracted_data, list):
                # Add source URL to each record
                for record in extracted_data:
                    if isinstance(record, dict):
                        record['source_url'] = source_url
                return extracted_data
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed for {content.get('url', 'unknown')}: {str(e)}")
            return []
    
    def _build_extraction_prompt(
        self, 
        text_content: str, 
        query: str, 
        fields: List[str], 
        source_url: str
    ) -> str:
        """
        Build the AI prompt for structured data extraction
        
        Args:
            text_content: Web page text content
            query: Original research query
            fields: Fields to extract
            source_url: Source URL
            
        Returns:
            Formatted extraction prompt
        """
        fields_list = ", ".join(fields)
        
        prompt = f"""You are a data extraction specialist. Extract structured information from the following web content.

ORIGINAL QUERY: {query}
SOURCE URL: {source_url}
FIELDS TO EXTRACT: {fields_list}

WEB CONTENT:
{text_content}

INSTRUCTIONS:
1. Find all relevant information that matches the query
2. Extract data for these specific fields: {fields_list}
3. Return data as a JSON array of objects
4. Each object should have the specified fields as keys
5. If a field is not found, use null or empty string
6. Extract multiple records if multiple items are found
7. Be precise and accurate - only extract information that is clearly stated
8. For dates, use YYYY-MM-DD format when possible
9. For prices, include currency symbols and amounts
10. For locations, be as specific as possible

EXAMPLE OUTPUT FORMAT:
[
  {{
    "name": "Example Event",
    "date": "2024-03-15",
    "location": "New York, NY",
    "price": "$50"
  }}
]

Extract the data now:"""

        return prompt
    
    async def _get_ai_extraction(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get structured data extraction from AI
        
        Args:
            prompt: Extraction prompt
            
        Returns:
            List of extracted records or None if failed
        """
        if not self.nvidia_api_key:
            logger.warning("⚠️ NVIDIA API key not available for extraction")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(self.nim_endpoint, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON from response
                try:
                    # Extract JSON from response (might have additional text)
                    json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        extracted_data = json.loads(json_str)
                        
                        if isinstance(extracted_data, list):
                            logger.info(f"✅ AI extracted {len(extracted_data)} records")
                            return extracted_data
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Failed to parse AI response as JSON: {str(e)}")
                    # Try to extract data using fallback method
                    return self._fallback_extraction(ai_response)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ AI extraction request failed: {str(e)}")
            return None
    
    def _fallback_extraction(self, ai_response: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fallback extraction method when JSON parsing fails
        
        Args:
            ai_response: AI response text
            
        Returns:
            List of extracted records or None
        """
        try:
            # Simple pattern-based extraction as fallback
            records = []
            
            # Look for structured patterns in the response
            lines = ai_response.split('\n')
            current_record = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('//'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('"').lower()
                        value = parts[1].strip().strip(',').strip('"')
                        
                        if value and value.lower() not in ['null', 'none', 'n/a']:
                            current_record[key] = value
                
                # Check if this looks like end of a record
                if line == '}' or (line == '' and current_record):
                    if current_record:
                        records.append(current_record.copy())
                        current_record = {}
            
            # Add final record if exists
            if current_record:
                records.append(current_record)
            
            if records:
                logger.info(f"✅ Fallback extraction found {len(records)} records")
                return records
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fallback extraction failed: {str(e)}")
            return None
    
    def _generate_field_definitions(
        self, 
        field_names: List[str], 
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate field definitions with inferred types
        
        Args:
            field_names: List of field names
            records: Sample records to infer types from
            
        Returns:
            List of field definition dictionaries
        """
        field_definitions = []
        
        for field_name in field_names:
            field_type = "string"  # Default type
            description = field_name.replace('_', ' ').title()
            
            # Infer type from sample data
            sample_values = []
            for record in records[:10]:  # Check first 10 records
                if field_name in record and record[field_name]:
                    sample_values.append(record[field_name])
            
            if sample_values:
                field_type = self._infer_field_type(sample_values)
            
            # Add semantic descriptions
            if 'date' in field_name.lower() or 'time' in field_name.lower():
                description = f"Date/time information for {field_name}"
                if field_type == "string":
                    field_type = "date"
            elif 'price' in field_name.lower() or 'cost' in field_name.lower():
                description = f"Price or cost information"
                if field_type == "string":
                    field_type = "currency"
            elif 'url' in field_name.lower() or 'link' in field_name.lower():
                description = f"URL or web link"
                field_type = "url"
            elif 'email' in field_name.lower():
                description = f"Email address"
                field_type = "email"
            elif 'phone' in field_name.lower():
                description = f"Phone number"
                field_type = "phone"
            
            field_definitions.append({
                'name': field_name,
                'type': field_type,
                'description': description
            })
        
        return field_definitions
    
    def _infer_field_type(self, sample_values: List[Any]) -> str:
        """
        Infer field type from sample values
        
        Args:
            sample_values: List of sample values
            
        Returns:
            Inferred field type
        """
        if not sample_values:
            return "string"
        
        # Check for numeric values
        numeric_count = 0
        date_count = 0
        url_count = 0
        email_count = 0
        
        for value in sample_values:
            str_value = str(value).strip()
            
            # Check for numbers (including currency)
            if re.match(r'^[\$€£¥]?[\d,]+\.?\d*$', str_value.replace(',', '')):
                numeric_count += 1
            
            # Check for dates
            if re.match(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', str_value):
                date_count += 1
            
            # Check for URLs
            if str_value.startswith(('http://', 'https://', 'www.')):
                url_count += 1
            
            # Check for emails
            if '@' in str_value and '.' in str_value:
                email_count += 1
        
        total_samples = len(sample_values)
        
        # Determine type based on majority
        if url_count > total_samples * 0.5:
            return "url"
        elif email_count > total_samples * 0.5:
            return "email"
        elif date_count > total_samples * 0.5:
            return "date"
        elif numeric_count > total_samples * 0.5:
            return "number"
        else:
            return "string"
    
    def _clean_and_deduplicate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and deduplicate extracted records
        
        Args:
            records: List of extracted records
            
        Returns:
            Cleaned and deduplicated records
        """
        if not records:
            return []
        
        # Clean records
        cleaned_records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            
            cleaned_record = {}
            for key, value in record.items():
                # Clean key
                clean_key = str(key).strip().lower().replace(' ', '_')
                
                # Clean value
                if value is None or str(value).strip().lower() in ['null', 'none', 'n/a', '']:
                    clean_value = ''
                else:
                    clean_value = str(value).strip()
                
                cleaned_record[clean_key] = clean_value
            
            # Only add records with meaningful content
            if any(value for value in cleaned_record.values() if value):
                cleaned_records.append(cleaned_record)
        
        # Simple deduplication based on record similarity
        unique_records = []
        for record in cleaned_records:
            is_duplicate = False
            
            for existing in unique_records:
                # Check if records are too similar (>80% field overlap)
                matching_fields = 0
                total_fields = max(len(record), len(existing))
                
                for key, value in record.items():
                    if key in existing and existing[key] == value:
                        matching_fields += 1
                
                similarity = matching_fields / total_fields if total_fields > 0 else 0
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_records.append(record)
        
        logger.info(f"🧹 Cleaned records: {len(records)} → {len(unique_records)} (removed {len(records) - len(unique_records)} duplicates)")
        return unique_records 