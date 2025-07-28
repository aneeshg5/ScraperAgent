"""
Output formatting utilities for the Multimodal Web Research Intelligence Agent
Supports CSV, Excel, JSON, and HTML table generation from structured data
"""
import csv
import json
import io
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("⚠️ Excel support not available. Install openpyxl and pandas for Excel output.")


class OutputFormatter:
    """Handles formatting structured data into various output formats"""
    
    @staticmethod
    def format_as_csv(structured_data: Dict[str, Any]) -> str:
        """
        Format structured data as CSV string
        
        Args:
            structured_data: Dictionary with 'fields' and 'records' keys
            
        Returns:
            CSV formatted string
        """
        try:
            if not structured_data or 'records' not in structured_data:
                return "No structured data available"
            
            records = structured_data['records']
            if not records:
                return "No records found"
            
            # Create CSV in memory
            output = io.StringIO()
            
            # Get fieldnames from first record or fields definition
            if 'fields' in structured_data and structured_data['fields']:
                fieldnames = [field['name'] for field in structured_data['fields']]
            else:
                fieldnames = list(records[0].keys()) if records else []
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                # Clean record to only include defined fields
                clean_record = {field: record.get(field, '') for field in fieldnames}
                writer.writerow(clean_record)
            
            csv_content = output.getvalue()
            output.close()
            
            logger.info(f"✅ Generated CSV with {len(records)} records")
            return csv_content
            
        except Exception as e:
            logger.error(f"❌ CSV formatting failed: {str(e)}")
            return f"Error generating CSV: {str(e)}"
    
    @staticmethod
    def format_as_excel(structured_data: Dict[str, Any]) -> Optional[str]:
        """
        Format structured data as Excel file (base64 encoded)
        
        Args:
            structured_data: Dictionary with 'fields' and 'records' keys
            
        Returns:
            Base64 encoded Excel file or None if Excel not available
        """
        if not EXCEL_AVAILABLE:
            logger.warning("⚠️ Excel formatting not available - missing dependencies")
            return None
        
        try:
            if not structured_data or 'records' not in structured_data:
                return None
            
            records = structured_data['records']
            if not records:
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(records)
            
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Research Results')
                
                # Get the workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Research Results']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            excel_buffer.seek(0)
            excel_base64 = base64.b64encode(excel_buffer.getvalue()).decode('utf-8')
            excel_buffer.close()
            
            logger.info(f"✅ Generated Excel file with {len(records)} records")
            return excel_base64
            
        except Exception as e:
            logger.error(f"❌ Excel formatting failed: {str(e)}")
            return None
    
    @staticmethod
    def format_as_json(structured_data: Dict[str, Any]) -> str:
        """
        Format structured data as JSON string
        
        Args:
            structured_data: Dictionary with 'fields' and 'records' keys
            
        Returns:
            JSON formatted string
        """
        try:
            if not structured_data:
                return json.dumps({"error": "No structured data available"}, indent=2)
            
            # Add metadata
            output_data = {
                "metadata": {
                    "total_records": structured_data.get('total_records', len(structured_data.get('records', []))),
                    "fields": structured_data.get('fields', []),
                    "generated_at": datetime.now().isoformat()
                },
                "data": structured_data.get('records', [])
            }
            
            json_content = json.dumps(output_data, indent=2, default=str)
            
            logger.info(f"✅ Generated JSON with {len(structured_data.get('records', []))} records")
            return json_content
            
        except Exception as e:
            logger.error(f"❌ JSON formatting failed: {str(e)}")
            return json.dumps({"error": f"JSON formatting failed: {str(e)}"}, indent=2)
    
    @staticmethod
    def format_as_html_table(structured_data: Dict[str, Any]) -> str:
        """
        Format structured data as HTML table
        
        Args:
            structured_data: Dictionary with 'fields' and 'records' keys
            
        Returns:
            HTML table string
        """
        try:
            if not structured_data or 'records' not in structured_data:
                return "<p>No structured data available</p>"
            
            records = structured_data['records']
            if not records:
                return "<p>No records found</p>"
            
            # Get field definitions for headers
            fields = structured_data.get('fields', [])
            if fields:
                headers = [(field.get('name', ''), field.get('description', '')) for field in fields]
            else:
                headers = [(key, key) for key in records[0].keys()] if records else []
            
            # Build HTML table
            html_parts = [
                '<div class="data-table-container">',
                '<table class="data-table" border="1" cellpadding="8" cellspacing="0">',
                '<thead>',
                '<tr>'
            ]
            
            # Add header row
            for name, description in headers:
                title_attr = f'title="{description}"' if description and description != name else ''
                html_parts.append(f'<th {title_attr}>{name}</th>')
            
            html_parts.extend(['</tr>', '</thead>', '<tbody>'])
            
            # Add data rows
            for record in records:
                html_parts.append('<tr>')
                for name, _ in headers:
                    value = record.get(name, '')
                    # Handle different data types
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)
                    html_parts.append(f'<td>{str(value)}</td>')
                html_parts.append('</tr>')
            
            html_parts.extend([
                '</tbody>',
                '</table>',
                f'<p class="record-count">Total records: {len(records)}</p>',
                '</div>'
            ])
            
            # Add CSS styling
            css = """
            <style>
            .data-table-container {
                margin: 20px 0;
                overflow-x: auto;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
            }
            .data-table th {
                background-color: #f8f9fa;
                font-weight: 600;
                text-align: left;
                border: 1px solid #dee2e6;
                padding: 12px 8px;
            }
            .data-table td {
                border: 1px solid #dee2e6;
                padding: 8px;
                vertical-align: top;
            }
            .data-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .data-table tr:hover {
                background-color: #e9ecef;
            }
            .record-count {
                font-size: 12px;
                color: #6c757d;
                margin-top: 10px;
            }
            </style>
            """
            
            html_content = css + ''.join(html_parts)
            
            logger.info(f"✅ Generated HTML table with {len(records)} records")
            return html_content
            
        except Exception as e:
            logger.error(f"❌ HTML table formatting failed: {str(e)}")
            return f"<p>Error generating HTML table: {str(e)}</p>"


class MediaExtractor:
    """Handles extraction of individual media files from web pages"""
    
    @staticmethod
    def extract_images_from_html(html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract image URLs and metadata from HTML content
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image metadata dictionaries
        """
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html_content, 'html.parser')
            images = []
            
            # Find all img tags
            img_tags = soup.find_all('img')
            
            for img in img_tags:
                src = img.get('src')
                if not src:
                    continue
                
                # Resolve relative URLs
                full_url = urljoin(base_url, src)
                
                # Skip data URLs and very small images (likely icons)
                if src.startswith('data:') or 'icon' in src.lower():
                    continue
                
                image_data = {
                    'url': base_url,
                    'media_url': full_url,
                    'media_type': 'image',
                    'filename': urlparse(full_url).path.split('/')[-1],
                    'alt_text': img.get('alt', ''),
                    'mime_type': 'image/' + (full_url.split('.')[-1].lower() if '.' in full_url else 'unknown')
                }
                
                # Try to get dimensions
                width = img.get('width')
                height = img.get('height')
                if width and height:
                    try:
                        image_data['width'] = int(width)
                        image_data['height'] = int(height)
                    except ValueError:
                        pass
                
                images.append(image_data)
            
            logger.info(f"✅ Extracted {len(images)} images from {base_url}")
            return images
            
        except Exception as e:
            logger.error(f"❌ Image extraction failed for {base_url}: {str(e)}")
            return []
    
    @staticmethod
    def extract_videos_from_html(html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract video URLs and metadata from HTML content
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of video metadata dictionaries
        """
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html_content, 'html.parser')
            videos = []
            
            # Find video tags
            video_tags = soup.find_all('video')
            for video in video_tags:
                src = video.get('src')
                if src:
                    full_url = urljoin(base_url, src)
                    videos.append({
                        'url': base_url,
                        'media_url': full_url,
                        'media_type': 'video',
                        'filename': urlparse(full_url).path.split('/')[-1],
                        'mime_type': 'video/' + (full_url.split('.')[-1].lower() if '.' in full_url else 'unknown')
                    })
                
                # Check source tags within video
                source_tags = video.find_all('source')
                for source in source_tags:
                    src = source.get('src')
                    if src:
                        full_url = urljoin(base_url, src)
                        videos.append({
                            'url': base_url,
                            'media_url': full_url,
                            'media_type': 'video',
                            'filename': urlparse(full_url).path.split('/')[-1],
                            'mime_type': source.get('type', 'video/unknown')
                        })
            
            # Find embedded YouTube/Vimeo videos
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                src = iframe.get('src', '')
                if 'youtube.com' in src or 'youtu.be' in src or 'vimeo.com' in src:
                    videos.append({
                        'url': base_url,
                        'media_url': src,
                        'media_type': 'video',
                        'filename': 'embedded_video',
                        'mime_type': 'video/embedded'
                    })
            
            logger.info(f"✅ Extracted {len(videos)} videos from {base_url}")
            return videos
            
        except Exception as e:
            logger.error(f"❌ Video extraction failed for {base_url}: {str(e)}")
            return [] 