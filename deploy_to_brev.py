# deploy_to_brev.py
import requests
import zipfile
import os

def create_deployment_package():
    """Create a deployment package for Brev"""
    
    # Files to exclude
    exclude_patterns = [
        'venv/', '.git/', '__pycache__/', '*.pyc', 
        '.env', 'agent_memory.db', 'logs/'
    ]
    
    print("📦 Creating deployment package...")
    
    # Create a zip file with your project
    with zipfile.ZipFile('brev_deployment.zip', 'w') as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(d.startswith(pattern.rstrip('/')) for pattern in exclude_patterns)]
            
            for file in files:
                if not any(file.endswith(pattern.lstrip('*')) for pattern in exclude_patterns):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)
    
    print("✅ Deployment package created: brev_deployment.zip")
    print("📤 Upload this file manually through Brev web interface")

if __name__ == "__main__":
    create_deployment_package()