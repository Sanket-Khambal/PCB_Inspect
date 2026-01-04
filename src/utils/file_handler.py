"""
File handling utilities for uploads and storage
"""

import os
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions
        
    Returns:
        True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def save_uploaded_file(file, upload_folder: str) -> str:
    """
    Save uploaded file with timestamp
    
    Args:
        file: File object from request
        upload_folder: Directory to save file
        
    Returns:
        Path to saved file
    """
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(upload_folder, unique_filename)
    
    os.makedirs(upload_folder, exist_ok=True)
    file.save(filepath)
    
    return filepath


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Remove files older than specified hours
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours
    """
    if not os.path.exists(directory):
        return
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    print(f"Removed old file: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
        
    Returns:
        Human-readable file size
    """
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def ensure_directories(*directories):
    """
    Create directories if they don't exist
    
    Args:
        *directories: Variable number of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")


def copy_file_with_timestamp(src: str, dst_dir: str) -> str:
    """
    Copy file to destination with timestamp prefix
    
    Args:
        src: Source file path
        dst_dir: Destination directory
        
    Returns:
        Path to copied file
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    filename = Path(src).name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_filename = f"{timestamp}_{filename}"
    dst_path = os.path.join(dst_dir, new_filename)
    
    shutil.copy2(src, dst_path)
    
    return dst_path