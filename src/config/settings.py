"""
Configuration settings for Flask application
"""

import os
from pathlib import Path


class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'pcb-defect-detection-secret-key-2024'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'data/uploads')
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', 'data/results')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/pcb_defect_yolov5n_model.pth')
    CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', 0.25))
    IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.45))
    
    # Defect severity mapping
    SEVERITY_MAP = {
        'Mouse_bite': 'WARNING',
        'Open_circuit': 'CRITICAL',
        'Short': 'CRITICAL',
        'Spur': 'WARNING',
        'Spurious_copper': 'WARNING',
        'Missing_hole': 'CRITICAL'
    }
    
    # Verdict colors for UI
    VERDICT_COLORS = {
        'PASS': '#22c55e',      # Green
        'MARGINAL': '#f59e0b',  # Orange
        'FAIL': '#ef4444'       # Red
    }
    
    # Class names (should match model training)
    CLASS_NAMES = {
        0: 'Mouse_bite',
        1: 'Open_circuit',
        2: 'Short',
        3: 'Spur',
        4: 'Spurious_copper',
        5: 'Missing_hole'
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}