"""
PCB Defect Detection using YOLOv11
Model predictor class - handles all inference logic
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PCBDefectPredictor:
    """YOLO-based PCB defect detection predictor"""
    
    def __init__(
        self, 
        model_path: str = "models/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ('cuda' if self._check_cuda() else 'cpu')
        
        # Load model
        print(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Get class names from model
        self.class_names = self.model.names
        self.num_classes = len(self.class_names)
        
        # Define severity levels for each defect type
        self.severity_map = {
            'Mouse_bite': 'WARNING',
            'Open_circuit': 'CRITICAL',
            'Short': 'CRITICAL',
            'Spur': 'WARNING',
            'Spurious_copper': 'WARNING',
            'Missing_hole': 'CRITICAL'
        }
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Detecting {self.num_classes} defect types: {list(self.class_names.values())}")
    
    @staticmethod
    def _check_cuda():
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def predict(
        self, 
        image_path: str,
        save_output: bool = False,
        output_dir: str = "data/results"
    ) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing detection results
        """
        image_path = Path(image_path)
        
        # Run YOLO inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        result = results[0]
        
        # Extract detections
        detections = []
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                class_name = self.class_names[cls_id]
                severity = self.severity_map.get(class_name, 'WARNING')
                
                detection = {
                    'class': class_name,
                    'class_id': int(cls_id),
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'severity': severity
                }
                detections.append(detection)
        
        # Determine overall verdict
        verdict = self._determine_verdict(detections)
        
        # Prepare result dictionary
        result_dict = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'timestamp': datetime.now().isoformat(),
            'verdict': verdict,
            'num_detections': len(detections),
            'detections': detections,
            'defect_summary': self._get_defect_summary(detections),
            'model_params': {
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'model': self.model_path.name
            }
        }
        
        # Save annotated image if requested
        if save_output:
            output_path = self._save_annotated_image(result, image_path, output_dir)
            result_dict['annotated_image_path'] = output_path
        
        return result_dict
    
    def predict_batch(
        self,
        image_paths: List[str],
        save_outputs: bool = False,
        output_dir: str = "data/results"
    ) -> List[Dict]:
        """Run inference on multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, save_outputs, output_dir)
            results.append(result)
        return results
    
    def predict_from_array(self, image: np.ndarray) -> Dict:
        """
        Run inference on numpy array (OpenCV image)
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Detection results
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        detections = []
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                class_name = self.class_names[cls_id]
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'severity': self.severity_map.get(class_name, 'WARNING')
                }
                detections.append(detection)
        
        verdict = self._determine_verdict(detections)
        
        return {
            'verdict': verdict,
            'num_detections': len(detections),
            'detections': detections,
            'defect_summary': self._get_defect_summary(detections)
        }
    
    def _determine_verdict(self, detections: List[Dict]) -> str:
        """Determine overall verdict based on detections"""
        if not detections:
            return 'PASS'
        
        critical_defects = [d for d in detections if d['severity'] == 'CRITICAL']
        if critical_defects:
            return 'FAIL'
        
        warning_defects = [d for d in detections if d['severity'] == 'WARNING']
        if len(warning_defects) >= 3:
            return 'MARGINAL'
        elif warning_defects:
            return 'MARGINAL'
        
        return 'PASS'
    
    def _get_defect_summary(self, detections: List[Dict]) -> Dict:
        """Get summary of detected defects by type"""
        summary = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in summary:
                summary[class_name] = {
                    'count': 0,
                    'severity': detection['severity'],
                    'avg_confidence': 0.0
                }
            summary[class_name]['count'] += 1
        
        for class_name in summary:
            class_detections = [d for d in detections if d['class'] == class_name]
            avg_conf = sum(d['confidence'] for d in class_detections) / len(class_detections)
            summary[class_name]['avg_confidence'] = round(avg_conf, 3)
        
        return summary
    
    def _save_annotated_image(
        self,
        result,
        image_path: Path,
        output_dir: str
    ) -> str:
        """Save annotated image with bounding boxes"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotated_img = result.plot()
        
        output_path = output_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        
        return str(output_path)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': str(self.model_path),
            'model_type': 'YOLOv11n',
            'task': 'Object Detection',
            'num_classes': self.num_classes,
            'classes': self.class_names,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }