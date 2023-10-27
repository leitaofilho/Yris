# vision.py
from ultralytics import YOLO
import os


class Vision:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image_path, confidence_threshold=0.4):
        output_dir = 'outputs'
        output_name = 'output_detection'
        result = self.model(source=image_path, show=False, conf=confidence_threshold, save=True, project=output_dir,
                            name=output_name)
        # O caminho do arquivo de saída será 'outputs/output_detection.jpg'
        return result
