# Test script to check if YOLOv5 is working correctly

from yolov5 import detect

# Run detection on a sample image (ensure the 'data/images/bus.jpg' exists or modify the path)
detect.run(source='data/images/bus.jpg')
