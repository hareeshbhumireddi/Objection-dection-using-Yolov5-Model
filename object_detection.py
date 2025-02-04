import torch
import cv2
from gtts import gTTS
import pygame
import os
import uuid
import time
from threading import Thread

# Load YOLOv5 model (COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # Use nano version for speed

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Initialize pygame mixer for audio feedback
pygame.mixer.init()

# Precaution messages for objects
precaution_messages = {
    "car": "A car is in front of you. Please stop or take precautions.",
    "tv":"A tv is infront of you,please take care with your path.",
    "person": "A person is nearby. Please be cautious.",
    "dog": "A dog is in your vicinity. Stay alert.",
    "bike": "A bike is approaching. Please move carefully.",
    "bus": "A bus is nearby. Please keep your distance.",
    "truck": "A truck is coming. Stay away and take precautions.",
    "chair": "A chair is in your path. Please navigate around it.",
    "table": "A table is nearby. Move carefully.",
    "cat": "A cat is close. Be cautious.",
    "tree": "A tree is ahead. Please avoid hitting it.",
    "motorcycle": "A motorcycle is approaching. Take care.",
    "traffic light": "A traffic light is ahead. Follow the signals.",
    "stop sign": "A stop sign is nearby. Be cautious.",
    "laptop": "A laptop is in front. Avoid hitting it.",
    "phone": "A phone is nearby. Stay cautious.",
    "bench": "A bench is ahead. Move carefully.",
    "stairs": "Stairs detected. Walk cautiously.",
    "door": "A door is ahead. Open it carefully.",
    "window": "A window is nearby. Watch your step.",
    "bag": "A bag is on your path. Avoid tripping.",
    "book": "A book is nearby. Be careful.",
    "cup": "A cup is in front. Avoid spilling it.",
    "bottle": "A bottle is detected. Avoid knocking it over.",
    "keyboard": "A keyboard is on your path. Avoid damaging it.",
    "mouse": "A mouse is nearby. Be cautious.",
    "monitor": "A monitor is in your way. Avoid hitting it.",
    "ball": "A ball is nearby. Watch your step.",
    "remote": "A remote is ahead. Avoid stepping on it.",
    "mirror": "A mirror is in your vicinity. Avoid breaking it.",
    "umbrella": "An umbrella is detected. Be careful.",
    "plant": "A plant is nearby. Avoid hitting it.",
    "shoe": "A shoe is on your path. Avoid tripping.",
    "bicycle": "A bicycle is coming. Stay alert.",
    "sofa": "A sofa is ahead. Navigate carefully.",
    "television": "A television is nearby. Avoid knocking it over.",
    "toy": "A toy is on your path. Watch your step.",
    "cable": "A cable is on the ground. Avoid tripping.",
    "box": "A box is in your way. Navigate around it.",
    "sign": "A signboard is ahead. Watch your head.",
    "rail": "A rail is nearby. Hold it for support if needed.",
    "road": "You are on a road. Stay alert.",
    "wall": "A wall is in front. Avoid collision.",
    "pole": "A pole is ahead. Move carefully.",  
    "watch":"A watch is on the path. please be carefull.",
    
}
default_precaution = "An object is detected. Please proceed with caution."

# Track the last time a precaution message was played for each object
last_played_time = {}

# Minimum time interval (in seconds) between repeating the same message
MIN_INTERVAL = 5

# Function to provide audio feedback
def audio_feedback(text):
    def play_audio():
        audio_file = f"temp_{uuid.uuid4().hex}.mp3"
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)

            if os.path.exists(audio_file):
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.delay(100)
                os.remove(audio_file)
        except Exception as e:
            print(f"Audio feedback error: {e}")

    Thread(target=play_audio).start()  # Run audio feedback in a separate thread

# Function to manage audio feedback with cooldown
def audio_feedback_with_cooldown(text, obj_name):
    current_time = time.time()
    last_time = last_played_time.get(obj_name, 0)
    if current_time - last_time >= MIN_INTERVAL:
        last_played_time[obj_name] = current_time
        audio_feedback(text)

# Function to initialize the camera
def initialize_camera(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_index}.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Increase frame rate
    return cap

# Main program loop
print("Press 'q' to quit the program.")
cap = initialize_camera(camera_index=0)

if not cap:
    exit()  # Exit if the camera could not be opened

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))

        # Perform object detection
        results = model(small_frame)
        labels = results.names

        detected_objects = set()
        for *xyxy, conf, cls in results.xyxy[0]:
            label = labels[int(cls)]
            if conf >= 0.2:  # Lower confidence threshold
                detected_objects.add(label)
                
                # Determine the bounding box color
                if label in ["person", "dog", "cat"]:
                    color = (0, 255, 0)  # Green
                elif label in ["car", "bus", "truck", "bike", "motorcycle"]:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow

                # Draw bounding box and label
                cv2.rectangle(frame, (int(xyxy[0]*2), int(xyxy[1]*2)), 
                              (int(xyxy[2]*2), int(xyxy[3]*2)), 
                              color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", 
                            (int(xyxy[0]*2), int(xyxy[1]*2) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Handle detected objects
        for obj in detected_objects:
            precaution = precaution_messages.get(obj, default_precaution)
            audio_feedback_with_cooldown(precaution, obj)

        cv2.imshow("YOLOv5 Live Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Resources released successfully.")
