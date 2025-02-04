import os
import pygame
from gtts import gTTS

# Function to provide audio feedback
def audio_feedback(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "temp.mp3"
    tts.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    
    # Quit the mixer to release the audio file properly
    pygame.mixer.quit()

    # Remove the temporary audio file
    try:
        os.remove(audio_file)
        print("Temporary audio file deleted.")
    except PermissionError:
        print("Permission error: The file is still being used.")
