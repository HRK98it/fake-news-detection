import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile

def extract_text_from_video(video_path: str) -> str:
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    video = AudioSegment.from_file(video_path)
    video.export(audio_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        text = ""

    os.remove(audio_path)
    return text
