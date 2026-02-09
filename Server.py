# app.py
import queue
import threading
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from VerificationManager import VerificationManager, FaceVerification, VoiceVerification
from FaceDetection import detect_bbox_and_eyes, rotate_and_crop
import cv2
import numpy as np
import time
import torch
from io import BytesIO
import pyaudio
import wave
from mediapipe.python.solutions import face_detection

from sortedcontainers import SortedList

app = FastAPI()

class EnrollReq(BaseModel):
    name: str

class FaceSelector:
    def __init__(self, max_faces: int = 5):
        self.best_faces = SortedList()
        self.max_faces = max_faces

    def add_face(self, face_img: np.ndarray, score: float):
        self.best_faces.add((score, face_img))
        if len(self.best_faces) > self.max_faces:
            self.best_faces.pop(0)

    def get_best_faces(self) -> np.ndarray:
        if not self.best_faces:
            return None
        return [(score, face) for score, face in self.best_faces]
        
class Engine:
    def __init__(self):
        fv_model = FaceVerification(model_name='Facenet512')
        self.verification_manager = VerificationManager(methods={'face_verification': fv_model, 'voice_verification': VoiceVerification(lang='en')})



    def record_data(self, duration_s: int = 5, max_faces: int = 5) -> tuple[BytesIO, list[np.ndarray]]:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = duration_s
        WAVE_OUTPUT_FILENAME = "output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []
        face_selector = FaceSelector(max_faces=max_faces)

        cap = cv2.VideoCapture(1)
        with face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
            start = time.time()
            previous_time = time.time() * 1000
            fps = 0.0
            best_score = 0.0
            total_time = 0.0

            while time.time() - start < RECORD_SECONDS:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                ok, frame = cap.read()
                if not ok:
                    break

                out = detect_bbox_and_eyes(frame, detector)

                if out is not None:
                    bbox, left_eye, right_eye, score = out
                    face112 = rotate_and_crop(frame, bbox, left_eye, right_eye, margin=0.9, out_size=512)
                    if face112 is not None:
                        face_selector.add_face(face112, score)
                        best_score = max(best_score, score)

                current_time = time.time() * 1000
                elapsed_time = current_time - previous_time
                previous_time = current_time
                total_time += elapsed_time

                fps = 1000.0 / elapsed_time if elapsed_time > 0 else 0.0
                print(f"Current FPS: {fps:.2f}, Best Score: {best_score:.4f}, Total Time: {total_time/1000:.2f}s", end='\r')

        cap.release()

        print("* done recording")

        stream.stop_stream()
        stream.close()
        sampwidth = p.get_sample_size(FORMAT)
        p.terminate()

        file_in_memory = BytesIO()

        # Write a proper WAV into RAM (closing finalizes the header)
        with wave.open(file_in_memory, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sampwidth)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        # Rewind again because torchaudio.load consumed the stream
        file_in_memory.seek(0)

        return file_in_memory, face_selector.get_best_faces()

    def enroll(self, name: str) -> dict:
        audio, video = self.record_data(duration_s=5, max_faces=5)
        if any(score < 0.85 for score, _ in video):
            raise HTTPException(status_code=400, detail="Not enough faces detected during enrollment. Please try again.")
        try:
            self.verification_manager.enroll_user(name, examples={'face_verification': [image for _, image in video], 'voice_verification': [audio]})
        except Exception as e:
            print(f"Error during enrollment: {e}")
            raise HTTPException(status_code=500, detail="Enrollment failed due to an internal error.")
        return {"name": name, "ok": True}

    def verify(self) -> dict:
        audio, video = self.record_data(duration_s=5, max_faces=1)
        if not video or video[0][0] < 0.92:
            raise HTTPException(status_code=400, detail="No face detected during verification. Please try again.")
        try:
            result = self.verification_manager.verify(data={'face_verification': video[0][1], 'voice_verification': audio})
        except Exception as e:
            print(f"Error during verification: {e}")
            raise HTTPException(status_code=500, detail="Verification failed due to an internal error.")
        print(f"Verification result: {result}")
        return result

engine = Engine()
busy_lock = threading.Lock()

@app.get("/status")
def status():
    cv2.imwrite("test_face.jpg", engine.record_data(duration_s=5, max_faces=5)[1][0][1])
    return {"status": "ok"}

@app.post("/enroll")
def enroll(req: EnrollReq):
    if not busy_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy processing another request. Please try again later.")
    try:
        result = engine.enroll(req.name)
    finally:
        busy_lock.release()
    return result
    

@app.post("/verify")
def verify():
    if not busy_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy processing another request. Please try again later.")
    try:
        result = engine.verify()
    finally:
        busy_lock.release()
    return result

## 192.168.1.150
