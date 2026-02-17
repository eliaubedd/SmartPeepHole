import queue
import threading
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, Header
from typing import Annotated, Optional
from pydantic import BaseModel
import sqlite3
import requests

from VerificationManager import RecognitionManager, FaceRecognition, VoiceRecognition
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
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEVICE_CODE = os.getenv("DEVICE_ID")

if not BOT_TOKEN or not DEVICE_CODE:
    raise ValueError("Missing required environment variables: TELEGRAM_BOT_TOKEN or DEVICE_ID")

app = FastAPI()

class EnrollReq(BaseModel):
    name: str

class OwnerRegReq(BaseModel):
    code: str

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
        fv_model = FaceRecognition(model_name='Facenet512')
        self.verification_manager = RecognitionManager(methods={'face_verification': fv_model, 'voice_verification': VoiceRecognition(lang='en')})
        self.conn = sqlite3.connect("pairings.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS pairings (
            user_id INTEGER PRIMARY KEY,
            raspberry_id TEXT NOT NULL
        )""")
        #self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.conn.commit()

        self.history_conn = sqlite3.connect("history.db", check_same_thread=False)
        self.history_cursor = self.history_conn.cursor()
        self.history_cursor.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            verification_result TEXT NOT NULL,
            match TEXT,
            combined_score REAL
        )""")
        self.history_conn.commit()
        self.history_cursor = self.history_conn.cursor()

    def record_data(self, duration_s: int = 5, max_faces: int = 5) -> tuple[BytesIO, list[np.ndarray]]:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000 ## MUST BE KEPT AT 16000 FOR VAD TO WORK PROPERLY ON RASPBERRY PI, DO NOT CHANGE TO 44100 ##
        RECORD_SECONDS = duration_s

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []
        face_selector = FaceSelector(max_faces=max_faces)

        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        print("cap opened:", cap.isOpened())
        if not cap.isOpened():
            raise RuntimeError("Failed to open /dev/video0")

        cap.set(cv2.CAP_PROP_FPS, 30)

        print("backend:", cap.getBackendName())
        print("fourcc:", int(cap.get(cv2.CAP_PROP_FOURCC)))
        print("size:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("fps:", cap.get(cv2.CAP_PROP_FPS))
        
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
        audio, video = self.record_data(duration_s=5, max_faces=10)
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

        self.send_notification(video[0][1], result)

        self.history_cursor.execute("INSERT INTO history (verification_result, match, combined_score) VALUES (?, ?, ?)",
                                    (result.get("result"), result.get("voice_match")[1] if result.get("voice_match") else None, result.get("combined_score")))
        self.history_conn.commit()

        return result

    def send_notification(self, face_img: np.ndarray, verification_result: dict):
        self.cursor.execute("SELECT user_id FROM pairings WHERE raspberry_id = ?", (DEVICE_CODE,))
        rows = self.cursor.fetchall()
        if not rows:
            print("No paired Telegram user found for this device.")
            return

        for row in rows:
            user_id = row[0]

            if verification_result.get("result") == "accept":
                caption = f"""🔔 Visitor Alert
    A visitor has been detected at your door.
    It appears to be {verification_result['voice_match'][1]} (confidence: {verification_result['combined_score']:.2f}).
    See attached image."""
            else:
                caption = f"""🔔 Visitor Alert
    A visitor has been detected at your door.
    We can't identify this person, it appears to be none of your enrolled users.
    See attached image."""

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            _, img_encoded = cv2.imencode(".jpg", face_img)
            files = {"photo": ("image.jpg", img_encoded.tobytes())}
            data = {"chat_id": user_id, "caption": caption}
            try:
                r = requests.post(url, data=data, files=files, timeout=20)
                r.raise_for_status()
            except Exception as e:
                print(f"Failed to send notification to user {user_id}: {e}")

engine = Engine()
busy_lock = threading.Lock()

@app.get("/status")
def status():
    return {"ring_active": True}

#add user token to authenticate request
@app.post("/enroll")
def enroll(req: EnrollReq, authorization: Annotated[Optional[str], Header()] = None):
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not busy_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy processing another request. Please try again later.")
    try:
        result = engine.enroll(req.name)
    finally:
        busy_lock.release()
    return result
    

@app.post("/verify")
def verify(authorization: Annotated[Optional[str], Header()] = None):
    print("Received verification request with token:", authorization)
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not busy_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy processing another request. Please try again later.")
    try:
        result = engine.verify()
    finally:
        busy_lock.release()
    return result

@app.post("/history")
def history(authorization: Annotated[Optional[str], Header()] = None):
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    engine.history_cursor.execute("SELECT timestamp, verification_result, match, combined_score FROM history ORDER BY timestamp DESC LIMIT 20")
    rows = engine.history_cursor.fetchall()
    history = []
    for row in rows:
        history.append({
            "timestamp": row[0],
            "verification_result": row[1],
            "match": row[2],
            "combined_score": row[3]
        })
    print("Returning history:", history)
    return {"history": history}

@app.post("/users")
def users(authorization: Annotated[Optional[str], Header()] = None):
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    rows = engine.verification_manager.get_user_list()
    users = [row[1] for row in rows]
    return {"users": users}

@app.delete("/users/{name}")
def delete_user(name: str, authorization: Annotated[Optional[str], Header()] = None):
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    engine.verification_manager.delete_user(name)
    return {"ok": True}
