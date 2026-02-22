import queue
import threading
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Header
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
    """
    Request body model for user enrollment.

    Attributes:
        name (str): Human-readable identifier for the user to be enrolled.
                    This name will be used as the key when storing embeddings
                    and when reporting matches during verification.
    """
    name: str

class FaceSelector:
    """
    Utility class to keep track of the best face crops observed during recording.

    This class maintains a sorted list of (score, face_img) tuples and retains
    only the top `max_faces` entries by score. It is used to:
        - Collect multiple high-quality face samples for enrollment
        - Collect a single best face sample for verification

    Attributes:
        best_faces (SortedList): Sorted container of (score, face_img) tuples.
        max_faces (int): Maximum number of best faces to retain.
    """
    def __init__(self, max_faces: int = 5):
        """
        Initialize the selector.

        Args:
            max_faces (int, optional): Maximum number of best-scoring faces to store.
                                       Defaults to 5.
        """
        self.best_faces = SortedList()
        self.max_faces = max_faces

    def add_face(self, face_img: np.ndarray, score: float):
        """
        Add a detected face crop along with its detection confidence score.

        The face is inserted in a sorted container. If the number of stored
        faces exceeds `max_faces`, the lowest-scoring face is removed.

        Args:
            face_img (np.ndarray): Cropped, aligned face image.
            score (float): Detection confidence score produced by the face detector.

        Returns:
            None
        """
        self.best_faces.add((score, face_img))
        if len(self.best_faces) > self.max_faces:
            self.best_faces.pop(0)

    def get_best_faces(self) -> np.ndarray:
        """
        Retrieve the stored faces ordered by increasing score (as kept by SortedList).

        Returns:
            np.ndarray | None:
                - None if no faces were collected
                - Otherwise, a list of tuples (score, face_img)

        Notes:
            The return type annotation in the original code states `np.ndarray`,
            but the actual return value is either `None` or a Python list of tuples.
            The code is left unmodified as requested.
        """
        if not self.best_faces:
            return None
        return [(score, face) for score, face in self.best_faces]
        
class Engine:
    """
    Core application engine.

    Responsibilities:
        - Initialize and hold heavy ML models for face and voice recognition
        - Capture synchronized audio/video samples for a fixed duration
        - Perform enrollment by computing and storing centroids (via RecognitionManager)
        - Perform verification against enrolled users
        - Send Telegram notifications with a captured face snapshot
        - Persist verification events into a local history SQLite database

    Notes:
        This class owns:
            - A RecognitionManager instance containing face and voice methods
            - Database connections/cursors for pairings and history
    """
    def __init__(self):
        """
        Initialize ML models, recognition pipeline, and local databases.

        Side Effects:
            - Loads face and voice recognition models (potentially heavy operations)
            - Opens/creates SQLite databases: pairings.db and history.db
            - Creates required tables if missing

        Raises:
            sqlite3.Error: If database initialization fails.
            Exception: If model initialization fails.
        """
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
        """
        Record audio and video for a fixed duration, while selecting best face crops.

        Audio:
            - Captured using PyAudio
            - Mono, 16-bit PCM, 16 kHz sample rate
            - Stored as a valid WAV file in memory (BytesIO)

        Video:
            - Captured using OpenCV from /dev/video0 (V4L2 backend)
            - Faces detected using MediaPipe FaceDetection
            - Face crops are aligned/rotated using detect_bbox_and_eyes + rotate_and_crop
            - Up to `max_faces` best face samples are retained via FaceSelector

        Args:
            duration_s (int, optional): Recording duration in seconds. Defaults to 5.
            max_faces (int, optional): Maximum number of best face samples to return.
                                       Defaults to 5.

        Returns:
            tuple[BytesIO, list[np.ndarray]]:
                - BytesIO: In-memory WAV file containing the recorded audio.
                - list[np.ndarray] | None: List of (score, face_img) tuples from FaceSelector.
                  (See FaceSelector.get_best_faces() notes about actual return type.)

        Raises:
            RuntimeError: If the video capture device cannot be opened.
            Exception: For audio/video capture errors, detector errors, or encoding errors.

        Operational Notes:
            - The recording loop runs until `RECORD_SECONDS` has elapsed.
            - FPS is printed continuously for runtime diagnostics.
            - The sample rate is intentionally fixed at 16 kHz due to the VAD constraint
              noted in the code comment.
        """
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
        """
        Enroll a new user by recording AV data and computing face/voice centroids.

        Workflow:
            1. Record 5 seconds of audio and collect up to 10 best face samples.
            2. Reject enrollment if any retained face sample has score < 0.85.
            3. Pass collected samples to RecognitionManager.enroll_user()
            4. Return a success payload.

        Args:
            name (str): User identifier used to store the enrolled embeddings.

        Returns:
            dict: Enrollment result payload, e.g. {"name": <name>, "ok": True}.

        Raises:
            fastapi.HTTPException:
                - 400 if face detection quality is insufficient
                - 500 if enrollment fails due to internal errors
        """
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
        """
        Verify the identity of a visitor by recording AV data and running matching.

        Workflow:
            1. Record 5 seconds of audio and collect the single best face sample.
            2. Reject verification if no face is detected or score < 0.92.
            3. Run RecognitionManager.verify() using the face crop and audio WAV.
            4. Send a Telegram notification to paired users with the face snapshot.
            5. Save the verification event to the history database.
            6. Return the full verification result from the RecognitionManager.

        Returns:
            dict: Verification result as returned by RecognitionManager.verify().
                  Expected to include keys such as:
                    - "result" (str): e.g., "accept" / "reject"
                    - "voice_match" (tuple | None): match information (if any)
                    - "combined_score" (float): fused multimodal score

        Raises:
            fastapi.HTTPException:
                - 400 if no suitable face is detected
                - 500 if verification fails due to internal errors
        """
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
        """
        Send a Telegram notification with the visitor face snapshot to paired users.

        Pairing lookup:
            - Queries `pairings.db` to retrieve Telegram user IDs associated with
              the current Raspberry device (`DEVICE_CODE`).

        Notification:
            - Sends a photo message through Telegram Bot API (`sendPhoto`)
            - Message content depends on the verification outcome

        Args:
            face_img (np.ndarray): Face image (BGR OpenCV image) to attach to the notification.
            verification_result (dict): Verification output containing at least:
                - "result" (str): accept/reject decision
                - "voice_match" (tuple | None): match details (if available)
                - "combined_score" (float): score used for confidence reporting

        Returns:
            None

        Side Effects:
            - Performs network I/O to Telegram API.
            - Prints errors to stdout on failure.

        Notes:
            - Uses a 20-second timeout to avoid hanging indefinitely.
            - The caption uses triple-quoted strings for readability.
        """
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
    """
    Status endpoint.

    Returns:
        dict: Static status payload currently reporting that the ring trigger is active.
    """
    return {"ring_active": True}

#add user token to authenticate request
@app.post("/enroll")
def enroll(req: EnrollReq, authorization: Annotated[Optional[str], Header()] = None):
    """
    Enroll endpoint.

    Authentication:
        Requires an `Authorization` header equal to DEVICE_CODE.

    Concurrency Control:
        Uses a non-blocking lock to ensure only one job (enroll/verify)
        runs at a time. If the lock cannot be acquired, returns HTTP 429.

    Args:
        req (EnrollReq): Request body containing the user's name.
        authorization (Optional[str]): Authorization token read from the HTTP header.

    Returns:
        dict: Enrollment result returned by Engine.enroll().

    Raises:
        fastapi.HTTPException:
            - 401 Unauthorized if token mismatch
            - 429 Too Many Requests if the engine is busy
            - Propagates exceptions raised by Engine.enroll()
    """
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
    """
    Verification endpoint.

    This endpoint is typically triggered by the doorbell workflow and performs:
        - 5-second AV recording
        - multimodal verification
        - Telegram notification
        - history persistence

    Authentication:
        Requires an `Authorization` header equal to DEVICE_CODE.

    Concurrency Control:
        Uses a non-blocking lock to ensure only one heavy job runs at a time.

    Args:
        authorization (Optional[str]): Authorization token read from the HTTP header.

    Returns:
        dict: Verification result returned by Engine.verify().

    Raises:
        fastapi.HTTPException:
            - 401 Unauthorized if token mismatch
            - 429 Too Many Requests if the engine is busy
            - Propagates exceptions raised by Engine.verify()
    """
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
    """
    Retrieve recent verification history.

    Authentication:
        Requires an `Authorization` header equal to DEVICE_CODE.

    Behavior:
        - Returns up to 20 most recent events ordered by timestamp descending.

    Args:
        authorization (Optional[str]): Authorization token read from the HTTP header.

    Returns:
        dict: {"history": [...]} where each item contains:
            - timestamp (str)
            - verification_result (str)
            - match (str | None)
            - combined_score (float | None)

    Raises:
        fastapi.HTTPException:
            - 401 Unauthorized if token mismatch
    """
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
    """
    List enrolled users.

    Authentication:
        Requires an `Authorization` header equal to DEVICE_CODE.

    Returns:
        dict: {"users": [<user_name>, ...]} extracted from the underlying
              RecognitionManager user list.

    Raises:
        fastapi.HTTPException:
            - 401 Unauthorized if token mismatch
    """
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    rows = engine.verification_manager.get_user_list()
    users = [row[1] for row in rows]
    return {"users": users}

@app.delete("/users/{name}")
def delete_user(name: str, authorization: Annotated[Optional[str], Header()] = None):
    """
    Delete an enrolled user.

    Authentication:
        Requires an `Authorization` header equal to DEVICE_CODE.

    Args:
        name (str): Name/identifier of the user to remove.
        authorization (Optional[str]): Authorization token read from the HTTP header.

    Returns:
        dict: {"ok": True} on successful deletion.

    Raises:
        fastapi.HTTPException:
            - 401 Unauthorized if token mismatch
            - Propagates errors raised by RecognitionManager.delete_user()
    """
    if authorization != DEVICE_CODE:
        raise HTTPException(status_code=401, detail="Unauthorized")
    engine.verification_manager.delete_user(name)
    return {"ok": True}
