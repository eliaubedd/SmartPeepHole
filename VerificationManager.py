"""
Multimodal Recognition Module

This module implements the core biometric recognition logic of the SmartPeephole
system. It provides:

1) An abstract interface (RecognitionMethod) defining the contract for any
   biometric modality (e.g., face, voice).
2) Concrete implementations for:
      - Face recognition using DeepFace embeddings
      - Speaker recognition using WeSpeaker embeddings + Silero VAD
3) A RecognitionManager that:
      - Stores enrolled user centroids in SQLite
      - Orchestrates multimodal verification
      - Fuses face and voice scores into a final decision

The module is fully modular: additional biometric methods can be integrated
by implementing the RecognitionMethod interface.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import IO, Sequence, Any, Union, Tuple, Mapping

import sqlite3
import numpy as np
import torch
from deepface import DeepFace
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import wespeakerruntime as wespeaker



# Abstract Interface
class RecognitionMethod(ABC):
    """
    Abstract base class for a biometric recognition method.

    Each modality must implement:
        - verify(): compare input data against stored centroids
        - enroll_user(): compute centroid embedding from enrollment samples
    """

    @abstractmethod
    def verify(
        self,
        data: Union[str, NDArray[Any], IO[bytes], Sequence[Union[str, NDArray[Any], IO[bytes]]]],
        centroids: Sequence[Tuple[int, NDArray[Any]]]
    ) -> Sequence[Tuple[int, float]]:
        """
        Compare input sample against enrolled centroids.

        Args:
            data: Input sample (path, ndarray, in-memory stream, etc.).
            centroids: List of (user_id, embedding_blob) pairs.

        Returns:
            Sequence of (user_id, similarity_score) sorted by descending score.
        """
        pass
    
    @abstractmethod
    def enroll_user(
        self,
        examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]
    ) -> NDArray[Any]:
        """
        Compute centroid embedding from enrollment examples.

        Args:
            examples: Sequence of enrollment samples.

        Returns:
            NDArray[Any]: Mean embedding (centroid) representing the user.
        """
        pass



# Face Recognition
class FaceRecognition(RecognitionMethod):
    """
    Face recognition implementation using DeepFace embeddings.

    Embeddings are generated via DeepFace.represent() with detection skipped
    (face alignment is handled upstream in the pipeline).
    """

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name (str, optional): DeepFace backbone model name
                                        (e.g., 'Facenet512').
        """
        self.model_name = model_name
    
    def verify(
        self,
        data: Union[str, NDArray[Any], IO[bytes]],
        centroids: Sequence[Tuple[int, NDArray[Any]]]
    ) -> Sequence[Tuple[int, float]]:
        """
        Verify a face sample against enrolled centroids.

        Args:
            data: Face image (path, ndarray, or stream).
            centroids: Sequence of (user_id, embedding_blob).

        Returns:
            Top-5 (user_id, similarity_score) pairs sorted descending.
            Similarity is cosine similarity normalized to [0, 1].
        """
        target_embedding = DeepFace.represent(
            img_path=data,
            model_name=self.model_name,
            detector_backend='skip'
        )[0]["embedding"]

        user_scores = []

        for user_id, embedding_blob in centroids:
            enrolled_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = ((cosine_similarity([target_embedding], [enrolled_embedding]) + 1) / 2)[0][0].item()
            user_scores.append((user_id, similarity))

        return sorted(user_scores, key=lambda x: x[1], reverse=True)[:5]
    
    def enroll_user(
        self,
        examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]
    ) -> NDArray[Any]:
        """
        Enroll a user by computing the centroid of multiple face embeddings.

        Args:
            examples: Sequence of aligned face images.

        Returns:
            NDArray[Any]: Mean embedding vector (centroid).
        """
        embedding = DeepFace.represent(
            img_path=examples,
            model_name=self.model_name,
            detector_backend='skip'
        )

        embeddings = [entry[0]["embedding"] for entry in embedding]
        centroid = np.mean(embeddings, axis=0)

        return centroid



# Voice Recognition
class VoiceRecognition(RecognitionMethod):
    """
    Speaker recognition implementation using WeSpeaker embeddings.

    Includes:
        - Silero VAD for speech activity filtering
        - Cosine similarity scoring
    """

    def __init__(self, lang: str = "en"):
        """
        Initialize speaker model and VAD.

        Args:
            lang (str, optional): Language configuration for WeSpeaker.
                                  Defaults to "en".
        """
        self.lang = lang
        self.spk = wespeaker.Speaker(
            lang=lang,
            inter_op_num_threads=4,
            intra_op_num_threads=4
        )

        self.vad, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )

        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
    
    def verify(
        self,
        data: Union[str, NDArray[Any], IO[bytes]],
        centroids: Sequence[Tuple[int, NDArray[Any]]]
    ) -> Sequence[Tuple[int, float]]:
        """
        Verify a voice sample against enrolled speaker centroids.

        Args:
            data: Audio sample (path, ndarray, or in-memory stream).
            centroids: Sequence of (user_id, embedding_blob).

        Returns:
            Top-5 (user_id, similarity_score) sorted descending.
            Returns empty list if insufficient speech is detected.
        """
        speech_length_ok, total_speech_duration = self._check_audio_length(data)

        if not speech_length_ok:
            return []

        if isinstance(data, BytesIO):
            data.seek(0)

        target_embedding = self.spk.extract_embedding(data)[0, :]
        user_scores = []

        for user_id, embedding_blob in centroids:
            enrolled_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            try:
                similarity = ((cosine_similarity([target_embedding], [enrolled_embedding]) + 1) / 2)[0][0].item()
            except ValueError:
                similarity = 0.0

            user_scores.append((user_id, similarity))

        return sorted(user_scores, key=lambda x: x[1], reverse=True)[:5]
    
    def enroll_user(
        self,
        examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]
    ) -> NDArray[Any]:
        """
        Compute centroid speaker embedding from multiple enrollment samples.

        Args:
            examples: Sequence of audio samples.

        Returns:
            NDArray[Any]: Mean speaker embedding vector.
        """
        embeddings = []

        for example in examples:
            speech_length_ok, _ = self._check_audio_length(example)

            if isinstance(example, BytesIO):
                example.seek(0)

            if not speech_length_ok:
                continue

            embedding = self.spk.extract_embedding(example)[0, :]
            embeddings.append(embedding)

        centroid = np.mean(embeddings, axis=0)
        return centroid
    
    def _check_audio_length(
        self,
        audio_data: Union[str, NDArray[Any], IO[bytes]],
        threshold: float = 2
    ) -> Tuple[bool, float]:
        """
        Validate that audio contains sufficient speech.

        Args:
            audio_data: Audio input.
            threshold (float, optional): Minimum speech duration in seconds.
                                         Defaults to 2.

        Returns:
            Tuple:
                - bool: True if speech duration >= threshold
                - float: Total detected speech duration in seconds
        """
        if isinstance(audio_data, str) or isinstance(audio_data, BytesIO):
            audio_data = self.read_audio(audio_data, sampling_rate=16000)

        speech_timestamps = self.get_speech_timestamps(
            torch.tensor(audio_data),
            self.vad,
            return_seconds=True
        )

        total_speech_duration = sum(
            subset['end'] - subset['start']
            for subset in speech_timestamps
        )

        return total_speech_duration >= threshold, total_speech_duration



# Recognition Manager (Multimodal Fusion)
class RecognitionManager:
    """
    Orchestrates multimodal enrollment and verification.

    Responsibilities:
        - Store user centroids in SQLite
        - Execute verification for each modality in parallel
        - Fuse modality scores into a final decision
    """

    def __init__(
        self,
        methods: Mapping[str, RecognitionMethod],
        enrolled_users_db_name: str = 'enrolled_users.db'
    ):
        """
        Args:
            methods: Mapping of method_name -> RecognitionMethod instance.
            enrolled_users_db_name (str, optional): SQLite database filename.
        """
        self.methods = methods

        self.enrolled_users_db = sqlite3.connect(
            enrolled_users_db_name,
            check_same_thread=False
        )

        self.cursor = self.enrolled_users_db.cursor()

        is_empty = self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users';"
        ).fetchone() is None

        if is_empty:
            self.cursor.execute(
                f'''CREATE TABLE users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL UNIQUE,
                     {' BLOB NOT NULL,'.join(self.methods.keys())} BLOB NOT NULL);'''
            )
            self.enrolled_users_db.commit()
    
    def enroll_user(
        self,
        name: str,
        examples: Mapping[str, Sequence[Union[str, NDArray[Any], IO[bytes]]]]
    ) -> None:
        """
        Enroll a new user across all modalities.

        Args:
            name (str): User identifier.
            examples: Mapping of method_name -> list of samples.
        """
        def run_method(method_name: str):
            method = self.methods[method_name]
            centroid = method.enroll_user(examples[method_name])
            return method_name, centroid.astype(np.float32).tobytes()

        centroid_embeddings = {}

        with ThreadPoolExecutor(max_workers=min(2, len(self.methods))) as ex:
            for method_name, blob in ex.map(run_method, self.methods.keys()):
                centroid_embeddings[method_name] = blob

        columns = ", ".join(["name"] + list(centroid_embeddings.keys()))
        placeholders = ", ".join(["?"] * (1 + len(centroid_embeddings)))
        values = [name] + [centroid_embeddings[k] for k in centroid_embeddings.keys()]

        self.cursor.execute(
            f"INSERT INTO users ({columns}) VALUES ({placeholders});",
            values
        )

        self.enrolled_users_db.commit()
    
    def verify(
        self,
        data: Mapping[str, Union[str, NDArray[Any], IO[bytes]]]
    ):
        """
        Perform multimodal verification.

        Args:
            data: Mapping method_name -> input sample.

        Returns:
            Final fused decision dictionary from _combine_scores().
        """
        id_to_name = dict(
            self.cursor.execute("SELECT id, name FROM users;").fetchall()
        )

        centroids_by_method = {}

        for method_name in self.methods.keys():
            rows = self.cursor.execute(
                f"SELECT id, {method_name} FROM users;"
            ).fetchall()
            centroids_by_method[method_name] = rows

        def run_method(method_name: str):
            method = self.methods[method_name]
            centroids = centroids_by_method[method_name]
            best_users_list = method.verify(data[method_name], centroids)

            return method_name, [
                (uid, id_to_name.get(uid, "<unknown>"), score)
                for uid, score in best_users_list
            ]

        best_users = {}

        with ThreadPoolExecutor(max_workers=min(len(self.methods), 4)) as ex:
            futures = [ex.submit(run_method, name) for name in self.methods.keys()]
            for fut in as_completed(futures):
                method_name, result = fut.result()
                best_users[method_name] = result

        return self._combine_scores(best_users)
    
    def _combine_scores(self, best_users):
        """
        Fuse face and voice scores into a final decision.

        Decision logic:
            - Reject if insufficient speech
            - Reject if voice score below threshold
            - Reject if best voice user not present in face candidates
            - Otherwise interpolate voice + face confidence

        Returns:
            dict containing:
                - result: 'accept' or 'reject'
                - voice_match
                - face_match
                - combined_score
                OR rejection reason
        """
        voice_threshold = 0.7
        face_threshold = 0.84
        alpha_voice = 0.3
        interpolation_weight = 0.85

        if len(best_users['voice_verification']) == 0:
            return {'result': 'reject', 'reason': 'not enough speech detected'}

        if best_users['voice_verification'][0][2] < voice_threshold:
            return {'result': 'reject', 'reason': 'voice verification below threshold'}
        
        best_voice_user_id, best_voice_user_name, best_voice_score = best_users['voice_verification'][0]

        delta_best_voice = best_voice_score - best_users['voice_verification'][1][2]
        voice_confidence = min(1.0, best_voice_score + alpha_voice * delta_best_voice)

        if not best_voice_user_id in list(map(lambda entry: entry[0], best_users['face_verification'])):
            return {'result': 'reject', 'reason': 'best voice match not in face verification results'}
        
        face_user_id, face_user_name, face_score = next(
            filter(lambda entry: entry[0] == best_voice_user_id, best_users['face_verification'])
        )

        clipped_face_confidence = np.clip(face_score / face_threshold, 0, 1)

        combined = voice_confidence * interpolation_weight + clipped_face_confidence * (1 - interpolation_weight)

        return {
            'result': 'accept' if combined >= 0.5 else 'reject',
            'voice_match': (best_voice_user_id, best_voice_user_name, best_voice_score),
            'face_match': (face_user_id, face_user_name, face_score),
            'combined_score': combined.item()
        }

    def get_user_list(self) -> Sequence[Tuple[int, str]]:
        """
        Retrieve list of enrolled users.

        Returns:
            Sequence of (user_id, name).
        """
        return self.cursor.execute("SELECT id, name FROM users;").fetchall()
    
    def delete_user(self, name: str) -> None:
        """
        Remove an enrolled user from the database.

        Args:
            name (str): User identifier.
        """
        self.cursor.execute("DELETE FROM users WHERE name = ?;", (name,))
        self.enrolled_users_db.commit()
