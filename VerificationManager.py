from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from typing import IO, Sequence, Any, Union, Tuple, Mapping
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from io import BytesIO
import numpy as np
import wespeakerruntime as wespeaker

class VerificationMethod(ABC):
    @abstractmethod
    def verify(self, data: Union[str, NDArray[Any], IO[bytes], Sequence[Union[str, NDArray[Any], IO[bytes]]]], centroids: Sequence[Tuple[int, NDArray[Any]]]) -> Sequence[Tuple[int, float]]:
        pass
    
    @abstractmethod
    def enroll_user(self, examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]) -> NDArray[Any]:
        pass

class FaceVerification(VerificationMethod):
    def __init__(self, model_name: str = None):
        self.model_name = model_name
    
    def verify(self, data: Union[str, NDArray[Any], IO[bytes]], centroids: Sequence[Tuple[int, NDArray[Any]]]) -> Sequence[Tuple[int, float]]:
        target_embedding = DeepFace.represent(img_path=data, model_name=self.model_name, detector_backend='skip')[0]["embedding"]
        user_scores = []
        for user_id, embedding_blob in centroids:
            enrolled_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = ((cosine_similarity([target_embedding], [enrolled_embedding]) + 1) / 2)[0][0].item()
            user_scores.append((user_id, similarity))
        return sorted(user_scores, key=lambda x: x[1], reverse=True)[:5]
    
    def enroll_user(self, examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]) -> NDArray[Any]:
        embedding = DeepFace.represent(img_path=examples, model_name=self.model_name, detector_backend='skip')
        embeddings = [entry[0]["embedding"] for entry in embedding]
        centroid = np.mean(embeddings, axis=0)
        return centroid
    
class VoiceVerification(VerificationMethod):
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.spk = wespeaker.Speaker(lang=lang, inter_op_num_threads=4, intra_op_num_threads=4)
        self.vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
    
    def verify(self, data: Union[str, NDArray[Any], IO[bytes]], centroids: Sequence[Tuple[int, NDArray[Any]]]) -> Sequence[Tuple[int, float]]:
        speech_length_ok, total_speech_duration = self._check_audio_length(data)
        print(f"Total speech duration: {total_speech_duration:.2f} seconds")
        if not speech_length_ok:
            return []
        if isinstance(data, BytesIO):
            data.seek(0)

        target_embedding = self.spk.extract_embedding(data)[0, :]
        user_scores = []
        for user_id, embedding_blob in centroids:
            enrolled_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = ((cosine_similarity([target_embedding], [enrolled_embedding]) + 1) / 2)[0][0].item()
            user_scores.append((user_id, similarity))
        return sorted(user_scores, key=lambda x: x[1], reverse=True)[:5]
    
    def enroll_user(self, examples: Sequence[Union[str, NDArray[Any], IO[bytes]]]) -> NDArray[Any]:
        embeddings = []
        for example in examples:
            speech_length_ok, total_speech_duration = self._check_audio_length(example)
            if isinstance(example, BytesIO):
                example.seek(0)
            if not speech_length_ok:
                continue
            embedding = self.spk.extract_embedding(example)[0, :]
            embeddings.append(embedding)
        centroid = np.mean(embeddings, axis=0)
        return centroid
    
    def _check_audio_length(self, audio_data: Union[str, NDArray[Any], IO[bytes]], threshold: float = 2) -> Tuple[bool, float]:
        if isinstance(audio_data, str) or isinstance(audio_data, BytesIO):
            audio_data = self.read_audio(audio_data, sampling_rate=16000)
        speech_timestamps = self.get_speech_timestamps(torch.tensor(audio_data), self.vad, return_seconds=True)
        total_speech_duration = sum(subset['end'] - subset['start'] for subset in speech_timestamps)
        return total_speech_duration >= threshold, total_speech_duration

class VerificationManager:
    def __init__(self, methods: Mapping[str, VerificationMethod], enrolled_users_db_name: str = 'enrolled_users.db'):
        self.methods = methods
        self.enrolled_users_db = sqlite3.connect(enrolled_users_db_name, check_same_thread=False)
        self.cursor = self.enrolled_users_db.cursor()
        is_empty = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';").fetchone() is None
        if is_empty:
            self.cursor.execute(f'''CREATE TABLE users
                                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    name TEXT NOT NULL UNIQUE,
                                    {' BLOB NOT NULL,'.join(self.methods.keys())} BLOB NOT NULL);''')
            self.enrolled_users_db.commit()
    
    def enroll_user(self, name: str, examples: Mapping[str, Sequence[Union[str, NDArray[Any], IO[bytes]]]]) -> None:
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

        self.cursor.execute(f"INSERT INTO users ({columns}) VALUES ({placeholders});", values)
        self.enrolled_users_db.commit()
    
    def verify(self, data: Mapping[str, Union[str, NDArray[Any], IO[bytes], Sequence[Union[str, NDArray[Any], IO[bytes]]]]]) -> Mapping[str, Tuple[int, str, float]]:
        id_to_name = dict(self.cursor.execute("SELECT id, name FROM users;").fetchall())

        centroids_by_method = {}
        for method_name in self.methods.keys():
            rows = self.cursor.execute(f"SELECT id, {method_name} FROM users;").fetchall()
            centroids_by_method[method_name] = rows

        def run_method(method_name: str):
            method = self.methods[method_name]
            centroids = centroids_by_method[method_name]
            best_users_list = method.verify(data[method_name], centroids)

            return method_name, [(uid, id_to_name.get(uid, "<unknown>"), score) for uid, score in best_users_list]

        best_users = {}
        max_workers = min(len(self.methods), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_method, name) for name in self.methods.keys()]
            for fut in as_completed(futures):
                method_name, result = fut.result()
                best_users[method_name] = result

        print("Best users by method:", best_users)
        return self._combine_scores(best_users)
    
    def _combine_scores(self, best_users: Mapping[str, Sequence[Tuple[int, str, float]]]) -> Union[Mapping[str, Tuple[int, str, float]], Mapping[str, str]]:
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
        
        face_user_id, face_user_name, face_score = next(filter(lambda entry: entry[0] == best_voice_user_id, best_users['face_verification']))

        clipped_face_confidence = np.clip(face_score / face_threshold, 0, 1)

        return {'result': 'accept' if voice_confidence * interpolation_weight + clipped_face_confidence * (1 - interpolation_weight) >= 0.5 else 'reject',
                'voice_match': (best_voice_user_id, best_voice_user_name, best_voice_score),
                'face_match': (face_user_id, face_user_name, face_score),
                'combined_score': (voice_confidence * interpolation_weight + clipped_face_confidence * (1 - interpolation_weight)).item()}