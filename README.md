# SmartPeepHole

**A Raspberry Pi-based smart doorbell with multimodal face and speaker recognition.**

**Authors:** [Elia Leonardo Martin](https://github.com/eliaubedd) · [Alessandro Dubini](https://github.com/Alexd1303)

SmartPeepHole is a collaborative university project exploring the integration of deep learning, computer vision, speaker recognition, and IoT technologies to develop an intelligent doorbell system.

The system captures a visitor's face and voice, compares their biometric features against previously enrolled identities, and sends notifications through Telegram.

The project combines face and speaker recognition models with real-time audio/video acquisition, a FastAPI backend, SQLite databases, and Raspberry Pi hardware.

> **Project Status:** This repository contains a completed university coursework project. Development concluded following the final examination, and the original implementation has been preserved for academic documentation and portfolio purposes. The system is an experimental prototype, not a production-ready security solution.

---

## Key Features

- **Face Detection and Recognition:** Face detection and alignment using MediaPipe, followed by feature extraction through FaceNet512.
- **Speaker Recognition:** Voice embedding extraction using WeSpeaker, supported by Silero Voice Activity Detection (VAD).
- **Multimodal Biometric Verification:** Custom decision logic combining facial and vocal similarities to identify visitors.
- **User Enrollment:** Registration of identities through representative facial and vocal embeddings.
- **Smart Doorbell Integration:** Raspberry Pi GPIO integration for a physical doorbell button and LED.
- **Telegram Notifications:** Automated visitor notifications, including captured images.
- **REST API:** FastAPI endpoints for enrollment, verification, user management, and event history.
- **Local Data Storage:** SQLite databases for enrolled identities, verification history, and device pairings.

## System Architecture

SmartPeepHole is organized around three main components: biometric feature extraction, multimodal identity verification, and hardware/software integration.

### 1. Biometric Feature Extraction

The system acquires facial images and audio recordings from the connected camera and microphone.

**Face Recognition Pipeline**

1. Detect faces in the acquired frames using MediaPipe.
2. Extract facial landmarks and perform face alignment.
3. Generate facial embeddings using the FaceNet512 model through DeepFace.
4. Compare embeddings against enrolled identities using cosine similarity.

**Speaker Recognition Pipeline**

1. Capture the visitor's audio recording.
2. Detect speech activity through Silero VAD.
3. Extract speaker embeddings using WeSpeaker.
4. Compare vocal features against stored speaker representations.

### 2. Multimodal Identity Verification

The verification pipeline combines information from both biometric modalities.

During enrollment, facial and vocal embeddings are extracted from multiple samples and aggregated into representative embeddings for each identity.

During verification:

1. Face and voice embeddings are independently compared against enrolled identities.
2. The system identifies the most relevant candidates from both modalities.
3. A custom verification strategy evaluates the consistency between the facial and vocal predictions.
4. Matching scores are combined to determine the final verification result.

This approach explores whether combining facial and vocal information can provide more reliable identity verification than relying on a single biometric modality.

The final combined score is a custom decision score rather than a calibrated probability of identity.

### 3. Hardware and Notification System

The Raspberry Pi acts as the interface between the physical doorbell and the recognition software.

When the doorbell button is pressed:

1. The system triggers the acquisition of audio and video.
2. The biometric verification pipeline processes the captured samples.
3. The verification result is recorded in a local SQLite database.
4. A notification containing the visitor's image is sent through Telegram.

The backend is implemented using FastAPI, allowing the different components to communicate through HTTP requests.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Hardware | Raspberry Pi |
| Face Detection | MediaPipe |
| Face Recognition | DeepFace / FaceNet512 |
| Speaker Recognition | WeSpeaker |
| Voice Activity Detection | Silero VAD |
| Computer Vision | OpenCV |
| Audio Processing | PyAudio |
| Numerical Computing | NumPy |
| Backend | FastAPI / Uvicorn |
| Database | SQLite |
| Notifications | Telegram Bot API |
| Dependency Management | uv |

## Experimental Evaluation

The development process included experiments to evaluate the facial, vocal, and combined biometric recognition pipelines.

The experiments are documented in the Jupyter notebooks included in this repository.

### Datasets

Two publicly available datasets were used for biometric recognition experiments.

**Labeled Faces in the Wild (LFW)**

The [LFW dataset](https://vis-www.cs.umass.edu/lfw/) was used for facial recognition experiments, specifically its deep-funneled image version.

LFW contains facial photographs of public figures collected from the web, providing images captured under varying conditions, including differences in lighting, pose, and appearance.

**LibriSpeech**

The [LibriSpeech dataset](https://www.openslr.org/12) was used for speaker recognition experiments, particularly the `train-clean-100` subset.

LibriSpeech is a corpus of read English speech derived from audiobooks, containing recordings from multiple speakers.

These datasets provided the facial images and audio samples used in the experimental enrollment and verification procedures.

Since LFW and LibriSpeech contain independent identities, the multimodal experiments use artificially associated facial and vocal identities rather than synchronized audiovisual recordings of the same individuals.

### Evaluation Notebooks

The repository contains three notebooks documenting different stages of development and experimentation:

| Notebook | Description |
|---|---|
| `Face.ipynb` | Face detection, embedding extraction, and recognition experiments |
| `Voice.ipynb` | Speaker embedding extraction and voice recognition experiments |
| `ModelEvaluation.ipynb` | Evaluation of the biometric recognition and verification pipeline |

The repository also includes `model_eval_df.csv` and `combined_eval.db`, which were used to store experimental evaluation data.

These experiments were conducted as part of the original university project. The repository preserves the corresponding development artifacts.

## Repository Structure

The original directory structure has been preserved to maintain compatibility with the implementation developed during the project.

```text
SmartPeepHole/
│
├── Doorbell.py
├── Server.py
├── VerificationManager.py
├── FaceDetection.py
├── TelegramBot.py
│
├── Face.ipynb
├── Voice.ipynb
├── ModelEvaluation.ipynb
│
├── enrolled_users.db
├── combined_eval.db
├── history.db
├── pairings.db
├── model_eval_df.csv
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

### Main Components

- **`Server.py`** — Implements the FastAPI backend, audio/video acquisition, verification endpoints, and notification orchestration.
- **`VerificationManager.py`** — Manages identity enrollment, biometric embedding extraction, similarity comparisons, and multimodal verification.
- **`FaceDetection.py`** — Handles facial detection, landmark extraction, cropping, and alignment.
- **`Doorbell.py`** — Interfaces with the Raspberry Pi GPIO components and triggers verification requests.
- **`TelegramBot.py`** — Implements the Telegram bot and device pairing functionality.

## Installation and Execution

The project was developed as a hardware-dependent prototype and requires the corresponding Raspberry Pi peripherals and software environment.

### Requirements

- Raspberry Pi with GPIO support
- Compatible camera and microphone
- Python 3.9
- [uv](https://docs.astral.sh/uv/) package manager
- Telegram bot token

### Clone the Repository

```bash
git clone https://github.com/eliaubedd/SmartPeepHole.git
cd SmartPeepHole
```

### Install Dependencies

The project includes a `pyproject.toml` file and a `uv.lock` file for dependency management.

```bash
uv sync
```

Some audio/video processing and machine learning dependencies may require additional native libraries, depending on the operating system and hardware configuration.

### Configuration

The implementation reads the following environment variables from a local `.env` file:

```dotenv
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
DEVICE_ID=your_device_identifier
```

These values must be configured before starting the corresponding services.

### Run the Application

Start the FastAPI backend:

```bash
uv run uvicorn Server:app --host 0.0.0.0 --port 8080
```

Start the Telegram bot in a separate terminal:

```bash
uv run python TelegramBot.py
```

Start the Raspberry Pi doorbell handler:

```bash
uv run python Doorbell.py
```

The GPIO configuration uses BCM pin 26 for the physical button and BCM pin 16 for the LED.

**Note:** These commands describe the original execution approach. The application has not been revalidated on newer hardware, operating systems, or dependency versions. It should only be run on a trusted local network.

## API Overview

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Check server availability |
| `POST` | `/enroll` | Register a new identity |
| `POST` | `/verify` | Perform biometric visitor verification |
| `POST` | `/history` | Retrieve recent verification events |
| `POST` | `/users` | List enrolled identities |
| `DELETE` | `/users/{name}` | Remove an enrolled identity |

The original implementation uses a shared device identifier for basic request authorization. It does not implement a production-grade authentication mechanism.

## Limitations

SmartPeepHole was developed as an academic prototype to investigate the integration of biometric recognition and IoT technologies.

Several limitations should be considered:

- Biometric verification performance depends on the quality of the acquired facial images and voice recordings.
- Face and voice recognition were evaluated using separate datasets, with artificially paired identities.
- The recognition pipeline does not include dedicated biometric liveness detection or anti-spoofing mechanisms.
- The application depends on specific hardware, device interfaces, and software versions.
- The authentication and Telegram pairing mechanisms were designed for experimental use rather than production deployment.
- The repository preserves the original implementation and has not been actively maintained following the completion of the university project.

## Privacy and Responsible Use

The experimental biometric data originate from publicly available datasets rather than a private collection of volunteer participants.

Nevertheless, facial and vocal embeddings can constitute biometric personal data, even when derived from public datasets. Their use and redistribution remain subject to applicable licensing conditions and privacy requirements.

The prototype should not be deployed as a real-world security system without additional validation, appropriate consent procedures, and stronger data protection mechanisms.

## Authors

SmartPeepHole was jointly developed by:

- **[Elia Leonardo Martin](https://github.com/eliaubedd)** — [@eliaubedd](https://github.com/eliaubedd)
- **[Alessandro Dubini](https://github.com/Alexd1303)** — [@Alexd1303](https://github.com/Alexd1303)

Both authors contributed to the design, implementation, and experimental evaluation of the system.

## Project Status

**Completed — University Coursework Project**

SmartPeepHole was developed and evaluated as part of university coursework.

The repository is preserved as an archive of the original implementation, documenting the exploration of multimodal biometric recognition, deep learning, backend development, and Raspberry Pi integration.

No active development or production support is currently planned.
