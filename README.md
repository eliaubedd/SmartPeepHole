
# SmartPeepHole

**A Raspberry Pi-based smart doorbell with multimodal face and speaker recognition.**

**Authors:** [Elia Leonardo Martin](https://github.com/eliaubedd) · [Alessandro Dubini](https://github.com/Alexd1303)

SmartPeepHole is a collaborative university project combining **Computer Vision, Speaker Recognition, Multimodal AI, and IoT** to develop an intelligent doorbell system.

The system captures a visitor's face and voice, extracts biometric representations, and compares them against previously enrolled identities. Face and voice predictions are combined through a custom verification strategy, with results accessible through a mobile application and Telegram notifications.

Unlike a purely experimental recognition pipeline, SmartPeepHole was implemented as a **functional hardware prototype**, integrating a Raspberry Pi, camera, microphone, physical doorbell button, backend services, and a Flutter mobile interface.

> **Project Status:** Completed university coursework project. The original implementation is preserved for academic documentation and portfolio purposes. SmartPeepHole is an experimental prototype, not a production-ready biometric security system.

---

## Key Features

- **Face Detection and Recognition:** Face detection and alignment using MediaPipe, followed by embedding extraction through FaceNet512.
- **Speaker Recognition:** Speaker embeddings extracted using WeSpeaker, supported by Silero Voice Activity Detection (VAD).
- **Multimodal Biometric Verification:** Custom decision logic combining facial and vocal similarities to identify visitors.
- **User Enrollment:** Registration of identities through representative facial and vocal embeddings.
- **Hardware Integration:** Raspberry Pi GPIO integration for a physical doorbell button and LED.
- **Mobile Application:** Flutter-based interface for enrollment, device status, user management, and recognition history.
- **Telegram Notifications:** Remote visitor notifications, including captured images.
- **REST API:** FastAPI endpoints for enrollment, verification, user management, and event history.
- **Local Data Storage:** SQLite databases for enrolled identities, verification history, and device pairings.

---

## Prototype and Demonstration

SmartPeepHole was implemented and demonstrated on physical Raspberry Pi hardware.

The prototype integrates:

- Raspberry Pi as the main processing device.
- Camera for facial image acquisition.
- Microphone for voice recording.
- Physical doorbell button and status LED.
- Biometric inference services.
- Flutter mobile application.
- Telegram bot for remote notifications.

The end-to-end demonstration included visitor recognition, mobile interaction, and remote notifications.

The hardware implementation provides a practical example of integrating pretrained AI models into a working edge-computing application.

---

## System Architecture

SmartPeepHole is organized around four main components:

1. Biometric feature extraction.
2. Multimodal identity verification.
3. Hardware and backend integration.
4. Mobile application and remote notifications.

### 1. Biometric Feature Extraction

The system acquires facial images and audio recordings from the connected camera and microphone.

#### Face Recognition Pipeline

1. Detect faces in the acquired frames using MediaPipe.
2. Extract facial landmarks and perform face alignment.
3. Generate facial embeddings using the FaceNet512 model through DeepFace.
4. Compare embeddings against enrolled identities using cosine similarity.
5. Retrieve the most relevant candidate identities.

#### Speaker Recognition Pipeline

1. Capture the visitor's audio recording.
2. Detect speech activity through Silero VAD.
3. Extract speaker embeddings using WeSpeaker.
4. Compare vocal features against stored speaker representations.
5. Retrieve the most relevant candidate identities.

Both pipelines use pretrained representation models, allowing identity comparison through embedding similarity rather than training a separate classification model for each enrolled person.

### 2. Multimodal Identity Verification

The verification pipeline combines information from facial and vocal recognition.

During enrollment, facial and vocal embeddings are extracted from multiple samples and aggregated into representative embeddings for each identity.

During verification:

1. Face and voice embeddings are independently compared against enrolled identities.
2. The system retrieves the highest-ranked candidates from both modalities.
3. A custom decision strategy checks consistency between facial and vocal predictions.
4. Matching scores are combined to determine the final verification result.
5. The system accepts or rejects the candidate identity based on the verification criteria.

This strategy was designed to investigate whether combining facial and vocal information could reduce false acceptances compared with individual biometric modalities.

The resulting combined score is a custom verification score, not a calibrated probability of identity.

### 3. Hardware and Backend Integration

The Raspberry Pi connects the physical doorbell components to the recognition software.

When the doorbell button is pressed:

1. The system triggers audio and video acquisition.
2. Facial and vocal embeddings are extracted.
3. The multimodal verification pipeline evaluates the visitor.
4. The verification result is recorded in a local SQLite database.
5. The event becomes accessible through the application.
6. A Telegram notification containing the visitor's image is sent.

The backend is implemented using **FastAPI**, providing HTTP endpoints for communication between system components.

### 4. Mobile Application and Notifications

A **Flutter mobile application** was developed to provide an interface to the physical system.

Its main functionality includes:

- **User enrollment:** Register identities using audio and video samples.
- **Device status:** Check the availability of the Raspberry Pi.
- **User management:** View and remove enrolled identities.
- **Recognition history:** Inspect previous verification events and their results.
- **Device pairing:** Discover and connect to Raspberry Pi devices on the local network.
- **Notifications:** Receive remote alerts through the Telegram integration.

Local device discovery uses Network Service Discovery (NSD), while the backend provides the corresponding application endpoints.

Telegram integration was adopted for remote notifications, including visitor images and recognition outcomes.

The mobile interface was developed as part of the original university project. Its source code is not included in the Python backend file structure documented below.

---

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
| Mobile Application | Flutter |
| Database | SQLite |
| Device Discovery | Network Service Discovery (NSD) |
| Notifications | Telegram Bot API |
| Dependency Management | uv |

---

## Experimental Evaluation

The project included experiments to evaluate three biometric recognition configurations:

1. Face recognition only.
2. Speaker recognition only.
3. Combined face and speaker recognition.

The objective was to investigate the relationship between correct identification of known users and erroneous acceptance of unknown users.

### Datasets

Two publicly available datasets were used for biometric recognition experiments.

#### Labeled Faces in the Wild (LFW)

The [LFW dataset](https://vis-www.cs.umass.edu/lfw/) was used for facial recognition experiments, specifically its deep-funneled image version.

LFW contains facial photographs of public figures collected from the web under varying conditions, including differences in pose, illumination, and appearance.

#### LibriSpeech

The [LibriSpeech dataset](https://www.openslr.org/12) was used for speaker recognition experiments, particularly the `train-clean-100` subset.

LibriSpeech contains recordings of read English speech from multiple speakers.

**Important experimental limitation:** LFW and LibriSpeech contain independent identities. Multimodal experiments therefore used artificially associated facial and vocal identities rather than synchronized recordings of the same individuals.

The reported results characterize performance under this experimental protocol and should not be interpreted as real-world biometric security validation.

### Evaluation Metrics

Two principal metrics were considered.

**Detection & Identification Rate (DIR)**

The percentage of known users who are both accepted and correctly identified.

This metric evaluates the system's ability to recognize enrolled identities.

**False Accept Rate (FAR)**

The percentage of unknown users erroneously accepted by the verification system.

This metric evaluates the system's tendency to incorrectly accept identities that should be rejected.

### Quantitative Results

| Recognition Pipeline | DIR ↑ | FAR ↓ |
|---|---:|---:|
| Face Recognition | 80.77% | 4.50% |
| Speaker Recognition | 98.08% | 14.81% |
| **Combined Multimodal Recognition** | **96.21%** | **1.90%** |

The multimodal verification strategy achieved a **96.21% Detection & Identification Rate**, while reducing the **False Accept Rate to 1.90%**.

Compared with the individual modalities:

- Face-only recognition achieved a lower FAR than voice-only recognition, but also a lower DIR.
- Voice-only recognition achieved the highest DIR, at the cost of a substantially higher FAR.
- The combined system reduced false acceptances compared with both individual pipelines while maintaining a high identification rate.

These results illustrate the potential benefit of multimodal verification, where complementary biometric signals can improve acceptance decisions.

However, the fusion strategy involves a trade-off between identification performance and false acceptance behavior. The combined system does not outperform every individual modality on every metric.

### Evaluation Notebooks

The repository contains three Jupyter notebooks documenting different stages of development and experimentation.

| Notebook | Description |
|---|---|
| `Face.ipynb` | Face detection, embedding extraction, and recognition experiments |
| `Voice.ipynb` | Speaker embedding extraction and voice recognition experiments |
| `ModelEvaluation.ipynb` | Evaluation of the biometric recognition and verification pipeline |

Additional experimental artifacts include:

- `model_eval_df.csv`
- `combined_eval.db`

These files document the evaluation procedures developed during the university project.

---

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
- **`FaceDetection.py`** — Handles face detection, landmark extraction, cropping, and alignment.
- **`Doorbell.py`** — Interfaces with the Raspberry Pi GPIO components and triggers verification requests.
- **`TelegramBot.py`** — Implements the Telegram bot and device pairing functionality.

The notebooks contain exploratory experiments and evaluation procedures, while the Python modules implement the operational backend and hardware integration.

---

## Installation and Execution

SmartPeepHole was developed as a hardware-dependent prototype and requires the corresponding Raspberry Pi peripherals and software environment.

### Requirements

- Raspberry Pi with GPIO support.
- Compatible camera and microphone.
- Python 3.9.
- [uv](https://docs.astral.sh/uv/) package manager.
- Telegram bot token.

### Clone the Repository

```bash
git clone https://github.com/eliaubedd/SmartPeepHole.git
cd SmartPeepHole
```

### Install Dependencies

The project includes `pyproject.toml` and `uv.lock` for dependency management.

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

> **Note:** These commands describe the original execution approach. The application has not been revalidated on newer hardware, operating systems, or dependency versions. It should only be run on a trusted local network.

---

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

The original implementation uses a shared device identifier for basic request authorization.

It does not implement a production-grade authentication mechanism.

---

## Limitations

SmartPeepHole was developed as an academic prototype to investigate the integration of biometric recognition and IoT technologies.

Several limitations should be considered:

- Biometric verification performance depends on the quality of acquired facial images and voice recordings.
- Face and voice recognition were evaluated using separate datasets with artificially paired identities.
- The reported metrics do not establish performance under real-world visitor conditions.
- The recognition pipeline does not include dedicated biometric liveness detection or anti-spoofing mechanisms.
- The application depends on specific hardware, device interfaces, and software versions.
- The authentication and Telegram pairing mechanisms were designed for experimental use rather than production deployment.
- The repository preserves the original implementation and has not been actively maintained following completion of the university project.

---

## Privacy and Responsible Use

The experimental biometric data originate from publicly available datasets rather than a private collection of volunteer participants.

Nevertheless, facial and vocal embeddings may constitute biometric personal data, even when derived from public datasets.

Their use and redistribution remain subject to applicable licensing conditions and privacy requirements.

The prototype should not be deployed as a real-world security system without additional validation, appropriate consent procedures, and stronger data protection mechanisms.

---

## Authors

SmartPeepHole was jointly developed by:

- **[Elia Leonardo Martin](https://github.com/eliaubedd)** — [@eliaubedd](https://github.com/eliaubedd)
- **[Alessandro Dubini](https://github.com/Alexd1303)** — [@Alexd1303](https://github.com/Alexd1303)

Both authors contributed to the design, implementation, and experimental evaluation of the system.
