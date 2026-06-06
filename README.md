```markdown
# ASL Detection 🤟

Real-time American Sign Language (ASL) detection using MediaPipe hand landmarks and a PyTorch MLP model — **97% accuracy** across 29 classes.

## How it works
1. **MediaPipe** extracts 21 hand landmarks (63 values: x, y, z) from each frame
2. A **PyTorch MLP** classifies the landmarks into one of 29 ASL signs (A-Z, space, del, nothing)
3. A **stability buffer** confirms a letter only after 15 consistent frames

## Features
- Real-time camera detection
- Sentence builder with space and delete support
- Progress bar showing letter confirmation
- 97% test accuracy
- GPU support (CUDA)
- REST API (FastAPI + Swagger UI)
- Docker containerization
- CI/CD pipeline (GitHub Actions)

## Setup

### 1. Install dependencies

```

pip install torch torchvision mediapipe opencv-python numpy fastapi uvicorn

```

### 2. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) and place it in `ASL_Alphabet_Dataset/asl_alphabet_train/`

### 3. Extract landmarks

```

python "step1_extract_landmarks_improved (1).py"

```

### 4. Train the model

```

python "train_landmark_model (1).py"

```

### 5. Run the camera

```

python ASL_Alphabet_Dataset/camera_mediapipe_test.py

```

### 6. Run the API

```

python api.py

```
Then open http://localhost:8000/docs

### 7. Run with Docker

```

docker build -t asl-detection-api . docker run -p 8000:8000 asl-detection-api

```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Model info and labels |
| POST | `/predict` | Upload image → predicted letter |
| POST | `/predict/landmarks` | Send 63 landmarks → predicted letter |
| DELETE | `/session` | Clear session |

## Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| C | Clear sentence |
| SPACE | Add space |

## Results
| Split | Accuracy |
|-------|----------|
| Train | 96.76% |
| Validation | 96.61% |
| Test | 96.69% |

## Tech Stack
- Python 3.10
- PyTorch 2.5
- MediaPipe
- OpenCV
- FastAPI
- Docker
- GitHub Actions

```