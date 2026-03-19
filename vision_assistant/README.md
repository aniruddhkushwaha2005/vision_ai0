# 🦯 Vision Assistant for Blind People

Production-grade AI navigation system for visually impaired users.
Real-time object detection + depth estimation + intelligent navigation + voice feedback.

---

## Architecture Overview

```
React Native App
     ↕ WebSocket (base64 frames @ 12fps)
FastAPI Gateway
     ↓ asyncio.gather
   ┌──────────────┬──────────────┐
YOLOv8        MiDaS Depth    Navigation Engine
Detection     Estimation     (smoother + motion)
   └──────────────┴──────────────┘
                  ↓
              TTS Service (gTTS / Azure)
                  ↓
        Audio base64 → client → speaker
```

---

## Project Structure

```
vision_assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory + lifespan
│   │   ├── core/
│   │   │   ├── config.py              # All settings via env vars
│   │   │   └── logger.py              # JSON logging
│   │   ├── api/routes/
│   │   │   ├── stream.py              # WebSocket real-time stream
│   │   │   ├── detection.py           # REST single-frame endpoint
│   │   │   └── health.py              # Health + readiness probes
│   │   ├── services/
│   │   │   ├── detection/
│   │   │   │   └── detector.py        # YOLOv8 singleton
│   │   │   ├── depth/
│   │   │   │   └── depth_estimator.py # MiDaS singleton
│   │   │   ├── navigation/
│   │   │   │   └── navigation_engine.py  # ★ Core intelligence
│   │   │   └── tts/
│   │   │       └── tts_service.py     # Multi-lang TTS with cache
│   │   └── models/
│   │       └── schemas.py             # All Pydantic models
│   ├── tests/
│   │   └── test_navigation_engine.py  # 15 unit tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── screens/
│       │   └── NavigationScreen.tsx   # Main camera UI
│       ├── services/
│       │   └── VisionWebSocketService.ts  # WS client + audio queue
│       ├── utils/
│       │   └── frameUtils.ts
│       └── config.ts
└── infra/
    └── docker/
        ├── Dockerfile.backend         # Multi-stage CUDA image
        ├── docker-compose.yml         # Full stack + GPU profile
        └── prometheus.yml
```

---

## Quick Start (Local Development)

### Prerequisites
- Docker Desktop with GPU support (optional but recommended)
- Node.js 18+ and React Native CLI
- Python 3.11+

### 1. Backend (Docker — recommended)

```bash
cd infra/docker

# CPU mode (works everywhere)
docker compose up --build

# GPU mode (requires NVIDIA GPU + nvidia-container-toolkit)
docker compose --profile gpu up --build
```

API available at:
- REST: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/api/v1/stream
- Metrics: http://localhost:8000/metrics
- Grafana: http://localhost:3000

### 2. Backend (Local Python)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set MIDAS_DEVICE=cpu for CPU-only machines

uvicorn app.main:app --reload --port 8000
```

### 3. Frontend (React Native)

```bash
cd frontend
npm install

# iOS
npx pod-install ios
npx react-native run-ios

# Android
npx react-native run-android
```

Edit `src/config.ts` — set `API_WS_URL` to your machine's LAN IP:
```ts
export const API_WS_URL = 'ws://192.168.1.X:8000';
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v --tb=short
```

---

## Navigation Engine Logic

The navigation engine is a deterministic state machine with temporal smoothing:

```
Frame Input
    ↓
Depth Fusion      ← Fuse MiDaS depth per object
    ↓
Motion Detection  ← BBox area growth = approaching
    ↓
Zone Analysis     ← LEFT / CENTER / RIGHT threat scores
    ↓
Danger Check      ← CRITICAL + approaching → DANGER (skip normal flow)
    ↓
Decision Logic:
    CENTER clear?        → FORWARD
    CENTER blocked:
      LEFT clear?        → TURN_LEFT
      RIGHT clear?       → TURN_RIGHT
      Both blocked?      → STOP
    ↓
Temporal Smoother ← Majority vote (last 5 frames)
    ↓
Dedup Filter      ← Suppress speech if same decision < 2s ago
    ↓
TTS Generation    ← English + Hindi speech text
    ↓
NavigationResult
```

### Priority Scores (configurable in .env)
| Object     | Score |
|------------|-------|
| Car / Truck| 10    |
| Bus        | 9     |
| Motorcycle | 8     |
| Bicycle    | 7     |
| Person     | 6     |
| Dog        | 5     |
| Furniture  | 2–3   |

---

## Deployment

### Render (simplest — CPU only)

```bash
# render.yaml
services:
  - type: web
    name: vision-assistant-api
    env: docker
    dockerfilePath: ./infra/docker/Dockerfile.backend
    envVars:
      - key: MIDAS_DEVICE
        value: cpu
      - key: REDIS_URL
        fromService:
          type: redis
          name: vision-redis
          property: connectionString
```

### AWS EC2 + GPU

```bash
# Launch g4dn.xlarge (T4 GPU) — ~$0.53/hr
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)

ssh -i key.pem ubuntu@<ec2-ip>

git clone https://github.com/your-org/vision-assistant
cd vision-assistant/infra/docker

# Install nvidia-container-toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker compose --profile gpu up -d
```

### Hugging Face Spaces (CPU — free tier)

```bash
# Create a Gradio space pointing to Dockerfile.backend
# Note: WebSocket not supported on HF Spaces free tier
# Use the REST /analyse endpoint instead
```

---

## Performance Benchmarks (approximate)

| Hardware         | YOLOv8n | MiDaS_small | Total/frame |
|------------------|---------|-------------|-------------|
| NVIDIA T4 (GPU)  | ~8ms    | ~12ms       | ~25ms       |
| Apple M2 (MPS)   | ~15ms   | ~20ms       | ~40ms       |
| Intel i7 (CPU)   | ~80ms   | ~120ms      | ~220ms      |

Target: < 100ms end-to-end for real-time feel at 10fps.

---

## Scaling Strategy

For > 100 concurrent users:

1. **Horizontal scaling**: Run multiple backend replicas behind nginx/Traefik
2. **GPU sharing**: Use NVIDIA MIG or TensorRT batch inference
3. **Frame queuing**: Route frames through Celery for burst handling
4. **Model optimization**: Export YOLOv8 to TensorRT (.engine) for 3–5× speedup
5. **CDN for TTS**: Cache gTTS MP3 files in S3/CloudFront by hash

### TensorRT Export (for production GPU)
```bash
yolo export model=yolov8n.pt format=engine device=0 half=True
# Update YOLO_MODEL_PATH=models/yolov8n.engine in .env
```

---

## Environment Variables Reference

See `.env.example` for full documentation of all supported variables.

Key variables:
- `MIDAS_DEVICE` — `cuda` / `cpu` / `mps`
- `NAV_SMOOTHING_WINDOW` — frames for temporal smoothing (default: 5)
- `NAV_CENTER_BLOCK_THRESH` — depth threshold for blocking (default: 0.45)
- `TTS_ENGINE` — `gtts` / `pyttsx3` / `azure`
- `TTS_HINDI_ENABLED` — enable Hindi voice output

---

## Extending the System

### Add a new object class
Edit `NAV_PRIORITY_SCORES` in `.env`:
```
NAV_PRIORITY_SCORES={"car":10,"truck":10,"fire":10,"stairs":7,...}
```

### Add a new language
In `navigation_engine.py`, add entries to `SPEECH_MAP_*` and `OBJECT_NAME_*` dicts.
Pass the language code in the WebSocket `preferred_language` field.

### Swap detection model
Replace `YOLO_MODEL_PATH` with any `.pt` YOLOv8 variant (n/s/m/l/x).
Or implement `DetectorServiceBase` interface to swap in RT-DETR, DETR, etc.

---

## License
MIT
