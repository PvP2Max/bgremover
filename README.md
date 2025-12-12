# MODNet Background Removal Service

GPU-accelerated FastAPI service that runs MODNet (portrait matting) to produce high-quality RGBA cutouts tuned for people and event photography. Designed to replace the older `withoutbg` service and integrate with BoothOS batch workflows.

## Features
- MODNet portrait matting on NVIDIA GPUs (loads once and stays on GPU).
- Quality modes (`fast`, `standard`, `high`) to balance throughput vs detail.
- Downloads input images by URL, returns Cloudflare R2 URL to the cutout.
- Clear separation of model loading, preprocessing, inference, post-processing, and API.

## Prerequisites
- NVIDIA GPU with CUDA drivers available to Docker (`--gpus all`).
- MODNet checkpoint on disk (TorchScript preferred) referenced by `MODNET_MODEL_PATH`.
- Cloudflare R2 (or any S3-compatible endpoint) credentials for uploads.

## Environment Variables
- `MODNET_MODEL_PATH` **(required)**: Path to TorchScript or state_dict checkpoint (e.g., `/models/modnet_portrait.ckpt`).
- `MODNET_MAX_LONG_EDGE` (default `1024`): Longest edge used for `fast`/`standard`.
- `MODNET_MAX_LONG_EDGE_HIGH_QUALITY` (default `1536`): Longest edge for `high`.
- `DEFAULT_QUALITY_MODE` (default `standard`): `fast|standard|high`.
- `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` **(required for uploads)**.
- `R2_PUBLIC_BASE_URL` (optional): Public CDN base for returned URLs; falls back to presigned URL when absent.
- `REQUEST_TIMEOUT_SECONDS` (default `30`): Download timeout.
- `LOG_LEVEL` (default `INFO`).

See `.env.example` for a ready-to-copy template.

## Local (Python) Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # installs CPU wheels by default; works on macOS
                                        # for Apple Silicon with GPU, PyTorch will use MPS automatically.
export MODNET_MODEL_PATH=/models/modnet_portrait.ckpt  # adjust to your path
uvicorn modnet_service.api:app --host 0.0.0.0 --port 8000
```
Notes:
- Device selection is automatic: CUDA (Linux GPU) → MPS (Apple Silicon) → CPU.
- On macOS without CUDA, performance will be slower but functional for verification.

## Docker Build & Run
```bash
docker build -t modnet-service:latest .
docker run --gpus all --rm \
  --net apps-net \  # Cloudflare tunnel network
  -p 8000:8000 \
  --env-file .env \
  -v /models:/models \
  modnet-service:latest
```

## API
- Health: `GET /health` → `{"status": "ok"}`
- Background removal:
```bash
curl -X POST http://localhost:8000/remove-bg \
  -H "Content-Type: application/json" \
  -d '{"imageUrl":"https://example.com/input.jpg","qualityMode":"standard"}'
```
Response: `{"outputUrl": "<public-url-to-cutout>", "mode": "standard"}`

## Using the core pipeline directly
- See `scripts/local_test.py` for a quick offline test:
```bash
python scripts/local_test.py --input /path/to/photo.jpg --output /tmp/output.png --mode high
```
This skips R2 and writes `output.png` locally for visual inspection.

## Notes on checkpoints
- TorchScript checkpoints load fastest (`torch.jit.save` from the official MODNet repo).
- Plain `state_dict` checkpoints are also supported; unexpected/missing keys are logged.
- The model is loaded once and cached on first request.

## Operational reminders
- Keep this service running alongside the existing `withoutbg` until validation is complete.
- Attach the container to the Cloudflare tunnel network `apps-net` to expose it at the desired hostname.
