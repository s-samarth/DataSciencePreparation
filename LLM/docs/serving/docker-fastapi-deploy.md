# Docker and FastAPI Deploy

A crash course from "Python script that runs a model" to "HTTP service in a container I can deploy anywhere." Minimal FastAPI app, the Dockerfile that wraps it, run commands for GPU and CPU, then the seven production upgrades you need to know about before shipping.

!!! tip "Rapid Recall"
    **FastAPI** = Python web framework, turns functions into HTTP endpoints. **Docker** = packages code + dependencies + Python runtime into one image. Load the model ONCE at startup (never per request). Use Pydantic models for request/response schemas (auto-validates). For production: **swap Transformers for vLLM** (`FROM vllm/vllm-openai:latest` is the whole Dockerfile). **Stream tokens** via SSE for chat UX. **Do not bake model weights into the image** (images become 10-50 GB); download at startup, mount as volume, or use a model store. **Put an API gateway in front** for auth and rate limiting; never expose ports directly. **Modal, Runpod, Kubernetes + vLLM, or managed providers** (Together AI, Anyscale, Fireworks) — pick based on team size and infra preference.

## §1 The mental model

- **FastAPI** = a Python web framework. Turns Python functions into HTTP endpoints.
- **Docker** = a way to package your code + dependencies + Python runtime into a single image that runs identically anywhere.

Together: FastAPI app → Dockerfile builds the image → deploy the image to any cloud (AWS, GCP, Azure, Render, Fly.io, Modal, etc.).

## §2 Step 1 — a minimal FastAPI service

`app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load ONCE at startup (not per-request).
MODEL_NAME = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
        )
    text = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return {"text": text}
```

Run locally:

```bash
pip install fastapi uvicorn transformers torch
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what is 2+2?", "max_tokens": 50}'
```

## §3 Step 2 — the Dockerfile

`Dockerfile`:

```dockerfile
# Start from an NVIDIA CUDA base image for GPU support.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
torch==2.5.0
transformers==4.46.0
accelerate==1.1.0
pydantic==2.9.0
```

## §4 Step 3 — build and run

```bash
# Build the image.
docker build -t my-llm-api:v1 .

# Run with GPU (needs nvidia-docker runtime).
docker run --gpus all -p 8000:8000 my-llm-api:v1

# Run CPU-only (slower but works anywhere).
docker run -p 8000:8000 my-llm-api:v1
```

## §5 Production upgrades

The seven things you should know about before shipping.

### 5.1 Use vLLM instead of Transformers

```dockerfile
FROM vllm/vllm-openai:latest
# That's the whole dockerfile. vLLM ships OpenAI-compatible server out of the box.
```

Run:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Now your API is OpenAI-compatible and benefits from PagedAttention, continuous batching, etc. See [vLLM](vllm.md).

### 5.2 Workers

For CPU-bound APIs, `uvicorn --workers N`. For LLM APIs, usually 1 worker per GPU (model is huge; you cannot replicate). Multi-GPU on a single host is handled by vLLM's tensor parallelism, not Uvicorn workers.

### 5.3 Async endpoints

Use `async def` for I/O-bound handlers, but NOT for the generate call itself (compute-bound, blocks anyway). For long-running requests, use a background queue (Redis, Celery) and return a job ID immediately.

### 5.4 Streaming responses

Users want tokens as they are generated. Use FastAPI's `StreamingResponse` + Server-Sent Events (SSE). vLLM's OpenAI-compatible server does this natively (`stream: true` in the request).

```python
from fastapi.responses import StreamingResponse

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    def event_stream():
        for token in model.generate_stream(...):
            yield f"data: {token}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 5.5 Health checks, metrics, logging

- `/health` endpoint for k8s liveness/readiness probes.
- `/metrics` for Prometheus (vLLM exposes this by default).
- Log to stdout (Docker captures it).
- Use a real logger, not `print()`.

### 5.6 Security

- **Never expose model ports directly.** Put an API gateway (nginx, Traefik, cloud LB) in front.
- **Add auth** (API keys, JWT) at the gateway.
- **Rate limit** per API key.
- **Validate inputs** with Pydantic (FastAPI does this automatically; do not bypass it).
- **Sanitize prompts** if they contain user-supplied content that might be passed to tools.

### 5.7 Model weights

**Do NOT bake model weights into the Docker image.** Images become 10-50 GB. Three patterns:

- **Download at container startup** from HuggingFace Hub or your S3 bucket.
- **Mount as a volume** (k8s PVC or Docker volume).
- **Use a model store** like Modal or Replicate that handles caching and warm-loading for you.

## §6 Deployment targets — 2026 landscape

- **Modal.** Python-native, zero-infra, auto-scales GPUs on demand. Popular for smaller teams. You decorate a function with `@modal.asgi_app()` and deploy.
- **Runpod / Vast.ai.** Cheap GPU rentals, roll-your-own. You manage the container.
- **AWS SageMaker / Bedrock, GCP Vertex, Azure ML.** Cloud-managed serving. Expensive, but enterprise-friendly with compliance certs.
- **Kubernetes + vLLM.** The production standard for anyone serious. Use KServe or Ray Serve on top for autoscaling and traffic management.
- **Together AI, Anyscale, Fireworks.** Managed inference providers. You do not deploy; they do. Best for teams that do not want to run their own infra.

## §7 The minimal-viable production setup

For a small team in 2026 that needs to ship a custom-fine-tuned LLM behind an API in a week, the answer is one of:

- **Modal**, if you are Python-native and want zero infra. Cost: a 2-3× premium vs raw GPU rentals.
- **Kubernetes + vLLM** on a managed cluster (EKS, GKE, AKS), if you have a DevOps person. Cheaper at scale.
- **Together AI** or **Fireworks**, if you can use one of their pre-deployed base models with LoRA adapters. Cheapest and fastest if your model fits their catalog.

For prototypes and demos, Modal wins on time-to-first-deploy. For real production volume, Kubernetes + vLLM wins on cost per token. For "I just want an OpenAI-compatible API," managed providers win on operational simplicity.

## Interview Questions

**Q1: Walk me through what you would do to go from "Python script that runs a model" to "HTTP service in a container."**

Wrap the model in a FastAPI app with health and generate endpoints, Pydantic schemas for request/response. Load the model once at startup, not per request. Write a Dockerfile that starts from `nvidia/cuda:12.4-runtime-ubuntu22.04`, installs Python and dependencies, copies the app, and runs `uvicorn`. Build with `docker build`, run with `docker run --gpus all`. For production, swap to `FROM vllm/vllm-openai:latest` — that gets you PagedAttention, continuous batching, and an OpenAI-compatible API in one image.

**Q2: Why do you load the model at startup and not per request?**

Loading a 7B model takes 30-60 seconds and ~14 GB of VRAM. Per-request loading would make every API call 30 seconds + inference time and exhaust VRAM immediately. The model is shared state across all requests in a process; FastAPI's request handlers reuse the loaded model. The tradeoff is that your container restart is slow (30-60 seconds to first ready), which is what `/health` and Kubernetes readiness probes manage.

**Q3: Why not bake the model weights into the Docker image?**

Three reasons. (1) Image size: weights are 10-50 GB; a Docker image that big is impractical to pull on a fresh node. (2) Versioning: every weight update means rebuilding and re-pushing the entire image. (3) Caching: pulling 30 GB on every cold start defeats the purpose of caching small image layers. Patterns instead: download weights at container startup (with caching), mount as a Kubernetes PVC, or use a model store like Modal that warm-loads weights into containers.

**Q4: What does streaming a chat response look like in FastAPI?**

Use `StreamingResponse` with Server-Sent Events (SSE) format. The endpoint returns an async generator that yields `f"data: {token}\n\n"` for each token. Clients consume via `EventSource` or the OpenAI SDK with `stream=True`. vLLM's OpenAI-compatible server handles this natively; you just pass `stream: true` in the request and the response is SSE-formatted. The reason chat UX feels fast is not raw token throughput, it is TTFT (time-to-first-token) being low and tokens streaming as they generate.

**Q5: When would you pick Modal vs Kubernetes + vLLM vs a managed API for production?**

Modal for small teams that are Python-native and want zero infra; you pay a ~2-3× premium on GPU cost in exchange for never touching k8s. Kubernetes + vLLM for teams with a DevOps person and meaningful volume — cheapest per token at scale, full control. Managed APIs (Together AI, Fireworks, Anyscale) when you can use a popular base model with optional LoRA adapters — fastest time-to-deploy, no infra, but you give up customization and pay per token. Decision: Modal if a prototype, Kubernetes if scale, managed if your model fits an existing catalog.

**Q6: What are the security gotchas of exposing an LLM API publicly?**

(1) Never expose the model port directly — put an API gateway in front. (2) Add auth (API keys, JWT) at the gateway and rate-limit per key. (3) Validate inputs with Pydantic (FastAPI does this automatically; do not bypass it). (4) Sanitize prompts if they contain user-supplied content that gets passed to tools (prompt injection vector). (5) Beware of cost: a single user can run up thousands of dollars in GPU compute via prompt-bombing; rate-limiting by request count is not enough, also limit total tokens-per-minute per key.
