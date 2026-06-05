# FastAPI and Model Serving

FastAPI is the web framework you use to put a model behind an HTTP endpoint. This page covers what it is, the WSGI vs ASGI distinction, Pydantic validation, the sync-vs-async endpoint rule, and the canonical pattern for serving a real scikit-learn model.

!!! tip "Rapid Recall"
    FastAPI is async-first (ASGI), so one worker can interleave many concurrent I/O-bound requests, unlike synchronous WSGI frameworks. It is popular for ML because of async-native serving, Pydantic validation from type hints, automatic OpenAPI docs, and speed. Use `async def` only for awaitable async I/O; for blocking work (sync DB, CPU inference) use plain `def` so FastAPI runs it in a threadpool. The serving pattern to memorize: load the model once at startup via `lifespan`, validate input with Pydantic, expose a `/health` endpoint, raise structured `HTTPException` errors, and return a typed response. Run multiple uvicorn workers to use all cores, because async gives I/O concurrency within one worker but one worker is one core.

## §1 What FastAPI is, and WSGI vs ASGI

FastAPI is a Python web framework for building APIs. To understand why it's popular, you need one piece of background: WSGI vs ASGI.

- **WSGI** (Web Server Gateway Interface) is the old standard. It's **synchronous**: one request is handled at a time per worker; a request that waits on I/O blocks that worker. Flask and Django (classic) are WSGI.
- **ASGI** (Asynchronous Server Gateway Interface) is the modern standard. It's **async-native**: a single worker can handle many concurrent requests by interleaving them at `await` points (the event-loop model from the [Asyncio](asyncio.md) page). FastAPI and Starlette are ASGI.

So FastAPI is "async-first." That matters for I/O-bound endpoints (calling a database, another API, an LLM), one worker concurrently serves many slow requests instead of blocking on each. It is everywhere in ML because it is async-native, validates request and response schemas with Pydantic type hints, generates automatic interactive docs at `/docs`, is among the fastest Python frameworks, and uses type hints throughout.

## §2 Pydantic validation, tested in-process

For request bodies you define a **Pydantic model**: a class with typed fields. FastAPI validates incoming JSON against it, rejects bad data, and gives you a typed Python object. `TestClient` lets you exercise the app in-process, no server or port needed.

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector for the model")
    model_version: str = Field(default="v1")

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # request is validated: request.features is a list[float].
    return PredictionResponse(prediction=sum(request.features), model_version=request.model_version)

client = TestClient(app)
client.post("/predict", json={"features": [1.0, 2.0, 3.0]}).json()
# Malformed input (a string in features, or a missing field) auto-returns a 422 with a clear error.
```

This is FastAPI's killer feature for ML: your API contract is declared as types, so malformed requests are rejected before they reach the model, with no hand-written validation boilerplate.

## §3 Sync vs async endpoints

FastAPI lets you write endpoints as either `def` (sync) or `async def` (async), and the distinction matters for performance:

- **`async def` endpoint** runs on the event loop. Use it when the endpoint does `await`-able I/O (async DB driver, httpx call, async LLM client). Many requests share one worker concurrently.
- **`def` (sync) endpoint** is run by FastAPI in a **threadpool** so it doesn't block the loop. Use it when your work is blocking (a sync DB driver, `model.predict()` on CPU, a blocking library).

The trap is putting blocking code inside an `async def` endpoint. That blocks the event loop and kills concurrency for all requests. If your handler is blocking, either make it a plain `def` or offload with `asyncio.to_thread`. For heavy CPU work, use a plain `def` and consider a process pool or a separate inference service, since a threadpool does not escape the GIL.

## §4 Serving a real scikit-learn model

The canonical shape of a model service: train and save the model offline, load it **once** at startup via `lifespan`, validate input, expose `/health`, and return structured errors and a typed response.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib, numpy as np

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["clf"] = joblib.load("/tmp/model.joblib")   # load ONCE at startup
    yield
    ml_models.clear()                                     # cleanup on shutdown

serving_app = FastAPI(lifespan=lifespan)

class Features(BaseModel):
    features: list[float]

class Prediction(BaseModel):
    predicted_class: int
    probability: float

@serving_app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "clf" in ml_models}

@serving_app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    model = ml_models.get("clf")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(payload.features) != model.n_features_in_:
        raise HTTPException(status_code=400, detail=f"Expected {model.n_features_in_} features")
    proba = model.predict_proba(np.array(payload.features).reshape(1, -1))[0]
    pred = int(proba.argmax())
    return Prediction(predicted_class=pred, probability=float(proba[pred]))
```

The pattern to memorize: load the model once at startup via `lifespan` (loading per request would be catastrophically slow); validate input with Pydantic plus explicit checks; expose a `/health` endpoint so load balancers and orchestrators know the service is ready; raise structured `HTTPException` errors with proper status codes (400 bad input, 503 not ready); and return a typed response via `response_model`.

## §5 Running it for real

In a notebook you use `TestClient`. In production you save the app to a file and run a server with multiple workers:

```bash
# Development: one process, auto-reload
uvicorn main:app --reload --port 8000

# Production: multiple worker processes (one per core) for parallelism + throughput
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Why multiple workers: one uvicorn worker is one process and one event loop on one core. Async gives you I/O concurrency *within* a worker, but to use **all** your cores (and to get CPU parallelism for inference) you run **multiple worker processes**. This is multiprocessing applied to web serving.

## Interview Questions

**Q: Why is FastAPI popular for ML serving?**
It is async-native (ASGI), so it handles concurrent I/O-bound requests efficiently when the model service calls databases or other services; it validates request and response schemas from Pydantic type hints, auto-rejecting bad input; it generates OpenAPI docs at `/docs`; and it is among the fastest Python frameworks. For putting a model behind an HTTP API it is the default.

**Q: What is the difference between WSGI and ASGI?**
WSGI is the older synchronous standard, one request per worker at a time, blocking on I/O (Flask, classic Django). ASGI is the asynchronous standard, where one worker interleaves many concurrent requests at `await` points (FastAPI, Starlette). ASGI suits I/O-bound services with many concurrent slow requests; WSGI is simpler and fine for fast, low-concurrency endpoints.

**Q (trap): Should every FastAPI endpoint be `async def`?**
No. Use `async def` only when the endpoint does awaitable async I/O. If the handler does blocking work (sync DB driver, CPU-bound `model.predict()`, a blocking library), write it as a plain `def`, FastAPI runs sync endpoints in a threadpool so they do not block the loop. Putting blocking code in an `async def` freezes the loop and kills concurrency for all requests.

**Q: Where should you load your model, in the endpoint or at startup?**
At startup, once, via the `lifespan` handler. Loading a model can take hundreds of milliseconds to seconds; doing it per request would make every request unbearably slow and waste memory. Load it once into a module-level holder and reference it in the endpoint.

**Q: Why run multiple uvicorn workers if FastAPI is async?**
Async gives concurrency within one process and core (overlapping I/O waits), but one worker uses one core and one GIL. To use all cores, and to get real parallelism for CPU-bound inference, you run multiple worker processes with `--workers N`. Async handles per-worker I/O concurrency; multiple workers handle multi-core parallelism. You need both.
