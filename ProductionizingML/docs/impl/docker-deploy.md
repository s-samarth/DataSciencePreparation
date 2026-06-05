# Docker and Deployment

The problem Docker solves is "it works on my machine." Docker packages your code, dependencies, and runtime into an image that runs identically everywhere. This page covers images vs containers, the layer-caching optimization, multi-stage builds, and the four ways to run the image in the cloud.

!!! tip "Rapid Recall"
    An image is the immutable blueprint (a layered filesystem of code plus dependencies plus runtime); a container is a running instance of it. The single most important Dockerfile optimization is layer caching: copy `requirements.txt` and `pip install` before copying code, so a code change reuses the cached dependency layer. Order instructions from least- to most-frequently-changed. Multi-stage builds install in a builder stage and copy only the artifacts into a slim final image. For deployment, the four options trade control for less ops: VM, container service, serverless, managed ML. The 2026 default for a stateless model API is a container service like Fargate or Cloud Run; scale horizontally and stay stateless.

## §1 Images vs containers

- An **image** is a **blueprint**, a frozen, layered filesystem with your code, dependencies, and runtime. Built once, stored, shared. (Like a class.)
- A **container** is a **running instance** of an image, an isolated process with its own filesystem view, network, and so on. You can run many containers from one image. (Like an object.)

Lifecycle: write a `Dockerfile`, `docker build` produces an **image**, `docker run` starts a **container**, push the image to a **registry** (Docker Hub, AWS ECR), and the cloud pulls and runs it.

## §2 A Dockerfile, and the layer-caching optimization

```dockerfile
# Slim base = smaller image, faster pulls, less attack surface.
FROM python:3.12-slim
WORKDIR /app

# Copy ONLY requirements first: a layer-caching optimization.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code AFTER deps (it changes more often).
COPY . .

EXPOSE 8000
# 0.0.0.0 (not 127.0.0.1) so connections from outside the container reach the app.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Each Dockerfile instruction creates a **layer**. Docker **caches** layers and only rebuilds from the first one that changed. Your code changes constantly; your dependencies rarely do. By installing deps first, Docker caches the slow `pip install` layer, so a code-only change reuses it and rebuilds in seconds. If you copied everything first, any code change would force a full re-install. **Order instructions from least-frequently-changed to most-frequently-changed.**

## §3 Multi-stage builds and image hygiene

A multi-stage build installs in a "builder" stage (with build tools), then copies only the finished artifacts into a clean slim final stage, so the final image excludes build tools and caches.

```dockerfile
# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: slim runtime ----
FROM python:3.12-slim
COPY --from=builder /opt/venv /opt/venv     # copy only the installed venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Add a `.dockerignore` (like `.gitignore`) to keep junk out of the image: `__pycache__/`, `.git/`, `.venv/`, `*.ipynb`, `data/`. Image-size best practices: use `-slim` base images, multi-stage builds, `--no-cache-dir` on pip, combine `RUN` commands with `&&`, and do not bake large datasets or models you can mount or download at runtime.

The Docker commands you run in a terminal:

```bash
docker build -t my-model-service:v1 .                 # build an image
docker run -p 8000:8000 my-model-service:v1           # run a container
docker push <account>.dkr.ecr.<region>.amazonaws.com/my-model-service:v1   # push to a registry
```

## §4 The four ways to run in the cloud

From most control / most ops work, to least:

| Option | What it is | You manage | Cloud manages | Example |
|---|---|---|---|---|
| **VM** | A rented virtual machine | OS, runtime, scaling, your app | Hardware | AWS EC2, GCP Compute Engine |
| **Container service** | Run containers, cloud handles hosts | Container image, config | Hosts, orchestration, scaling | AWS ECS/Fargate, GCP Cloud Run, K8s |
| **Serverless** | Run a function per request | Just the function code | Everything else (scales to zero) | AWS Lambda, GCP Cloud Functions |
| **Managed ML** | Purpose-built ML serving | Model artifact + config | Serving infra, autoscaling, monitoring | AWS SageMaker, GCP Vertex AI |

The 2026 default for a stateless model API is a **container service** like AWS Fargate or GCP Cloud Run: push your image, set CPU/memory and min/max instances, and it autoscales on traffic with no servers to patch. Reach for **VMs** when you need specific GPUs/drivers or persistent state, **serverless** for spiky low-volume or event-driven work (mind cold starts and the dependency-size limits that bite ML), and **managed ML** when you want built-in autoscaling, monitoring, and A/B testing and accept the lock-in.

The deploy path is automated by CI/CD: a push to main triggers a pipeline that builds the image, pushes to the registry, and rolls out the new version. Scaling concepts: **horizontal scaling** (more replicas behind a load balancer, the cloud-native way, which requires statelessness), **vertical scaling** (a bigger instance, limited ceiling), an **autoscaling policy** on a metric like CPU or p95 latency with min and max replicas, and a load balancer that routes only to healthy replicas via `/health`. Statelessness matters because any replica can then handle any request, so you can add or remove replicas freely; state lives in external stores (Postgres, Redis, S3), not in the process.

## Interview Questions

**Q: What's the difference between a Docker image and a container?**
An image is the immutable blueprint, a layered filesystem snapshot of code plus dependencies plus runtime. A container is a running instance of an image, an isolated process. One image, many containers. The analogy: image is a class, container is an object.

**Q: Why does Dockerfile instruction order matter?**
Layer caching. Each instruction is a cached layer, and Docker rebuilds from the first changed layer onward. Put rarely-changing steps (installing dependencies) before frequently-changing ones (copying code), so a code change reuses the cached dependency layer and rebuilds in seconds instead of re-running `pip install` every time.

**Q (trap): My Docker image is 3GB. How do I shrink it?**
Use a `-slim` base instead of full `python` (saves hundreds of MB), a multi-stage build to drop build tools, `pip install --no-cache-dir`, a `.dockerignore` to exclude data, notebooks, and git, and avoid baking large datasets or models you can mount or download at runtime. Also combine `RUN` layers and clean caches in the same layer.

**Q: EC2 vs Fargate vs Lambda for a model API, how do you choose?**
EC2 (a VM) gives full control but you manage OS, scaling, and deploys, pick it for special hardware/GPU/driver needs or persistent state. Fargate (containers) is the sweet spot for stateless model APIs, push an image and it autoscales with no server management. Lambda (serverless) suits spiky or event-driven traffic and scales to zero, but cold starts and dependency-size limits make it awkward for large ML deps and GPUs.

**Q: Why should a serving service be stateless?**
So any replica can handle any request, letting you add or remove replicas freely and route traffic anywhere via the load balancer. If a replica held per-user state in memory, requests would stick to specific instances, breaking autoscaling and losing data when an instance dies. Keep state in external stores (DB, Redis, S3) and keep the service itself stateless.
