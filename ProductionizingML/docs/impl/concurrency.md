# Concurrency and the GIL

The whole topic of "threading vs multiprocessing vs asyncio" reduces to two questions: is your work I/O-bound or CPU-bound, and do you need true parallelism or just concurrency? Answer those and the choice is mechanical.

!!! tip "Rapid Recall"
    I/O-bound work spends most of its time waiting (network, disk, database), so the CPU is idle and you just need to not sit idle: threading or asyncio. CPU-bound work spends its time computing, so the only way to go faster is more CPUs in parallel: multiprocessing. The GIL lets only one thread execute Python bytecode at a time, so threads give no speedup on CPU-bound work but a big speedup on I/O-bound work, because a thread releases the GIL while waiting. Concurrency is dealing with many things at once (interleaving); parallelism is doing many things at once (multiple cores). Asyncio is concurrent but not parallel; multiprocessing is both.

## §1 The most important distinction: I/O-bound vs CPU-bound

This is the insight that makes everything click.

**I/O-bound work** spends most of its time **waiting**, for a network response, a disk read, a database query, an API call. The CPU is idle during the wait. Examples: calling an external API, querying a database, reading files, downloading data.

**CPU-bound work** spends most of its time **computing**, the CPU is pegged at 100%. Examples: matrix multiplication, image resizing, training a model, parsing huge JSON, running inference on a big neural net.

Why it matters:

- For **I/O-bound** work, you don't need more CPUs, you need a way to **not sit idle while waiting**. Start request B while request A is waiting for the network. This means threading or asyncio.
- For **CPU-bound** work, the CPU is the bottleneck. The only way to go faster is **more CPUs working in parallel**. This means multiprocessing.

The classic mistake: using threads to speed up CPU-bound work in Python. It doesn't work, because of the GIL.

## §2 The GIL (Global Interpreter Lock)

CPython (the standard Python) has a **Global Interpreter Lock**: only **one thread can execute Python bytecode at a time**, even on a multi-core machine. The lock is passed around between threads.

Consequences:

- **CPU-bound + threads = no speedup.** Two threads doing math don't run in parallel, they take turns holding the GIL. You get the overhead of threading with none of the benefit.
- **I/O-bound + threads = big speedup.** When a thread is *waiting* for I/O (network, disk), it **releases the GIL**, letting other threads run. So threads overlap their waiting time.

This is *the* reason Python concurrency is confusing. In most languages, threads parallelize CPU work. In Python, they don't, you need processes for that. (Python 3.13+ has an experimental "free-threaded" build without the GIL, but as of 2026 the GIL is still the default and what you'll encounter in production.)

The empirical proof: on a genuinely CPU-bound task, threading gives no speedup while multiprocessing approaches an Nx speedup bounded by core count.

```python
def cpu_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

WORK, N = 5_000_000, 4

# Threading: ~1x speedup. The GIL serializes Python computation.
with ThreadPoolExecutor(max_workers=N) as ex:
    list(ex.map(cpu_task, [WORK] * N))

# Multiprocessing: each process has its own GIL and own core.
# Approaches Nx on an N-core machine; bounded by core count.
with ProcessPoolExecutor(max_workers=N) as ex:
    list(ex.map(cpu_task, [WORK] * N))
```

For **I/O-bound** work the result flips: simulating eight 0.5s "network calls", sequential takes ~4s while both threading and asyncio take ~0.5s, because they overlap all the waiting. The cost of multiprocessing is that each process is a full Python interpreter (heavy memory) and data must be **pickled** to cross process boundaries, so it pays off only for genuinely heavy CPU work.

## §3 The decision table

| Your situation | Use | Why |
|---|---|---|
| I/O-bound, few tasks (<100) | Threading | Simple; works with blocking libraries (requests, psycopg2) |
| I/O-bound, many tasks (1000s) | Asyncio | Scales to huge concurrency cheaply; needs async libraries |
| CPU-bound | Multiprocessing | Only way to parallelize Python computation (bypasses GIL) |
| CPU-bound, heavy numeric (numpy) | Often just numpy | numpy releases GIL in C; vectorize before parallelizing |
| Mix of I/O and CPU | Asyncio + process pool | Async for I/O, offload CPU chunks to a ProcessPoolExecutor |
| Web server handling requests | Async framework (FastAPI) + multiple worker processes | Async per-request I/O concurrency; processes for CPU + cores |

## §4 Concurrency vs parallelism

A common interview question. The crisp version (Rob Pike's framing):

- **Concurrency** is *dealing with* many things at once, structuring your program so multiple tasks can be **in progress** in overlapping time periods. One chef juggling several dishes, switching between them.
- **Parallelism** is *doing* many things at once, multiple tasks **literally executing at the same instant** on different cores. Several chefs each cooking one dish.

Mapping to Python:

- **Asyncio** = concurrency, **no** parallelism (single thread, one task runs at a time, but they interleave).
- **Threading** = concurrency, **no** parallelism for CPU work (GIL), but real overlap for I/O waits.
- **Multiprocessing** = concurrency **and** parallelism (multiple processes on multiple cores).

You can have concurrency without parallelism (asyncio), and parallelism is a *kind* of concurrency. The reason this matters: concurrency is enough for I/O-bound work (you just need to not wait idly); parallelism is required for CPU-bound work (you need more compute happening simultaneously).

## Interview Questions

**Q: What's the difference between concurrency and parallelism?**
Concurrency is structuring a program so multiple tasks can be in progress over overlapping time, they take turns. Parallelism is multiple tasks executing simultaneously on different cores. Asyncio is concurrent but not parallel (one thread). Multiprocessing is both. Concurrency suffices for I/O-bound work; parallelism is needed for CPU-bound work.

**Q: Why doesn't Python threading speed up CPU-bound code?**
The GIL allows only one thread to execute Python bytecode at a time. For CPU-bound work, threads just take turns holding the GIL, no parallelism, only overhead. For I/O-bound work threads do help, because a thread releases the GIL while waiting on I/O, letting others run.

**Q (trap): I used `ThreadPoolExecutor` with 8 workers on a CPU-bound task but it's not faster. Why?**
The GIL. Threads can't parallelize Python computation. Switch to `ProcessPoolExecutor` for true parallelism across cores, or, if the work is numeric, vectorize with numpy (which releases the GIL in its C routines) before reaching for processes. Also check that process startup and data pickling overhead is not dominating for small tasks.

**Q: When does threading actually help in Python?**
For I/O-bound work: network calls, disk reads, database queries, API requests. While one thread waits on I/O it releases the GIL and others proceed. Also, calls into C extensions that release the GIL (numpy, some crypto libraries) can parallelize even CPU work via threads, but that is the C code parallelizing, not Python.

**Q: What's the cost of multiprocessing vs threading?**
Processes are heavyweight: each is a full Python interpreter (high memory), and data crossing process boundaries must be pickled (serialization cost, no cheap sharing of large objects). Threads are lightweight and share memory directly. So multiprocessing wins only when the CPU parallelism gain outweighs the process and IPC overhead, that is, for substantial CPU-bound work, not many tiny tasks.
