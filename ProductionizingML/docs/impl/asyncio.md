# Asyncio Deep Dive

Async is counterintuitive until one idea clicks: a single thread that, whenever a task would wait, parks it and runs something else instead. Everything else, the event loop, coroutines, `await`, `gather`, the blocking sin, follows from that.

!!! tip "Rapid Recall"
    The event loop is a `while True` loop on one thread that runs a task until it hits `await` on something not ready, parks it, and runs the next ready task, resuming the parked one when its awaited thing completes. That is how one thread juggles thousands of concurrent I/O operations. Calling an `async def` returns a coroutine object (a paused computation) that does nothing until awaited. The cardinal sin is calling a blocking function (like `time.sleep` or sync `requests`) inside a coroutine, which freezes the entire loop because all tasks share one thread. `await` does not mean "block here", it means "release control so others can run." If you must call blocking code, offload it with `asyncio.to_thread`.

## §1 The event loop

The **event loop** is a `while True` loop running on **one thread**. It maintains a queue of tasks. Each task runs until it hits an `await` on something that isn't ready yet (a network response, a timer). At that `await`, the task **yields control back to the loop**, saying "wake me when this is ready." The loop then runs the next ready task. When the awaited thing completes, the loop resumes the parked task.

The magic: **while task A waits for the network, task B runs.** No thread sits idle. This is why a single async thread can handle thousands of simultaneous connections, they're nearly all waiting at any given moment, and the one thread weaves between the few that are actually ready.

A crucial consequence: if a task does CPU work (no `await`), it **hogs the loop**, no other task runs until it finishes. Async is for I/O concurrency, not CPU parallelism.

## §2 Coroutines, async def, await

- `async def f(): ...` defines a **coroutine function**. Calling `f()` does NOT run it, it returns a **coroutine object** (a paused computation).
- `await x` means "pause here, let the loop run other things, resume when `x` is done." You can only `await` inside an `async def`.
- To actually run a coroutine: `await` it (from inside async code) or `asyncio.run(coro)` (from sync code, top level).

```python
async def greet(name):
    await asyncio.sleep(0.1)        # simulate I/O
    return f"Hello, {name}"

coro = greet("Samarth")            # returns a coroutine object; nothing ran yet
result = asyncio.run(greet("Samarth"))   # now it runs
```

## §3 The cardinal sin: blocking the event loop

The #1 async bug. If you call a **blocking** function (like `time.sleep()` or a synchronous `requests.get()`) inside a coroutine, you **freeze the entire event loop**, every other task stops, because they all share the one thread.

```python
# RIGHT: await asyncio.sleep yields to the loop; 5 tasks overlap -> ~0.5s
async def good(i):
    await asyncio.sleep(0.5)
    return i

# WRONG: time.sleep blocks the loop; 5 tasks serialize -> ~2.5s
async def bad(i):
    time.sleep(0.5)        # BLOCKING - freezes the whole loop
    return i
```

The takeaway: in async code, every potentially-slow operation must be **awaitable and non-blocking**. That's why async needs special libraries: `httpx`/`aiohttp` instead of `requests`, `asyncpg` instead of `psycopg2`, `aiofiles` instead of plain file I/O. If you must call blocking code, offload it with `await asyncio.to_thread(blocking_fn, args)`, which runs it in a thread pool so it doesn't block the loop.

## §4 gather, create_task, and to_thread

Two ways to run coroutines concurrently:

- **`asyncio.gather(*coros)`** runs several coroutines concurrently, waits for **all**, and returns results in order. The workhorse. With `return_exceptions=True`, failures come back as exception objects instead of raising the first one.
- **`asyncio.create_task(coro)`** schedules a coroutine to run **in the background** immediately and returns a `Task` you can await later.

```python
async def fetch(item_id):
    await asyncio.sleep(0.2)              # simulate an API call
    return {"id": item_id, "data": item_id ** 2}

async def fetch_many():
    # 10 "API calls" happen concurrently -> ~0.2s instead of 2.0s sequential.
    return await asyncio.gather(*[fetch(i) for i in range(10)])
```

When you're in async code but must call something blocking, offload it so the loop keeps running:

```python
def legacy_blocking_call(x):
    time.sleep(0.4)
    return x * 10

async def use_legacy_safely():
    # Each blocking call runs in a thread pool; the loop stays free.
    return await asyncio.gather(*[asyncio.to_thread(legacy_blocking_call, i) for i in range(5)])
```

`asyncio.to_thread` is the bridge: it lets async code use blocking libraries without freezing the loop. The same pattern, plus per-call timeouts via `asyncio.wait_for` and turning failures into structured results, is the bread and butter of async services.

## Interview Questions

**Q: What is the event loop?**
A single-threaded loop that manages and schedules coroutines. It runs a task until the task hits an `await` on something not yet ready, parks it, and runs another ready task; when the awaited operation completes, it resumes the parked task. It is how one thread juggles thousands of concurrent I/O operations, almost all are waiting at any instant, and the loop weaves between the few that are ready.

**Q: What's the difference between a coroutine and a function?**
Calling a normal function runs it immediately. Calling a coroutine function (`async def`) returns a coroutine object, a paused computation that does nothing until you `await` it or pass it to `asyncio.run`/`gather`. Coroutines can suspend at `await` points and resume later; regular functions run start to finish.

**Q (trap): I put `time.sleep(5)` in my async endpoint and the whole API froze for all users. Why?**
`time.sleep` is blocking, it does not yield to the event loop. Since async runs on one thread, blocking it freezes every concurrent task. Use `await asyncio.sleep(5)` instead, and for unavoidable blocking calls offload with `await asyncio.to_thread(...)` so the loop stays responsive.

**Q: Can asyncio make CPU-bound code faster?**
No. Asyncio is single-threaded concurrency for I/O, it overlaps waiting, not computing. A CPU-bound coroutine with no `await` just hogs the loop. For CPU parallelism you need multiprocessing. A common production pattern is an async web server that offloads CPU-heavy inference to a `ProcessPoolExecutor`.

**Q (trap): Does `await` mean "wait here and block"?**
No, that's the counterintuitive part. `await` means "pause this coroutine and let the event loop run other coroutines until the awaited thing is ready." It is the opposite of blocking: the explicit point where you release control so others can progress. Blocking is when you do not yield (like `time.sleep`); `await` is precisely how you avoid blocking.
