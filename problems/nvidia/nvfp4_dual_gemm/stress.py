import httpx, asyncio, time, numpy as np

API = "http://localhost:8091/v1/audio/speech"
PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "The restaurant on the corner serves the best pasta I have ever tasted.",
    "After the meeting, we should discuss the quarterly results.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock.",
    "Could you please turn down the music, I'm trying to concentrate on my work.",
    "Learning a new language takes patience and genuine curiosity.",
    "I can't believe how beautiful the sunset looks from up here.",
    "Please remember to bring your identification documents tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time?",
    "The train leaves at half past seven, so we need to arrive before then.",
]

async def req(client, sem, i):
    async with sem:
        t0 = time.perf_counter()
        r = await client.post(API, json={"input": PROMPTS[i % len(PROMPTS)], "voice": "vivian", "language": "English"}, timeout=300)
        dt = time.perf_counter() - t0
        dur = len(r.content) / 48000  # approx audio duration
        return {"latency": dt, "rtf": dt / dur if dur > 0 else 999, "ok": r.status_code == 200}

async def main():
    CONCURRENCY, N = 8, 50
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as c:
        await req(c, asyncio.Semaphore(1), 0)  # warmup
        t0 = time.perf_counter()
        results = await asyncio.gather(*[req(c, sem, i) for i in range(N)])
        wall = time.perf_counter() - t0
    lats = [r["latency"] for r in results]
    rtfs = [r["rtf"] for r in results]
    print(f"Concurrency={CONCURRENCY}  Requests={N}  Wall={wall:.1f}s  Throughput={N/wall:.2f}req/s")
    print(f"Latency p50={np.percentile(lats,50):.3f}s  p95={np.percentile(lats,95):.3f}s  p99={np.percentile(lats,99):.3f}s")
    print(f"RTF    mean={np.mean(rtfs):.3f}  p95={np.percentile(rtfs,95):.3f}")
    print(f"Errors: {sum(1 for r in results if not r['ok'])}")

asyncio.run(main())