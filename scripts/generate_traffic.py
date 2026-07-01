import asyncio
import httpx
import random
import time
import numpy as np
from PIL import Image
import io
import sys

API_BASE = "http://localhost:8000"
SAMPLE_TEXTS = [
    "this product is absolutely amazing i love it", "terrible quality would not recommend to anyone",
    "great service highly satisfied with purchase", "worst experience ever total waste of money",
    "excellent quality and fast delivery", "very disappointed broken on arrival",
    "beautiful design works perfectly", "horrible product do not buy this"
]

def generate_random_image() -> bytes:
    pixels = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

async def send_image(client, n, model):
    img = generate_random_image()
    try:
        r = await client.post(f"{API_BASE}/predict/image?model={model}", files={"file": ("img.png", img, "image/png")}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            print(f"  [{n:3d}] {model:14s} -> {d['prediction']:12s} conf={d['confidence']:.3f} lat={d['latency']['total_ms']:.1f}ms")
    except Exception as e: print(f"  [{n:3d}] {model} failed: {e}")

async def send_text(client, n):
    text = random.choice(SAMPLE_TEXTS)
    try:
        r = await client.post(f"{API_BASE}/predict/text", json={"text": text}, timeout=10)
        if r.status_code == 200:
            d = r.json()
            print(f"  [{n:3d}] sentiment      -> {d['prediction']:12s} conf={d['confidence']:.3f} lat={d['latency']['total_ms']:.1f}ms")
    except Exception as e: print(f"  [{n:3d}] sentiment failed: {e}")

async def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"sending {total} requests to {API_BASE} (workers={concurrency})")

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def job(n):
            async with sem:
                roll = random.random()
                if roll < 0.4: await send_image(client, n, "fashion_mnist")
                elif roll < 0.7: await send_image(client, n, "mnist_digits")
                else: await send_text(client, n)
                await asyncio.sleep(random.uniform(0.05, 0.2))
        await asyncio.gather(*(job(i + 1) for i in range(total)))

if __name__ == "__main__":
    asyncio.run(main())
