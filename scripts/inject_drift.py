import asyncio
import httpx
import numpy as np
from PIL import Image
import io
import sys
import random

API_BASE = "http://localhost:8000"
B_NEG = ["terrible awful horrible bad worst garbage", "waste of money poor quality broken junk", "defective product never buying again disaster"]
B_POS = ["amazing wonderful fantastic perfect love", "best ever incredible outstanding beautiful", "superb excellent brilliant flawless great"]

def generate_biased_image(cls: int) -> bytes:
    img = np.zeros((28, 28), dtype=np.uint8)
    if cls % 3 == 0:
        img[4:8, 6:22] = random.randint(150, 255)
        img[8:22, 10:18] = random.randint(150, 255)
    elif cls % 3 == 1:
        img[4:24, 8:13] = random.randint(150, 255)
        img[4:24, 15:20] = random.randint(150, 255)
    else:
        for r in range(6, 24):
            w = int(4 + (r - 6) * 0.6)
            img[r, max(0, 14 - w):min(28, 14 + w)] = random.randint(150, 255)
    img = np.clip(img.astype(int) + np.random.randint(0, 40, (28, 28)), 0, 255).astype(np.uint8)
    image = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

async def main():
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    model = sys.argv[2] if len(sys.argv) > 2 else "fashion_mnist"
    bias_class = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    print(f"Injecting drift: {num} requests: model={model}, bias_class={bias_class}")

    async with httpx.AsyncClient() as client:
        for i in range(num):
            try:
                if model == "sentiment":
                    text = random.choice(B_NEG if bias_class == 0 else B_POS)
                    r = await client.post(f"{API_BASE}/predict/text", json={"text": text}, timeout=10)
                else:
                    img = generate_biased_image(bias_class if random.random() < 0.85 else random.randint(0, 9))
                    r = await client.post(f"{API_BASE}/predict/image?model={model}", files={"file": ("img.png", img, "image/png")}, timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    print(f"  [{i+1:3d}] {model} -> {d['prediction']} (conf: {d['confidence']:.3f})")
            except Exception as e: print(f"  [{i+1:3d}] ERROR: {e}")
            await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(main())
