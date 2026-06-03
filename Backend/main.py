from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import imagehash
from io import BytesIO
import json

app = FastAPI(title="AEC Image Similarity Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocessing Stage 0:
    1. Convert to grayscale
    2. Binarize using Otsu threshold
    3. Invert to black-on-white
    4. Strip title block: mask out bottom 15% height and right 20% width
    5. Apply light Gaussian blur
    """
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inverted = cv2.bitwise_not(binary)

    height, width = inverted.shape
    mask = np.ones((height, width), dtype=np.uint8) * 255

    mask[int(0.85 * height):, :] = 0
    mask[:, int(0.80 * width):] = 0

    masked = cv2.bitwise_and(inverted, inverted, mask=mask)

    blurred = cv2.GaussianBlur(masked, (3, 3), 0)

    return Image.fromarray(blurred)


def compute_dhash(image: Image.Image, hash_size: int = 16) -> str:
    """
    Compute dHash (difference hash) at specified size.
    Default hash_size=16 gives 128-bit hash (16x16 grid).
    """
    return str(imagehash.dhash(image, hash_size=hash_size))


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex hash strings.
    """
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')


def classify_similarity(hamming_dist: int, max_bits: int = 256) -> tuple[str, str]:
    """
    Classify similarity based on Hamming distance.
    Thresholds:
    - Hamming ≤ 6 → EXACT DUPLICATE
    - Hamming 7–18 → LIKELY DUPLICATE
    - Hamming 19–35 → SIMILAR SAME FAMILY
    - Hamming 36–55 → SIMILAR RELATED
    - Hamming > 55 → DIFFERENT
    """
    if hamming_dist <= 6:
        return "Exact Duplicate", "These images are virtually identical."
    elif hamming_dist <= 18:
        return "Likely Duplicate", "These images are very similar with minor differences."
    elif hamming_dist <= 35:
        return "Similar – Same Family", "These images share significant structural similarity."
    elif hamming_dist <= 55:
        return "Similar – Related", "These images are moderately similar."
    else:
        return "Different", "These images are substantially different."


@app.post("/compare")
async def compare_images(first: UploadFile = File(...), second: UploadFile = File(...)):
    """
    Compare two PNG images and return similarity metrics.
    """
    try:
        first_data = await first.read()
        second_data = await second.read()

        first_image = Image.open(BytesIO(first_data)).convert("RGB")
        second_image = Image.open(BytesIO(second_data)).convert("RGB")

        first_processed = preprocess_image(first_image)
        second_processed = preprocess_image(second_image)

        hash1 = compute_dhash(first_processed, hash_size=16)
        hash2 = compute_dhash(second_processed, hash_size=16)

        hamming_dist = hamming_distance(hash1, hash2)
        max_bits = 256
        dhash_similarity = 1 - (hamming_dist / max_bits)

        hybrid_score = dhash_similarity

        band, band_description = classify_similarity(hamming_dist, max_bits)

        return {
            "hamming_distance": hamming_dist,
            "dhash_similarity": round(dhash_similarity, 4),
            "hybrid_score": round(hybrid_score, 4),
            "band": band,
            "band_description": band_description,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
