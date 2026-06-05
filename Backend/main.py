from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import imagehash
from io import BytesIO

app = FastAPI(title="AEC Image Similarity Detection", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — SINGLE SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────────────────────────

HASH_SIZE = 16                      # 16×16 dHash grid
MAX_BITS  = HASH_SIZE * HASH_SIZE   # 256 bits

# Classification based on similarity score (0-1):
#   score = 1 - (hamming / 256)
#
# Band                Score ≥   Hamming ≤   Interpretation
# ────────────────────────────────────────────────────────────
# Exact Duplicate      0.973        7         97.3%+ similar
# Likely Duplicate     0.902       25         90.2%+ similar
# Similar – Same       0.781       56         78.1%+ similar
# Similar – Related    0.70        77         70-78% similar
# Different           < 0.70       > 77       < 70% similar

BANDS = [
    (7,   "Exact Duplicate",        "These images are virtually identical."),
    (25,  "Likely Duplicate",        "Very similar with only minor differences."),
    (56,  "Similar – Same Family",   "Significant structural similarity — same design family."),
    (77,  "Similar – Related",       "Moderately similar — related drawing type."),
    (256, "Different",               "Substantially different drawings."),
]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    1. Convert to grayscale
    2. Otsu binarization
    3. Invert to black-on-white
    4. Fixed-region title block masking (bottom 15% height, right 20% width)
    5. Light Gaussian blur to reduce rendering artifacts
    """
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    h, w = inverted.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[int(0.85 * h):, :] = 0    # bottom 15%
    mask[:, int(0.80 * w):] = 0    # right 20%
    masked = cv2.bitwise_and(inverted, inverted, mask=mask)

    blurred = cv2.GaussianBlur(masked, (3, 3), 0)
    return Image.fromarray(blurred)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — dHASH + CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_dhash(image: Image.Image) -> imagehash.ImageHash:
    """
    Compute dHash at HASH_SIZE.
    Returns ImageHash so subtraction (−) gives Hamming distance directly.
    """
    return imagehash.dhash(image, hash_size=HASH_SIZE)


def score_from_hamming(hamming_dist: int) -> float:
    """Single formula: similarity = 1 − (hamming / MAX_BITS)."""
    return 1.0 - (hamming_dist / MAX_BITS)


def classify(hamming_dist: int) -> tuple[str, str]:
    """
    Classify using Hamming thresholds (single source of truth).
    Label and score come from the same hamming_dist → always consistent.
    """
    for threshold, label, description in BANDS:
        if hamming_dist <= threshold:
            return label, description
    # Fallback
    return BANDS[-1][1], BANDS[-1][2]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/compare")
async def compare_images(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    """Compare two PNG images (Stage 0 + Stage 1)."""

    # Load images
    try:
        first_image = Image.open(BytesIO(await first.read())).convert("RGB")
        second_image = Image.open(BytesIO(await second.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open images: {e}")

    # Process and compare
    try:
        hash1 = compute_dhash(preprocess_image(first_image))
        hash2 = compute_dhash(preprocess_image(second_image))

        hamming_dist = hash1 - hash2          # imagehash built-in Hamming
        score = score_from_hamming(hamming_dist)
        band, description = classify(hamming_dist)

        return {
            "hamming_distance": hamming_dist,
            "dhash_similarity": round(score, 4),
            "hybrid_score": round(score, 4),
            "band": band,
            "band_description": description,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
