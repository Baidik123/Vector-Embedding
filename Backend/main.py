import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import imagehash
from io import BytesIO

try:
    from transformers import AutoModel, AutoProcessor
    import torch
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    from torchvision import models as tv_models
    import torchvision.transforms as tv_transforms
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

if _HAS_TRANSFORMERS:
    try:
        _dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-small")
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
        _dino_model.eval()
        DINO_AVAILABLE = True
    except Exception:
        DINO_AVAILABLE = False

    try:
        _dino_large_processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        _dino_large_model = AutoModel.from_pretrained("facebook/dinov2-large")
        _dino_large_model.eval()
        DINO_LARGE_AVAILABLE = True
    except Exception:
        DINO_LARGE_AVAILABLE = False

    try:
        _dino_base_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        _dino_base_model = AutoModel.from_pretrained("facebook/dinov2-base")
        _dino_base_model.eval()
        DINO_BASE_AVAILABLE = True
    except Exception:
        DINO_BASE_AVAILABLE = False

    try:
        _clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        CLIP_AVAILABLE = True
    except Exception:
        CLIP_AVAILABLE = False

    try:
        _dino_giant_processor = AutoProcessor.from_pretrained("facebook/dinov2-giant")
        _dino_giant_model = AutoModel.from_pretrained("facebook/dinov2-giant")
        _dino_giant_model.eval()
        DINO_GIANT_AVAILABLE = True
    except Exception:
        DINO_GIANT_AVAILABLE = False

    try:
        _efficientnet = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.DEFAULT)
        _efficientnet.classifier = torch.nn.Identity()
        _efficientnet.eval()
        EFFICIENTNET_AVAILABLE = True
    except Exception:
        EFFICIENTNET_AVAILABLE = False

    try:
        _convnext = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT)
        _convnext.classifier = torch.nn.Sequential()
        _convnext.eval()
        CONVNEXT_AVAILABLE = True
    except Exception:
        CONVNEXT_AVAILABLE = False

    try:
        _resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        _resnet.fc = torch.nn.Identity()
        _resnet.eval()
        RESNET_AVAILABLE = True
    except Exception:
        RESNET_AVAILABLE = False

    _cnn_transform = tv_transforms.Compose([
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

else:
    DINO_AVAILABLE = False
    DINO_LARGE_AVAILABLE = False
    DINO_BASE_AVAILABLE = False
    CLIP_AVAILABLE = False
    DINO_GIANT_AVAILABLE = False
    EFFICIENTNET_AVAILABLE = False
    CONVNEXT_AVAILABLE = False
    RESNET_AVAILABLE = False


app = FastAPI(title="AEC Image Similarity Detection", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


HASH_SIZE = 16
MAX_BITS  = HASH_SIZE * HASH_SIZE

DHASH_WEIGHT      = 0.4
DINO_WEIGHT       = 0.3
DINO_LARGE_WEIGHT = 0.3

DHASH_BANDS = [
    (7,   "Exact Duplicate",       "These images are virtually identical."),
    (25,  "Likely Duplicate",       "Very similar with only minor differences."),
    (56,  "Similar – Same Family",  "Significant structural similarity — same design family."),
    (77,  "Similar – Related",      "Moderately similar — related drawing type."),
    (256, "Different",              "Substantially different drawings."),
]

DINO_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

DINO_LARGE_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

DINO_GIANT_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

HYBRID_BANDS = [
    (0.973, "Exact Duplicate",       "These images are virtually identical."),
    (0.902, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.781, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.699, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

DINO_BASE_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

CLIP_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

EFFICIENTNET_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

CONVNEXT_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",       "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]

RESNET_BANDS = [
    (0.812, "Exact Duplicate",       "These images are virtually identical."),
    (0.693, "Likely Duplicate",      "Very similar with only minor differences."),
    (0.602, "Similar – Same Family",  "Significant structural similarity — same design family."),
    (0.563, "Similar – Related",      "Moderately similar — related drawing type."),
    (0.0,   "Different",              "Substantially different drawings."),
]


def preprocess_image(image: Image.Image) -> Image.Image:
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    h, w = inverted.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[int(0.85 * h):, :] = 0    
    mask[:, int(0.80 * w):] = 0    
    masked = cv2.bitwise_and(inverted, inverted, mask=mask)

    blurred = cv2.GaussianBlur(masked, (3, 3), 0)
    return Image.fromarray(blurred)


def compute_dhash(image: Image.Image) -> imagehash.ImageHash:
    return imagehash.dhash(image, hash_size=HASH_SIZE)


def score_from_hamming(hamming_dist: int) -> float:
    return 1.0 - (hamming_dist / MAX_BITS)


def classify_dhash(hamming_dist: int) -> tuple[str, str]:
    for threshold, label, description in DHASH_BANDS:
        if hamming_dist <= threshold:
            return label, description
    return DHASH_BANDS[-1][1], DHASH_BANDS[-1][2]


def _run_dhash(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous dHash pipeline — safe to call from thread pool."""
    hash1 = compute_dhash(preprocess_image(img1))
    hash2 = compute_dhash(preprocess_image(img2))
    hamming_dist = hash1 - hash2
    score = score_from_hamming(hamming_dist)
    band, description = classify_dhash(hamming_dist)
    return {
        "algorithm": "dhash",
        "similarity": round(score, 4),
        "hamming_distance": hamming_dist,
        "band": band,
        "band_description": description,
    }


def knn_similarity(emb1, emb2) -> float:
    e1 = F.normalize(emb1, p=2, dim=1).numpy()
    e2 = F.normalize(emb2, p=2, dim=1).numpy()
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(e1)
    dist, _ = nn.kneighbors(e2)
    return 1.0 / (1.0 + dist[0][0])


def knn_to_cosine_scale(knn_sim: float) -> float:
    dist = (1.0 / knn_sim) - 1.0
    return 1.0 - (dist ** 2) / 2.0


def get_dino_embedding(image: Image.Image):
    inputs = _dino_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _dino_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def classify_dino(score: float) -> tuple[str, str]:
    for threshold, label, description in DINO_BANDS:
        if score >= threshold:
            return label, description
    return DINO_BANDS[-1][1], DINO_BANDS[-1][2]


def _run_dino(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous DINO Small inference — safe to call from thread pool."""
    emb1 = get_dino_embedding(img1)
    emb2 = get_dino_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_dino(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "dino_small",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_dino_large_embedding(image: Image.Image):
    inputs = _dino_large_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _dino_large_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def classify_dino_large(score: float) -> tuple[str, str]:
    for threshold, label, description in DINO_LARGE_BANDS:
        if score >= threshold:
            return label, description
    return DINO_LARGE_BANDS[-1][1], DINO_LARGE_BANDS[-1][2]


def _run_dino_large(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous DINO Large inference — safe to call from thread pool."""
    emb1 = get_dino_large_embedding(img1)
    emb2 = get_dino_large_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_dino_large(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "dino_large",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_dino_giant_embedding(image: Image.Image):
    inputs = _dino_giant_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _dino_giant_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def classify_dino_giant(score: float) -> tuple[str, str]:
    for threshold, label, description in DINO_GIANT_BANDS:
        if score >= threshold:
            return label, description
    return DINO_GIANT_BANDS[-1][1], DINO_GIANT_BANDS[-1][2]


def _run_dino_giant(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous DINO Giant inference — safe to call from thread pool."""
    emb1 = get_dino_giant_embedding(img1)
    emb2 = get_dino_giant_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_dino_giant(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "dino_giant",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_dino_base_embedding(image: Image.Image):
    inputs = _dino_base_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _dino_base_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def classify_dino_base(score: float) -> tuple[str, str]:
    for threshold, label, description in DINO_BASE_BANDS:
        if score >= threshold:
            return label, description
    return DINO_BASE_BANDS[-1][1], DINO_BASE_BANDS[-1][2]


def _run_dino_base(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous DINO Base inference — safe to call from thread pool."""
    emb1 = get_dino_base_embedding(img1)
    emb2 = get_dino_base_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_dino_base(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "dino_base",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_clip_embedding(image: Image.Image):
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
    return outputs.last_hidden_state.mean(dim=1)


def classify_clip(score: float) -> tuple[str, str]:
    for threshold, label, description in CLIP_BANDS:
        if score >= threshold:
            return label, description
    return CLIP_BANDS[-1][1], CLIP_BANDS[-1][2]


def _run_clip(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous CLIP inference — safe to call from thread pool."""
    emb1 = get_clip_embedding(img1)
    emb2 = get_clip_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_clip(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "clip",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def _cnn_preprocess(image: Image.Image):
    preprocessed = preprocess_image(image).convert("RGB")
    return _cnn_transform(preprocessed).unsqueeze(0)


def get_efficientnet_embedding(image: Image.Image):
    tensor = _cnn_preprocess(image)
    with torch.no_grad():
        return _efficientnet(tensor)


def classify_efficientnet(score: float) -> tuple[str, str]:
    for threshold, label, description in EFFICIENTNET_BANDS:
        if score >= threshold:
            return label, description
    return EFFICIENTNET_BANDS[-1][1], EFFICIENTNET_BANDS[-1][2]


def _run_efficientnet(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous EfficientNet-B4 inference — safe to call from thread pool."""
    emb1 = get_efficientnet_embedding(img1)
    emb2 = get_efficientnet_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_efficientnet(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "efficientnet_b4",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_convnext_embedding(image: Image.Image):
    tensor = _cnn_preprocess(image)
    with torch.no_grad():
        output = _convnext(tensor)
    return output.view(output.size(0), -1)


def classify_convnext(score: float) -> tuple[str, str]:
    for threshold, label, description in CONVNEXT_BANDS:
        if score >= threshold:
            return label, description
    return CONVNEXT_BANDS[-1][1], CONVNEXT_BANDS[-1][2]


def _run_convnext(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous ConvNeXt-Tiny inference — safe to call from thread pool."""
    emb1 = get_convnext_embedding(img1)
    emb2 = get_convnext_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_convnext(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "convnext_tiny",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def get_resnet_embedding(image: Image.Image):
    tensor = _cnn_preprocess(image)
    with torch.no_grad():
        return _resnet(tensor)


def classify_resnet(score: float) -> tuple[str, str]:
    for threshold, label, description in RESNET_BANDS:
        if score >= threshold:
            return label, description
    return RESNET_BANDS[-1][1], RESNET_BANDS[-1][2]


def _run_resnet(img1: Image.Image, img2: Image.Image) -> dict:
    """Synchronous ResNet-50 inference — safe to call from thread pool."""
    emb1 = get_resnet_embedding(img1)
    emb2 = get_resnet_embedding(img2)
    knn_score = knn_similarity(emb1, emb2)
    band, description = classify_resnet(knn_score)
    display_score = knn_to_cosine_scale(knn_score)
    return {
        "algorithm": "resnet50",
        "similarity": round(display_score, 4),
        "band": band,
        "band_description": description,
    }


def classify_hybrid(score: float) -> tuple[str, str]:
    for threshold, label, description in HYBRID_BANDS:
        if score >= threshold:
            return label, description
    return HYBRID_BANDS[-1][1], HYBRID_BANDS[-1][2]


async def _load_images(
    first: UploadFile, second: UploadFile
) -> tuple[Image.Image, Image.Image]:
    try:
        first_bytes = await first.read()
        second_bytes = await second.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {e}")

    try:
        first_image = Image.open(BytesIO(first_bytes)).convert("RGB")
        second_image = Image.open(BytesIO(second_bytes)).convert("RGB")
        return first_image, second_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode images: {e}")


@app.post("/compare")
async def compare_images(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run_dhash, first_image, second_image)
        return {
            "hamming_distance": result["hamming_distance"],
            "dhash_similarity": result["similarity"],
            "hybrid_score": result["similarity"],
            "band": result["band"],
            "band_description": result["band_description"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/dhash")
async def compare_dhash(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_dhash, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/dino")
async def compare_dino(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not DINO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DINO V2 Small not available — run: pip install torch torchvision transformers",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_dino, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/dino-large")
async def compare_dino_large(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not DINO_LARGE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DINO V2 Large not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_dino_large, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/hybrid")
async def compare_hybrid(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not DINO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Hybrid requires DINO V2 Small — run: pip install torch torchvision transformers",
        )
    if not DINO_LARGE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Hybrid requires DINO V2 Large — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        dhash_result, dino_result, dino_large_result = await asyncio.gather(
            loop.run_in_executor(None, _run_dhash,      first_image, second_image),
            loop.run_in_executor(None, _run_dino,       first_image, second_image),
            loop.run_in_executor(None, _run_dino_large, first_image, second_image),
        )
        hybrid_score = (
            dhash_result["similarity"]       * DHASH_WEIGHT
            + dino_result["similarity"]      * DINO_WEIGHT
            + dino_large_result["similarity"] * DINO_LARGE_WEIGHT
        )
        band, description = classify_hybrid(hybrid_score)
        return {
            "algorithm": "hybrid",
            "similarity": round(hybrid_score, 4),
            "dhash_similarity": dhash_result["similarity"],
            "dino_similarity": dino_result["similarity"],
            "dino_large_similarity": dino_large_result["similarity"],
            "hamming_distance": dhash_result["hamming_distance"],
            "band": band,
            "band_description": description,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/dino-giant")
async def compare_dino_giant(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not DINO_GIANT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DINO V2 Giant not available — model may not be downloaded yet or ran out of memory on load (~8 GB RAM required).",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_dino_giant, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/dino-base")
async def compare_dino_base(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not DINO_BASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DINO V2 Base not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_dino_base, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/clip")
async def compare_clip(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not CLIP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="CLIP not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_clip, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/efficientnet")
async def compare_efficientnet(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not EFFICIENTNET_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="EfficientNet-B4 not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_efficientnet, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/convnext")
async def compare_convnext(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not CONVNEXT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ConvNeXt-Tiny not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_convnext, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.post("/compare/resnet")
async def compare_resnet(
    first: UploadFile = File(...),
    second: UploadFile = File(...),
):
    if not RESNET_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ResNet-50 not available — model may not be downloaded yet or ran out of memory on load.",
        )
    first_image, second_image = await _load_images(first, second)
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_resnet, first_image, second_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "3.0.0",
        "dino_small_available": DINO_AVAILABLE,
        "dino_large_available": DINO_LARGE_AVAILABLE,
        "dino_giant_available": DINO_GIANT_AVAILABLE,
        "dino_base_available": DINO_BASE_AVAILABLE,
        "clip_available": CLIP_AVAILABLE,
        "efficientnet_available": EFFICIENTNET_AVAILABLE,
        "convnext_available": CONVNEXT_AVAILABLE,
        "resnet_available": RESNET_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
