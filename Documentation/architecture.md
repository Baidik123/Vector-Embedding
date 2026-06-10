# Architecture Overview

## System Summary

This project is a two-tier image comparison application for PNG drawings. The frontend is an Angular standalone application that handles file upload, preview, algorithm selection, and result presentation. The backend is a FastAPI service that receives two uploaded images, runs one or more similarity algorithms, and returns normalized comparison results.

The application is designed around direct browser-to-API communication:

- Frontend runs separately from backend.
- Frontend sends multipart form data containing two PNG files.
- Backend returns JSON responses per selected algorithm.
- Frontend renders each algorithm result independently in the comparison modal.

## High-Level Architecture

```text
+-------------------------+            HTTP / JSON + multipart/form-data            +-------------------------+
| Angular Frontend        | -----------------------------------------------------> | FastAPI Backend         |
|                         |                                                        |                         |
| - File upload UI        | <----------------------------------------------------- | - Image decoding        |
| - PNG preview           |                  similarity results                    | - Similarity algorithms |
| - Algorithm selection   |                                                        | - Band classification   |
| - Slider comparison UI  |                                                        | - Health endpoint       |
+-------------------------+                                                        +-------------------------+
```

## Frontend Architecture

### Technology

- Framework: Angular 21 standalone application
- Language: TypeScript
- Styling: Component-scoped CSS plus minimal global styles
- State approach: Local component state using Angular signals
- Networking: Native `fetch` API

### Frontend Structure

The frontend is intentionally compact and mostly component-driven.

#### 1. Root application component

`Frontend/src/app/app.component.ts` is the entry screen and orchestrates the upload workflow.

Responsibilities:

- Maintains upload state for the first and second PNG.
- Stores preview object URLs for both files.
- Validates file type as PNG before allowing comparison.
- Tracks drag-and-drop state independently for both upload zones.
- Tracks selected algorithms.
- Opens and closes the comparison overlay.
- Revokes object URLs on replacement and component destroy to avoid browser memory leaks.

#### 2. Comparison overlay component

`Frontend/src/app/pdf-compare/pdf-compare.ts` is the result and visualization layer.

Responsibilities:

- Receives the two selected files and preview URLs as inputs.
- Automatically triggers comparison when files and selected algorithms are available.
- Sends one backend request per selected algorithm.
- Maintains per-algorithm loading, success, and error state.
- Displays score, qualitative band, band description, and algorithm-specific metrics.
- Renders a visual image comparison slider with zoom controls.

Note: despite the folder/component name `pdf-compare`, the current implementation compares PNG images, not PDFs.

### Frontend Data Flow

1. User uploads or drags a PNG into each upload zone.
2. `app.component.ts` validates that each file is PNG.
3. Browser object URLs are generated for preview display.
4. User selects one or more algorithms:
   - `dhash`
   - `dino_small`
   - `dino_large`
   - `hybrid`
5. User clicks `Compare PNGs`.
6. The comparison overlay component opens.
7. The overlay sends parallel POST requests to the backend, one per selected algorithm.
8. Responses update result cards independently, so one algorithm can fail while others succeed.
9. The user reviews side-by-side visual overlap and backend similarity scores together.

### Frontend UI Model

The UI has two distinct layers:

- Upload page
  - Two independent PNG drop zones
  - Local preview and file metadata
  - Algorithm multi-select area
  - Compare action button
- Full-screen comparison modal
  - Overlay image slider
  - Zoom in, zoom out, reset zoom
  - Results panel with one card per algorithm

### Frontend Routing and Application Setup

- Routing is present but currently unused; `app.routes.ts` exports an empty route array.
- The app is bootstrapped with Angular standalone APIs from `main.ts`.
- No shared service layer, store, or HTTP client abstraction exists yet.
- Backend base URL is hardcoded in the comparison component as `http://localhost:8000`.

## Backend Architecture

### Technology

- Framework: FastAPI
- Language: Python
- Image processing: Pillow, OpenCV, NumPy, ImageHash
- AI embeddings: Hugging Face Transformers + PyTorch
- Similarity search helper: scikit-learn `NearestNeighbors`

### Backend Structure

The backend is implemented in a single file: `Backend/main.py`.

Its responsibilities are grouped into five layers:

#### 1. Application and middleware setup

- Creates the FastAPI app with title `AEC Image Similarity Detection`.
- Enables permissive CORS for all origins, methods, headers, and credentials.

#### 2. Model and dependency initialization

- Attempts to import optional AI-related libraries.
- Tries to load DINO V2 Small and DINO V2 Large independently.
- Exposes availability through booleans:
  - `DINO_AVAILABLE`
  - `DINO_LARGE_AVAILABLE`

This means:

- dHash remains usable even if AI dependencies are unavailable.
- DINO Small and DINO Large can fail independently.
- Hybrid depends on DINO Small being available.

#### 3. Preprocessing and feature extraction

For dHash only, the backend preprocesses images before hashing:

- Convert RGB to grayscale
- Apply Otsu thresholding
- Invert to black-on-white
- Mask the bottom 15 percent and right 20 percent
- Apply a light Gaussian blur

This preprocessing is intended to reduce noise and de-emphasize title block regions before generating the perceptual hash.

For DINO algorithms, the backend uses raw RGB images without the dHash preprocessing path.

#### 4. Similarity scoring and classification

The backend supports four comparison modes.

##### dHash

- Produces a 16x16 perceptual hash.
- Uses Hamming distance across 256 bits.
- Converts distance to score using `1 - (distance / 256)`.
- Classifies result using `DHASH_BANDS`.

##### DINO V2 Small

- Generates a 384-dimensional image embedding.
- L2-normalizes embeddings.
- Uses Euclidean nearest-neighbor distance and converts it into `1 / (1 + distance)`.
- Converts that KNN score into a cosine-equivalent display score for the response.
- Classifies using `DINO_BANDS`.

##### DINO V2 Large

- Same scoring flow as DINO Small.
- Uses a larger 1024-dimensional model and separate availability flag.
- Classifies using `DINO_LARGE_BANDS`.

##### Hybrid

- Runs dHash and DINO Small concurrently.
- Combines both using weighted averaging:

```text
hybrid_score = (dhash_similarity * 0.4) + (dino_similarity * 0.6)
```

- Uses `HYBRID_BANDS` for the final qualitative label.
- Returns component scores and Hamming distance in addition to the combined score.

#### 5. API layer

The backend exposes the following HTTP endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/compare` | Legacy dHash-compatible endpoint kept for backward compatibility |
| POST | `/compare/dhash` | Runs dHash comparison |
| POST | `/compare/dino` | Runs DINO V2 Small comparison |
| POST | `/compare/dino-large` | Runs DINO V2 Large comparison |
| POST | `/compare/hybrid` | Runs combined dHash + DINO Small comparison |
| GET | `/health` | Returns service status, version, and model availability |

All compare endpoints expect multipart form-data with:

- `first`: first uploaded image
- `second`: second uploaded image

### Backend Execution Model

- Uploaded files are read asynchronously by FastAPI.
- Images are decoded into Pillow `Image` objects.
- CPU-heavy or model-heavy comparison logic is executed via `run_in_executor(...)` to avoid blocking the event loop.
- Hybrid uses `asyncio.gather(...)` to execute dHash and DINO Small concurrently.

This is a pragmatic design for a small service where request handling is async but the actual image-processing work is delegated to worker threads.

## Integration Contract

### Request Contract

Frontend sends:

- `POST` requests
- `multipart/form-data`
- Fields `first` and `second`

### Response Contract

The frontend expects each algorithm response to return:

- `similarity`
- `band`
- `band_description`

Optional metrics consumed by the UI:

- `hamming_distance`
- `dhash_similarity`
- `dino_similarity`

### Error Contract

Typical backend error conditions:

- `400`: uploaded files cannot be read or decoded as images
- `500`: internal processing failure
- `503`: DINO-based functionality unavailable because model or dependency loading failed

The frontend surfaces these per algorithm, which prevents one failed algorithm from blocking the rest of the comparison results.

## Current Design Characteristics

### Strengths

- Simple and easy-to-follow separation between UI and API.
- Independent algorithm execution and error handling.
- Lightweight frontend state model using signals.
- Supports both classical and AI-based image comparison.
- Health endpoint exposes runtime model availability.

### Current Constraints

- Frontend API URL is hardcoded to `http://localhost:8000`.
- No environment-based configuration layer is present.
- Backend logic is concentrated in a single Python module.
- No persistence, user accounts, job queue, or historical comparison storage exists.
- No frontend route segmentation; the app is a single-screen workflow plus modal overlay.
- DINO models load at application startup, which increases startup cost and memory usage.

## Suggested Mental Model

The application should be understood as:

- A client-side upload and visualization shell
- A stateless comparison API
- A pluggable algorithm engine with four selectable modes

The frontend owns interaction and presentation. The backend owns decoding, feature extraction, scoring, and classification.
