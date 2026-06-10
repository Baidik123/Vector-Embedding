# Feature Overview

## Product Purpose

This application compares two PNG images and helps users judge whether they are duplicates, near-duplicates, or visually related drawings. It combines interactive visual inspection in the browser with backend-generated similarity scores from multiple algorithms.

## User-Facing Features

### 1. Dual PNG Upload

Users can provide two files for comparison through either:

- click-to-upload file selection
- drag-and-drop upload zones

The upload experience is symmetrical for both inputs:

- First PNG
- Second PNG

Each side maintains its own file state, validation state, and preview.

### 2. PNG Validation

The frontend only accepts PNG uploads.

Validation behavior:

- Checks MIME type and filename extension
- Rejects non-PNG files immediately
- Shows a clear inline error message in the affected upload zone
- Prevents invalid files from participating in comparison

### 3. Instant Image Preview

After a valid upload:

- the image is previewed directly in the upload card
- the filename is displayed
- file size is formatted for readability
- users can clear the uploaded file with a dedicated remove button

The preview uses browser object URLs and cleans them up when files are replaced or when the component is destroyed.

### 4. Multi-Algorithm Selection

Users can select one or more comparison methods before running analysis.

Available options:

- `dHash`
- `DINO V2 Small`
- `DINO V2 Large`
- `Hybrid`

Selection behavior:

- Algorithms are selectable using checkboxes
- Multiple algorithms can be run in one comparison session
- The compare button stays disabled until both PNG files are available and at least one algorithm is selected

### 5. Visual Comparison Overlay

After clicking `Compare PNGs`, the app opens a full-screen comparison overlay.

This overlay includes:

- a layered image comparison viewer
- a draggable slider to reveal more or less of the second image
- zoom out control
- zoom in control
- zoom reset control
- close button to return to the upload screen

This lets users visually inspect structural or rendering differences alongside the algorithmic output.

### 6. Per-Algorithm Result Cards

Each selected algorithm produces its own result card in the lower results panel.

Each card can independently show:

- loading state
- error state
- similarity score
- qualitative result band
- short explanation of the band
- algorithm-specific supporting metrics

The UI does not wait for all algorithms to finish before showing feedback. Results appear as each request completes.

## Implemented Comparison Features

### 1. dHash Comparison

Purpose:

- fast perceptual comparison using a compact pixel fingerprint

Behavior:

- backend preprocesses each image before hashing
- computes 16x16 dHash
- calculates Hamming distance out of 256 bits
- converts distance into a normalized similarity score
- classifies the result into a qualitative band

Returned metrics:

- similarity
- band
- band description
- Hamming distance

### 2. DINO V2 Small Comparison

Purpose:

- AI-based semantic image similarity using a smaller embedding model

Behavior:

- generates 384-dimensional embeddings
- compares embeddings using nearest-neighbor distance after normalization
- returns a normalized display score
- classifies the result into a qualitative band

Usage characteristics:

- more semantically aware than dHash
- depends on optional AI libraries and model availability

### 3. DINO V2 Large Comparison

Purpose:

- higher-capacity AI comparison using a larger embedding model

Behavior:

- generates 1024-dimensional embeddings
- follows the same normalized scoring flow as DINO Small
- returns qualitative classification and similarity score

Usage characteristics:

- potentially richer representation
- heavier model footprint and runtime cost
- may be unavailable if the model fails to load or if memory is insufficient

### 4. Hybrid Comparison

Purpose:

- combine classical image fingerprinting with AI embedding similarity

Behavior:

- runs dHash and DINO Small concurrently
- combines both using weighted scoring
- uses dHash contribution of `0.4`
- uses DINO Small contribution of `0.6`
- returns the final hybrid score plus both component scores

Returned metrics:

- overall similarity
- dHash component similarity
- DINO component similarity
- Hamming distance
- band
- band description

This feature gives users a more balanced signal than relying on either a purely pixel-based or purely embedding-based method alone.

## Classification Features

All algorithms map numeric scores into human-readable categories so users do not need to interpret raw similarity values alone.

Available qualitative outputs:

- `Exact Duplicate`
- `Likely Duplicate`
- `Similar – Same Family`
- `Similar – Related`
- `Different`

Each category is returned with an explanatory sentence and rendered in the UI with color-coded styling.

## Backend Service Features

### 1. Dedicated API Endpoints

The backend exposes separate endpoints for each algorithm:

- `/compare/dhash`
- `/compare/dino`
- `/compare/dino-large`
- `/compare/hybrid`

Additional service endpoints:

- `/compare` for legacy compatibility
- `/health` for runtime availability and status checks

### 2. Health Monitoring

The health endpoint reports:

- service status
- backend version
- DINO Small availability
- DINO Large availability

This is useful for diagnosing whether optional AI models are ready before users invoke those algorithms.

### 3. Robust Image Loading

Before any comparison, the backend:

- reads uploaded files
- decodes them with Pillow
- converts them to RGB images
- returns a clear client error if reading or decoding fails

### 4. Non-Blocking Execution Pattern

The backend uses executor threads for heavy comparison logic so the FastAPI event loop can stay responsive while image hashing or AI inference is running.

### 5. Partial Availability Support

The system is built so that:

- dHash can work even when DINO dependencies are missing
- DINO Small and DINO Large can fail independently
- Hybrid is only enabled on the backend when DINO Small is available

This makes the backend more resilient than an all-or-nothing model-loading design.

## Frontend Experience Features

### 1. Independent Upload State

Each upload side separately tracks:

- selected file
- preview URL
- validation error
- drag-over state

This keeps the UI predictable and prevents one file interaction from disturbing the other.

### 2. Independent Algorithm State

Each selected algorithm tracks:

- label
- loading state
- error message
- similarity value
- classification band
- auxiliary metrics

This means one algorithm can show a successful result even if another returns an error.

### 3. Responsive Layout

The upload screen uses a two-column layout on wider screens and collapses to a single-column layout on smaller screens.

### 4. Minimal Flow Complexity

The product currently uses a single-screen interaction model:

- upload files
- select methods
- compare in overlay
- review results

There are no login steps, project setup flows, or multi-page navigation requirements.

## Operational Characteristics

### Current Assumptions

- Frontend and backend run as separate local services
- Backend is expected at `http://localhost:8000`
- Users compare PNG files only
- Results are computed on demand and not stored

### Current Limits

- No PDF upload support in the current implementation
- No persisted comparison history
- No batch processing for more than two files
- No authentication or role-based access
- No configurable backend endpoint from the UI

## Summary

The implemented feature set centers on one core workflow:

- upload two PNG images
- choose one or more similarity algorithms
- inspect differences visually
- review quantitative and qualitative comparison results

The strongest value of the product comes from combining interactive visual comparison with multiple backend scoring strategies in a single, lightweight interface.
