## Goals
- Correctly read text from rotated/scanned pages in `Jio.pdf` without changing output schema
- Minimize runtime impact and avoid modifying unrelated code paths

## Current Behavior
- Angle classification disabled at both init and call:
  - `main.py:160` uses `use_angle_cls=False`
  - `main.py:109` calls `ocr.ocr(img, cls=False)`
- PDF pages rasterized via `pdf2image` (`main.py:83`)

## Approach Overview
1. Enable line-level rotation handling using PaddleOCR angle classifier
2. Add page-level orientation detection with multi-rotation fallback (0/90/180/270)
3. Add small-angle deskew for skewed scans (≤15°)
4. Keep outputs unchanged; only add optional metadata about applied rotations

## Step 1: Angle Classification (Low Cost, High Gain)
- Set `use_angle_cls=True` at init (`main.py:160`)
- Call OCR with `cls=True` (`main.py:109`) to auto-correct per-line rotation
- Benefit: Handles mixed rotations within the same page without extra passes

## Step 2: Page-Level Orientation Selection
- If initial OCR score is poor, try rotated copies and choose the best:
  - Rotations: `0, 90, 180, 270`
  - Score function: weighted sum of `total_text_lines`, average confidence, and AWB presence count
  - Threshold: Only trigger multi-rotation when `(lines < L_min OR avg_conf < C_min)` to cap runtime
- Return results from the chosen orientation; keep schema identical

## Step 3: Small-Angle Deskew
- For skewed pages (not exactly 90/180), estimate skew angle:
  - Compute edges → Hough lines or minimal-area rectangle of text mask
  - If `|angle| ≤ 15°`, rotate via `cv2.getRotationMatrix2D` before OCR
- Trigger only when multi-rotation did not significantly improve score

## Step 4: PDF-Specific Considerations
- `pdf2image` generally respects page rotation, but embedded scans may still be rotated
- Apply Step 2/3 to each page independently after rasterization

## Configuration & Safety
- New flags (default ON): `--auto_rotate`, `--allow_deskew`
- Tunables: `--rotation_thresholds L_min,C_min`, `--max_rotations`
- Preserve current defaults if flags are off to avoid behavior changes

## Output & Metadata
- Keep existing fields unchanged
- Optionally add: `orientation_applied` (e.g., `{rotation_deg: 90, skew_deg: -7}`) and `preprocessing_steps` for debugging

## Validation Plan
- Baseline: Run on `Jio.pdf` with current settings; capture `total_text_lines`, avg confidence, AWB hits
- After Step 1: Expect increase in lines/quality on rotated lines
- After Step 2: Expect substantial improvement for fully rotated pages (90/180)
- After Step 3: Verify gains on slightly skewed scans
- Compare AWB extraction before/after; confirm higher `awb_count`

## Performance & Limits
- Step 1: negligible overhead
- Step 2: up to 4× runtime in worst case; gated by thresholds
- Step 3: modest CPU cost; only runs when needed

## Cloud Function Impact
- Same pipeline applies server-side
- Ensure timeouts accommodate multi-rotation; expose `auto_rotate` as a query/body toggle
- PDF note: Poppler required for PDF rasterization; images will work regardless

## Rollback
- Disable via flags to revert to current behavior quickly