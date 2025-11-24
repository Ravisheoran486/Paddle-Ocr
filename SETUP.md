# Quick Setup Guide

## Step 1: Create and Activate Virtual Environment

```bash
cd /Users/ravisheoran/Desktop/paddleocr
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For PDF Support (optional):**
```bash
brew install poppler
```

## Step 3: Test Installation

```bash
python -c "from paddleocr_onnx import PaddleOCR; print('✓ Installation successful!')"
```

## Step 4: Run OCR on an Image or PDF

```bash
# Process an image
python main.py --file sample.jpg

# Process a PDF
python main.py --file document.pdf
```

Replace `sample.jpg` or `document.pdf` with your actual file path.

## Notes

- Models are automatically downloaded on first run (may take a few minutes)
- No manual model download required
- Works on CPU (Apple Silicon optimized)
- Python 3.10 or 3.11 recommended
- Supports images (JPG, PNG, etc.) and PDF files

## Troubleshooting

If you encounter any issues:

1. **Ensure virtual environment is activated**: You should see `(venv)` in your terminal
2. **Check Python version**: `python3 --version` should show 3.10 or 3.11
3. **Reinstall dependencies**: `pip install --force-reinstall -r requirements.txt`
4. **PDF errors**: If PDF processing fails, install poppler: `brew install poppler`

