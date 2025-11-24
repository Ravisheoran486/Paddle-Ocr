# PaddleOCR ONNX - AWB Number Extraction Service

Production-ready OCR service for extracting Air Waybill (AWB) numbers from images and PDFs using PaddleOCR ONNX. Designed for local testing and Google Cloud Functions deployment.

## Requirements

- macOS 14+ (Sonoma or later)
- Python 3.10 or 3.11
- Apple Silicon CPU (M1/M2/M4)

## Setup Instructions

### 1. Create Virtual Environment

Open Terminal and navigate to the project directory:

```bash
cd /Users/ravisheoran/Desktop/paddleocr
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 2. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The `paddleocr-onnx` package will automatically download the ONNX models on first use. No manual model download is required.

**For PDF Support:** If you want to process PDF files, you also need to install `poppler`:

```bash
brew install poppler
```

This is required by the `pdf2image` library to convert PDF pages to images.

### 3. Verify Installation

Test that PaddleOCR ONNX is installed correctly:

```bash
python -c "from paddleocr_onnx import PaddleOCR; print('PaddleOCR ONNX installed successfully!')"
```

## Usage

### Basic Usage

Run OCR on an image or PDF:

```bash
python main.py --file sample.jpg
python main.py --file document.pdf
```

### Command Line Options

- `--file` (required): Path to the input image or PDF file
- `--output` (optional): Path to output JSON file (default: `output.json`)
- `--gpu` (optional): Use GPU acceleration (default: False, uses CPU)

### Examples

```bash
# Process an image with default settings
python main.py --file input.jpg

# Process a PDF file
python main.py --file document.pdf

# Specify custom output file
python main.py --file sample.jpg --output results.json

# Use GPU acceleration (if available)
python main.py --file sample.pdf --gpu
```

### Supported File Formats

- **Images**: JPG, JPEG, PNG, BMP, TIFF
- **Documents**: PDF (requires poppler installation)

## Output

The script generates:

1. **Console Output**: 
   - Extracted text with confidence scores
   - Detected AWB numbers (10-16 alphanumeric characters)

2. **JSON File** (`output.json` by default):
   - Complete extracted text with bounding boxes
   - Confidence scores for each text line
   - List of detected AWB numbers
   - Full text string

### Example JSON Output

```json
{
  "image_path": "sample.jpg",
  "total_text_lines": 5,
  "extracted_text": [
    {
      "text": "Sample Text",
      "confidence": 0.9876,
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "full_text": "Sample Text Line 1 Line 2...",
  "awb_numbers": ["ABC1234567890", "XYZ9876543210"],
  "awb_count": 2
}
```

## AWB Number Detection

The script automatically extracts AWB (Air Waybill) numbers using regex pattern:
- **Pattern**: 10-16 alphanumeric characters
- **Format**: Word boundaries to avoid partial matches
- **Case**: Case-insensitive (matches both uppercase and lowercase)

## Troubleshooting

### Import Errors

If you see import errors, ensure:
1. Virtual environment is activated (`source venv/bin/activate`)
2. All dependencies are installed (`pip install -r requirements.txt`)

### Model Download Issues

On first run, `paddleocr-onnx` will download ONNX models automatically. This may take a few minutes depending on your internet connection. Models are cached locally for subsequent runs.

### Performance Tips

- For Apple Silicon, CPU mode is recommended (default)
- Processing time depends on image size and complexity
- Larger images may take longer to process

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'paddleocr_onnx'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Image not found error
- **Solution**: Check that the file path is correct and the file exists

**Issue**: PDF processing fails with poppler error
- **Solution**: Install poppler: `brew install poppler`

**Issue**: Slow processing
- **Solution**: This is normal for CPU processing. Consider resizing very large images before processing.

## Deactivating Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
paddleocr/
├── main.py                 # Local testing script (CLI)
├── cloud_function.py       # Google Cloud Functions handler
├── main_cloud.py          # HTTP trigger entry point for Cloud Functions
├── requirements.txt       # Dependencies for local development
├── requirements_cloud.txt # Dependencies for Cloud Functions
├── README.md              # This file
├── SETUP.md               # Quick setup guide
└── DEPLOYMENT.md          # Cloud Functions deployment guide
```

## Code Architecture

The codebase is organized into modular classes:

- **`AWBExtractor`**: Handles AWB number extraction using regex patterns
- **`OCRProcessor`**: Manages PaddleOCR ONNX initialization and image processing
- **`AWBOCRService`**: Main service class that orchestrates OCR and AWB extraction

This structure makes the code:
- Easy to test and maintain
- Ready for Cloud Functions deployment
- Reusable across different contexts

## Google Cloud Functions Deployment

This code is designed to be deployed to Google Cloud Functions with two trigger options:

1. **Cloud Storage Trigger**: Automatically processes images when uploaded to a GCS bucket
2. **HTTP Trigger**: Accepts HTTP POST requests with image data

See `DEPLOYMENT.md` for detailed deployment instructions.

### Key Features for Cloud Functions

- Automatic processing on new object creation
- Webhook integration for sending results
- Error handling and logging
- Support for both direct uploads and GCS references

## Notes

- The ONNX models are automatically downloaded and cached on first use
- No manual model download or conversion is required
- Works entirely on CPU (no GPU required for Apple Silicon)
- Compatible with Python 3.10 and 3.11 on macOS
- Production-ready code structure with proper logging and error handling

## License

This setup uses PaddleOCR ONNX, which is open-source software. Please refer to the original PaddleOCR license for details.

