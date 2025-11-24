#!/usr/bin/env python3
"""
PaddleOCR ONNX - Local OCR Script for macOS
Extracts text from images/PDFs and identifies AWB numbers (10-16 alphanumeric characters)
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

try:
    from paddleocr_onnx import PaddleOCR
    import cv2
    import numpy as np
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print(f"  pip install -r requirements.txt")
    print(f"\nNote: For PDF support, you may also need to install poppler:")
    print(f"  brew install poppler  # on macOS")
    print(f"\nOriginal error: {e}")
    sys.exit(1)


def extract_awb_numbers(text: str, return_rejected: bool = False) -> Union[List[str], Dict[str, List[str]]]:
    pattern = r"\b(?:SF|R|RT)[\s\-\/]*[A-Z0-9](?:[\s\-\/]*[A-Z0-9]){7,24}(?:[\s\-\/]*[A-Z]{0,4})?\b|\bSF[\s\-\/]*[A-Z0-9](?:[\s\-\/]*[A-Z0-9]){7}[\s\-\/]*R\b"
    matches = re.findall(pattern, text)
    def normalize_and_validate(t: str):
        t = re.sub(r"[\s\-\/:;\.|]+", "", t).upper()
        
        # Fix common OCR errors in prefix
        if t.startswith("5F"): t = "SF" + t[2:]
        
        if t.startswith("SF") and t.endswith("R") and len(t) >= 11:
            core = t[2:-1]
            map_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'Q': '9', 'D': '0'}
            fixed = ''.join(map_table.get(ch, ch) for ch in core)
            if re.fullmatch(r"\d{8}", fixed):
                return "SF" + fixed + "R"
            return None
            
        m = re.match(r"^(SF|R|RT)([A-Z0-9]{6,24})([A-Z]{0,4})$", t)
        if not m:
            return None
        prefix, mid, suffix = m.groups()
        
        # If suffix is empty but mid ends with letters, try to shift them to suffix
        if not suffix and mid:
            last_digit_idx = -1
            for i in range(len(mid) - 1, -1, -1):
                if mid[i].isdigit():
                    last_digit_idx = i
                    break
            
            if last_digit_idx < len(mid) - 1:
                potential_suffix = mid[last_digit_idx+1:]
                if len(potential_suffix) <= 4 and potential_suffix.isalpha():
                    mid = mid[:last_digit_idx+1]
                    suffix = potential_suffix

        map_table = {'O': '0', 'I': '1', 'L': '1', '|': '1', '!': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'Q': '9', 'D': '0'}
        fixed_mid = ''.join(map_table.get(ch, ch) for ch in mid)
        
        # Allow minor non-digit intrusions if length is sufficient
        digit_count = sum(c.isdigit() for c in fixed_mid)
        if len(fixed_mid) - digit_count > 2:
             if not re.fullmatch(r"\d{8,21}", fixed_mid):
                return None

        # Force fix remaining chars if mostly digits
        final_mid = ""
        for ch in fixed_mid:
            if ch.isdigit():
                final_mid += ch
            elif ch in map_table:
                final_mid += map_table[ch]
            else:
                final_mid += ch
        
        if not re.fullmatch(r"\d{8,21}", final_mid):
            return None
            
        if suffix:
            suffix = suffix.replace('1', 'I').replace('|', 'I').replace('!', 'I')
            if not re.fullmatch(r"[A-Z]{1,4}", suffix):
                return None
        return prefix + final_mid + (suffix or "")
        
    seen = set()
    out = []
    rejected = []
    for m in matches:
        val = normalize_and_validate(m)
        if val and val not in seen:
            seen.add(val)
            out.append(val)
        elif not val:
            rejected.append(m)
            
    if return_rejected:
        return {'accepted': out, 'rejected': rejected}
    return out


def join_rows(extracted_data: List[Dict[str, Any]]) -> str:
    """
    Join text from extracted data by clustering boxes into rows.
    This helps reconstruct tokens split across multiple boxes (e.g. SF prefix and digits).
    """
    if not extracted_data:
        return ""
        
    # Sort by top-left Y coordinate
    sorted_data = sorted(extracted_data, key=lambda x: x['bbox'][0][1])
    
    rows = []
    current_row = []
    if sorted_data:
        current_y = sorted_data[0]['bbox'][0][1]
        row_threshold = 10 # pixels
        
        for item in sorted_data:
            y = item['bbox'][0][1]
            if abs(y - current_y) > row_threshold:
                # New row
                current_row.sort(key=lambda x: x['bbox'][0][0])
                rows.append(current_row)
                current_row = [item]
                current_y = y
            else:
                current_row.append(item)
        
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0][0])
            rows.append(current_row)
    
    full_text_parts = []
    for row in rows:
        # Join without spaces to merge split tokens, but add space between rows
        # Actually, we might want to be careful about merging distinct words.
        # But for AWB extraction, merging is better.
        # Let's try to be smart: if boxes are close, merge. If far, space.
        # For now, simple merge as requested.
        row_text = "".join([item['text'] for item in row])
        full_text_parts.append(row_text)
        
    return " ".join(full_text_parts)


def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF based on extension."""
    return Path(file_path).suffix.lower() == '.pdf'


def pdf_to_images(pdf_path: str, dpi: int = 450) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of PIL Image objects (one per page)
    """
    try:
        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} page(s) to images")
        return images
    except Exception as e:
        raise ValueError(f"Failed to convert PDF to images: {e}. Make sure poppler is installed (brew install poppler)")


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (numpy array)."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def process_image_with_ocr(ocr: PaddleOCR, img: np.ndarray, page_num: int = None) -> tuple:
    """
    Process a single image with OCR.
    
    Args:
        ocr: Initialized PaddleOCR instance
        img: Image as numpy array (OpenCV format)
        page_num: Optional page number for PDFs
        
    Returns:
        Tuple of (extracted_data_list, full_text_string)
    """
    # Perform OCR
    try:
        results = ocr.ocr(img, cls=True)
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        raise
    
    # Extract text and confidence scores
    extracted_data = []
    full_text_parts = []
    
    if results and len(results) > 0:
        for line in results[0]:
            if line and len(line) >= 2:
                # Line format: [[bbox], (text, confidence)]
                bbox = line[0]
                text_info = line[1]
                
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = float(text_info[1])
                    
                    extracted_data.append({
                        'text': text,
                        'confidence': round(confidence, 4),
                        'bbox': [[float(coord[0]), float(coord[1])] for coord in bbox],
                        'page': page_num
                    })
                    full_text_parts.append(text)
    
    # Use row-wise joining to reconstruct split tokens
    full_text = join_rows(extracted_data)
    return extracted_data, full_text

def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def score_result(extracted_data: List[Dict[str, Any]], full_text: str):
    lines = len(extracted_data)
    avg_conf = sum(d.get('confidence', 0.0) for d in extracted_data) / lines if lines > 0 else 0.0
    awb = len(extract_awb_numbers(full_text))
    score = lines + avg_conf * 10.0 + awb * 15.0
    return score, avg_conf, lines, awb

def to_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def generate_preprocess_variants(img: np.ndarray, scale: float = 1.5) -> List[np.ndarray]:
    variants = []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    v1 = to_bgr(clahe)
    variants.append(v1)
    den = cv2.bilateralFilter(clahe, 9, 75, 75)
    v2 = to_bgr(den)
    variants.append(v2)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    v3 = to_bgr(sharp)
    variants.append(v3)
    thr = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    v4 = to_bgr(thr)
    variants.append(v4)
    h, w = img.shape[:2]
    v5 = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    variants.append(v5)
    return variants

def process_with_orientation(ocr: PaddleOCR, img: np.ndarray, page_num: int = None, scale: float = 1.5):
    best_score = -1.0
    best_extracted, best_text = [], ""
    # Base image + variants
    base_e, base_t = process_image_with_ocr(ocr, img, page_num)
    base_s, _, _, _ = score_result(base_e, base_t)
    best_score, best_extracted, best_text = base_s, base_e, base_t
    for v in generate_preprocess_variants(img, scale=scale):
        e1, t1 = process_image_with_ocr(ocr, v, page_num)
        s1, _, _, _ = score_result(e1, t1)
        if s1 > best_score:
            best_score, best_extracted, best_text = s1, e1, t1
    # Rotations + variants
    for angle in [180]:
        rotated = rotate_image(img, angle)
        e2, t2 = process_image_with_ocr(ocr, rotated, page_num)
        s2, _, _, _ = score_result(e2, t2)
        if s2 > best_score:
            best_score, best_extracted, best_text = s2, e2, t2
        for v in generate_preprocess_variants(rotated, scale=scale):
            e3, t3 = process_image_with_ocr(ocr, v, page_num)
            s3, _, _, _ = score_result(e3, t3)
            if s3 > best_score:
                best_score, best_extracted, best_text = s3, e3, t3
    return best_extracted, best_text

def crop_awb_column(img: np.ndarray, extracted: List[Dict[str, Any]]) -> Union[np.ndarray, None]:
    awb_boxes = [e['bbox'] for e in extracted if isinstance(e.get('text'), str) and ('awb' in e['text'].lower())]
    if not awb_boxes:
        return None
    xs = []
    for box in awb_boxes:
        for pt in box:
            xs.append(pt[0])
    if not xs:
        return None
    x_min = max(0, int(min(xs)))
    x_max = int(max(xs))
    h, w = img.shape[:2]
    margin = int(0.05 * w)
    x0 = max(0, x_min - margin)
    x1 = min(w, x_max + margin)
    if x1 - x0 < 20:
        return None
    return img[:, x0:x1]

def crop_awb_column_by_histogram(img: np.ndarray, extracted: List[Dict[str, Any]]) -> Union[np.ndarray, None]:
    h, w = img.shape[:2]
    bins = 8
    counts = [0]*bins
    xs = [(w*i)//bins for i in range(bins+1)]
    for e in extracted:
        t = e.get('text')
        if not isinstance(t, str):
            continue
        # Prefer tokens that look like AWB patterns
        tt = re.sub(r"[\s\-/]+", "", t).upper()
        if re.match(r"^(SF|R)[A-Z0-9]{6,}$", tt):
            box = e.get('bbox')
            if not box:
                continue
            cx = sum(p[0] for p in box)/len(box)
            idx = min(bins-1, max(0, int(cx/(w/bins))))
            counts[idx] += 1
    if max(counts) == 0:
        return None
    idx = int(np.argmax(np.array(counts)))
    # Expand the crop window: idx-2 to idx+4 (wider context)
    x0 = xs[max(0, idx-2)]
    x1 = xs[min(bins, idx+4)]
    return img[:, x0:x1]


def run_ocr(file_path: str, use_gpu: bool = False, dpi: int = 700, upscale: float = 2.5) -> Dict[str, Any]:
    """
    Run PaddleOCR ONNX on an image or PDF and extract text with confidence scores.
    
    Args:
        file_path: Path to input image or PDF file
        use_gpu: Whether to use GPU (default: False for Apple Silicon CPU)
        
    Returns:
        Dictionary containing extracted data and AWB numbers
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Initialize PaddleOCR with ONNX backend
    # For macOS Apple Silicon, use CPU (ONNX Runtime supports Metal acceleration)
    print("Initializing PaddleOCR ONNX...")
    ocr = PaddleOCR(
        use_angle_cls=True,  # Disable angle classification for faster processing
        lang='en',            # English language
        use_gpu=use_gpu,      # Use CPU on Apple Silicon
        show_log=False        # Suppress verbose logging
    )
    
    all_extracted_data = []
    all_full_text_parts = []
    total_pages = 1
    
    # Handle PDF files
    if is_pdf(file_path):
        print(f"Processing PDF: {file_path}")
        pdf_images = pdf_to_images(file_path, dpi=dpi)
        total_pages = len(pdf_images)
        
        for page_num, pil_image in enumerate(pdf_images, 1):
            print(f"Processing page {page_num}/{total_pages}...")
            cv2_image = pil_to_cv2(pil_image)
            extracted_data, full_text = process_with_orientation(ocr, cv2_image, page_num=page_num, scale=upscale)
            all_extracted_data.extend(extracted_data)
            all_full_text_parts.append(full_text)
            col = crop_awb_column(cv2_image, extracted_data)
            if col is not None:
                e_col, t_col = process_with_orientation(ocr, col, page_num=page_num, scale=upscale)
                all_extracted_data.extend(e_col)
                all_full_text_parts.append(t_col)
            else:
                col2 = crop_awb_column_by_histogram(cv2_image, extracted_data)
                if col2 is not None:
                    e2, t2 = process_with_orientation(ocr, col2, page_num=page_num, scale=upscale)
                    all_extracted_data.extend(e2)
                    all_full_text_parts.append(t2)
            print(f"  Page {page_num}: Extracted {len(extracted_data)} text line(s)")
    
    # Handle image files
    else:
        print(f"Processing image: {file_path}")
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Unable to read image from: {file_path}. Please check the file path and format.")
        if upscale and upscale > 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*upscale), int(h*upscale)), interpolation=cv2.INTER_CUBIC)
        extracted_data, full_text = process_with_orientation(ocr, img, scale=upscale)
        all_extracted_data.extend(extracted_data)
        all_full_text_parts.append(full_text)
        col = crop_awb_column(img, extracted_data)
        if col is not None:
            e_col, t_col = process_with_orientation(ocr, col, scale=upscale)
            all_extracted_data.extend(e_col)
            all_full_text_parts.append(t_col)
        else:
            col2 = crop_awb_column_by_histogram(img, extracted_data)
            if col2 is not None:
                e2, t2 = process_with_orientation(ocr, col2, scale=upscale)
                all_extracted_data.extend(e2)
                all_full_text_parts.append(t2)
    
    # Combine all text
    full_text = " ".join(all_full_text_parts)
    
    # Extract AWB numbers from combined text
    awb_data = extract_awb_numbers(full_text, return_rejected=True)
    awb_numbers = awb_data['accepted']
    rejected_awbs = awb_data['rejected']
    
    # Prepare output
    output = {
        'file_path': file_path,
        'file_type': 'PDF' if is_pdf(file_path) else 'Image',
        'total_pages': total_pages,
        'total_text_lines': len(all_extracted_data),
        'extracted_text': all_extracted_data,
        'full_text': full_text.strip(),
        'awb_numbers': awb_numbers,
        'awb_count': len(awb_numbers),
        'rejected_candidates': rejected_awbs
    }
    
    return output


def save_results(output: Dict[str, Any], output_path: str = "output.json"):
    """Save extracted data to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def print_results(output: Dict[str, Any]):
    """Print extracted text and AWB numbers to console."""
    print("\n" + "="*60)
    print("OCR RESULTS")
    print("="*60)
    
    print(f"\nFile: {output['file_path']}")
    print(f"Type: {output['file_type']}")
    if output['file_type'] == 'PDF':
        print(f"Pages: {output['total_pages']}")
    print(f"Total text lines detected: {output['total_text_lines']}")
    
    if output['extracted_text']:
        print("\nExtracted Text (with confidence scores):")
        print("-" * 60)
        for i, item in enumerate(output['extracted_text'], 1):
            page_info = f" [Page {item.get('page', 1)}]" if item.get('page') else ""
            print(f"{i}. [{item['confidence']:.4f}]{page_info} {item['text']}")
    
    if output['awb_numbers']:
        print(f"\nAWB Numbers Found ({output['awb_count']}):")
        print("-" * 60)
        for awb in output['awb_numbers']:
            print(f"  • {awb}")
    else:
        print("\nNo AWB numbers found (matching AWB pattern)")
        
    if output.get('rejected_candidates'):
        print(f"\nRejected Candidates ({len(output['rejected_candidates'])}):")
        print("-" * 60)
        for rej in output['rejected_candidates']:
            print(f"  • {rej} (Failed validation)")
    
    print("\n" + "="*60)


def main():
    """Main function to run OCR on input image or PDF."""
    parser = argparse.ArgumentParser(
        description='Run PaddleOCR ONNX on an image or PDF and extract AWB numbers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file sample.jpg
  python main.py --file document.pdf
  python main.py --file input.jpg --output results.json
  python main.py --file sample.pdf --gpu

Supported formats: JPG, JPEG, PNG, BMP, TIFF, PDF
        """
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to the input image or PDF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Path to output JSON file (default: output.json)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (default: False, uses CPU)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=700,
        help='PDF rasterization DPI (higher improves OCR on scans)'
    )
    parser.add_argument(
        '--upscale',
        type=float,
        default=2.5,
        help='Image upscale factor before OCR (>=1.0)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run OCR
        output = run_ocr(args.file, use_gpu=args.gpu, dpi=args.dpi, upscale=args.upscale)
        
        # Print results to console
        print_results(output)
        
        # Save results to JSON
        save_results(output, args.output)
        
        print("\n✓ OCR processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
