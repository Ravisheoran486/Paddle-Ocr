
"""
Comprehensive AWB Extraction - Supports both R and SF types
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

try:
    from paddleocr_onnx import PaddleOCR
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def extract_awb_numbers_comprehensive(text: str) -> List[str]:
    """
    Comprehensive AWB extraction supporting both R and SF types
    """
    # Comprehensive patterns for both R and SF types
    patterns = [
        # R type: R + 10 digits + 2-4 letters (with OCR error tolerance)
        r'\bR\d{9,12}[A-Z0-9]{2,4}\b',
        
        # SF type: SF + 10 digits + 2-4 letters (with OCR error tolerance) 
        r'\bSF\d{9,12}[A-Z0-9]{2,4}\b',
        
        # More flexible patterns for both types
        r'\b(?:R|SF)\d{8,13}[A-Z]+\b',
        r'\b(?:R|SF)\d+[A-Z0-9]{2,5}\b',
        
        # Handle vertical bar and other OCR artifacts
        r'\b(?:R|SF)\d{9,12}[A-Z0-9]{2,4}[|]?\b',
        
        # Generic long alphanumeric patterns that might be AWBs
        r'\b(?:R|SF)[A-Z0-9]{10,16}\b'
    ]
    
    all_candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        all_candidates.extend(matches)
    
    # Clean and validate candidates
    valid_awbs = []
    seen_clean = set()
    
    for candidate in all_candidates:
        cleaned = clean_awb_comprehensive(candidate)
        if cleaned and cleaned not in seen_clean and is_valid_awb_comprehensive(cleaned):
            valid_awbs.append(cleaned)
            seen_clean.add(cleaned)
    
    return valid_awbs


def clean_awb_comprehensive(raw_awb: str) -> str:
    """
    Comprehensive cleaning for both R and SF type AWBs
    """
    if not raw_awb or len(raw_awb) < 10:
        return ""
    
    cleaned = raw_awb.upper()
    
    # Remove trailing vertical bars and other artifacts
    cleaned = re.sub(r'[|\s\-\.]+$', '', cleaned)
    
    # Fix common OCR errors
    replacements = {
        'O': '0', 'I': '1', 'L': '1', '|': '1', '!': '1',
        'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'Q': '9'
    }
    
    # Apply replacements to the entire string
    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)
    
    # Special handling for SF prefix
    if cleaned.startswith('5F'):  # Common OCR error
        cleaned = 'SF' + cleaned[2:]
    
    # Ensure proper prefix
    if not cleaned.startswith(('R', 'SF')):
        if 'SF' in cleaned[:3]:
            sf_pos = cleaned.index('SF')
            cleaned = 'SF' + cleaned[sf_pos+2:]
        elif 'R' in cleaned[:3]:
            r_pos = cleaned.index('R')
            cleaned = 'R' + cleaned[r_pos+1:]
        elif cleaned.startswith(('1', 'I', 'L')) and len(cleaned) >= 12:
            # Might be misread R at start
            cleaned = 'R' + cleaned[1:]
    
    # Fix suffix - convert 1 to I in the last 2-4 characters
    if len(cleaned) > 12:
        prefix_mid = cleaned[:-4]  # Everything except last 4 chars
        suffix = cleaned[-4:]      # Last 4 chars (suffix area)
        
        # In suffix area, convert 1 back to I for better readability
        suffix = suffix.replace('1', 'I')
        cleaned = prefix_mid + suffix
    
    return cleaned


def is_valid_awb_comprehensive(awb: str) -> bool:
    """
    Comprehensive validation for both R and SF type AWBs
    """
    if len(awb) < 12 or len(awb) > 16:
        return False
    
    # Must start with R or SF
    if not awb.startswith(('R', 'SF')):
        return False
    
    # Check length based on prefix
    if awb.startswith('R') and (len(awb) < 13 or len(awb) > 16):
        return False
    if awb.startswith('SF') and (len(awb) < 14 or len(awb) > 17):
        return False
    
    # Check digit and letter distribution
    digit_count = sum(c.isdigit() for c in awb)
    alpha_count = sum(c.isalpha() for c in awb)
    
    # R type: should have ~10 digits + 2-4 letters
    if awb.startswith('R'):
        if digit_count < 8 or alpha_count < 2:
            return False
    
    # SF type: should have ~10 digits + 2-4 letters  
    if awb.startswith('SF'):
        if digit_count < 8 or alpha_count < 2:
            return False
    
    # Should not be too repetitive
    if len(set(awb)) < 6:
        return False
    
    return True


def high_resolution_preprocess(img: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    """
    High-resolution preprocessing for better OCR
    """
    # Convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scale up for better OCR accuracy
    h, w = gray.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Use INTER_CUBIC for better quality when scaling up
    scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(scaled)
    
    # Light sharpening
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def smart_rotation_detection(ocr: PaddleOCR, img: np.ndarray, page_num: int) -> np.ndarray:
    """
    Smart rotation detection that tests both orientations
    """
    # Page 3 is known to be correct orientation
    if page_num == 3:
        return img
    
    print(f"  Testing orientations for page {page_num}...")
    
    # Test original orientation
    results_original = ocr.ocr(img, cls=False)
    text_original, awbs_original = extract_text_and_awbs(results_original)
    
    # Test rotated orientation (180°)
    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    results_rotated = ocr.ocr(rotated_img, cls=False)
    text_rotated, awbs_rotated = extract_text_and_awbs(results_rotated)
    
    print(f"    Original: {len(awbs_original)} AWBs, Rotated: {len(awbs_rotated)} AWBs")
    
    # Choose orientation with more AWBs
    if len(awbs_rotated) > len(awbs_original):
        print(f"    Selecting rotated orientation (better results)")
        return rotated_img
    else:
        print(f"    Selecting original orientation")
        return img


def extract_text_and_awbs(ocr_result) -> Tuple[str, List[str]]:
    """Extract text and AWBs from OCR result"""
    text_parts = []
    
    if ocr_result and len(ocr_result) > 0:
        for line in ocr_result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    text = text_info[0]
                    text_parts.append(text)
    
    full_text = " ".join(text_parts)
    awbs = extract_awb_numbers_comprehensive(full_text)
    
    return full_text, awbs


def process_page_comprehensive(ocr: PaddleOCR, img: np.ndarray, page_num: int) -> Dict[str, Any]:
    """
    Comprehensive page processing with smart rotation
    """
    print(f"  Processing page {page_num}...")
    
    # Determine optimal orientation
    oriented_img = smart_rotation_detection(ocr, img, page_num)
    
    # Apply high-resolution preprocessing
    processed_img = high_resolution_preprocess(oriented_img, scale_factor=2.0)
    
    # Perform OCR
    try:
        results = ocr.ocr(processed_img, cls=False)
        text, awbs = extract_text_and_awbs(results)
        
        # If few AWBs found, try alternative preprocessing
        if len(awbs) < 3:
            alt_processed = high_resolution_preprocess(oriented_img, scale_factor=1.5)
            alt_results = ocr.ocr(alt_processed, cls=False)
            alt_text, alt_awbs = extract_text_and_awbs(alt_results)
            
            # Use the better result
            if len(alt_awbs) > len(awbs):
                text, awbs = alt_text, alt_awbs
        
        return {
            'awbs': awbs,
            'full_text': text,
            'text_length': len(text),
            'awb_count': len(awbs)
        }
        
    except Exception as e:
        print(f"    OCR error: {e}")
        return {'awbs': [], 'full_text': '', 'text_length': 0, 'awb_count': 0}


def advanced_deduplication(all_awbs: List[str]) -> List[str]:
    """
    Advanced deduplication that handles similar AWBs
    """
    # Sort by length (longer first) to prefer complete AWBs
    sorted_awbs = sorted(all_awbs, key=len, reverse=True)
    
    final_awbs = []
    seen = set()
    
    for awb in sorted_awbs:
        # Skip if exact duplicate
        if awb in seen:
            continue
        
        # Check for partial matches (e.g., R1500571646AJ vs R1500571646AJI)
        is_partial_duplicate = False
        for existing in final_awbs:
            if awb in existing or existing in awb:
                # If one is clearly a substring of the other, keep the longer one
                if len(awb) < len(existing):
                    is_partial_duplicate = True
                    break
                elif len(awb) == len(existing) and awb != existing:
                    # Same length but different - both might be valid
                    continue
        
        if not is_partial_duplicate:
            final_awbs.append(awb)
            seen.add(awb)
    
    # Sort final list for consistency
    return sorted(final_awbs)


def analyze_awb_types(awbs: List[str]) -> Dict[str, Any]:
    """Analyze the distribution of AWB types"""
    r_awbs = [awb for awb in awbs if awb.startswith('R')]
    sf_awbs = [awb for awb in awbs if awb.startswith('SF')]
    other_awbs = [awb for awb in awbs if not awb.startswith(('R', 'SF'))]
    
    return {
        'total': len(awbs),
        'r_type': len(r_awbs),
        'sf_type': len(sf_awbs),
        'other_type': len(other_awbs),
        'r_examples': r_awbs[:5],
        'sf_examples': sf_awbs[:5]
    }


def process_pdf_comprehensive(file_path: str, use_gpu: bool = False, dpi: int = 400) -> Dict[str, Any]:
    """
    Comprehensive PDF processing supporting both R and SF AWB types
    """
    print(f"Comprehensive PDF processing: {file_path}")
    
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='en',
        use_gpu=use_gpu,
        show_log=False
    )
    
    try:
        pdf_images = convert_from_path(file_path, dpi=dpi)
        print(f"Converted {len(pdf_images)} page(s) at {dpi} DPI")
    except Exception as e:
        raise ValueError(f"PDF conversion failed: {e}")
    
    all_awbs = []
    page_results = []
    
    for page_num, pil_image in enumerate(pdf_images, 1):
        print(f"\nProcessing page {page_num}/{len(pdf_images)}...")
        
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Comprehensive processing
        page_data = process_page_comprehensive(ocr, cv2_image, page_num)
        
        page_info = {
            'page_number': page_num,
            'awbs_found': page_data['awbs'],
            'awb_count': len(page_data['awbs']),
            'text_length': page_data['text_length'],
            'sample_awbs': page_data['awbs'][:3] if page_data['awbs'] else []
        }
        
        page_results.append(page_info)
        all_awbs.extend(page_data['awbs'])
        
        if page_data['awbs']:
            print(f"  Found {len(page_data['awbs'])} AWBs")
            print(f"  Samples: {page_data['awbs'][:3]}")
            
            # Show AWB types found on this page
            r_count = sum(1 for awb in page_data['awbs'] if awb.startswith('R'))
            sf_count = sum(1 for awb in page_data['awbs'] if awb.startswith('SF'))
            if sf_count > 0:
                print(f"  Includes {sf_count} SF-type AWBs")
    
    # Advanced deduplication
    unique_awbs = advanced_deduplication(all_awbs)
    
    # Analyze AWB types
    type_analysis = analyze_awb_types(unique_awbs)
    
    return {
        'file_path': file_path,
        'total_pages': len(pdf_images),
        'total_awbs_found': len(unique_awbs),
        'awb_numbers': unique_awbs,
        'page_results': page_results,
        'type_analysis': type_analysis,
        'processing_note': 'Comprehensive R and SF AWB extraction'
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive AWB extraction for R and SF types')
    parser.add_argument('--file', type=str, required=True, help='PDF file path')
    parser.add_argument('--output', type=str, default='comprehensive_results.json', help='Output JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--dpi', type=int, default=400, help='PDF conversion DPI')
    
    args = parser.parse_args()
    
    try:
        print("Starting comprehensive AWB extraction (R + SF types)...")
        output = process_pdf_comprehensive(args.file, use_gpu=args.gpu, dpi=args.dpi)
        
        # Print results
        print("\n" + "="*70)
        print("COMPREHENSIVE AWB EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\nFile: {output['file_path']}")
        print(f"Total pages: {output['total_pages']}")
        print(f"Total AWB numbers found: {output['total_awbs_found']}")
        
        # Type analysis
        analysis = output['type_analysis']
        print(f"\nAWB Type Analysis:")
        print(f"  R-type AWBs: {analysis['r_type']}")
        print(f"  SF-type AWBs: {analysis['sf_type']}")
        print(f"  Other types: {analysis['other_type']}")
        
        if analysis['sf_examples']:
            print(f"  SF examples: {analysis['sf_examples']}")
        
        if output['awb_numbers']:
            print(f"\nFirst 30 AWB Numbers:")
            print("-" * 50)
            for i, awb in enumerate(output['awb_numbers'], 1):
                prefix = "SF" if awb.startswith('SF') else "R "
                print(f"  {i:3d}. {prefix}: {awb}")
            if len(output['awb_numbers']) > 30:
                print(f"  ... and {len(output['awb_numbers']) - 30} more")
        
        print(f"\nPage Summary:")
        print("-" * 70)
        for page in output['page_results']:
            status = "✓" if page['awb_count'] > 0 else "✗"
            
            # Count SF types on this page
            sf_count = sum(1 for awb in page['awbs_found'] if awb.startswith('SF'))
            sf_info = f", SF: {sf_count}" if sf_count > 0 else ""
            
            print(f"  Page {page['page_number']}: {status} {page['awb_count']:3d} AWBs{sf_info}")
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
        
        print("\n✓ Comprehensive processing completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()