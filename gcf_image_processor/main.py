import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import urllib.parse
import urllib.request
from paddleocr_onnx import PaddleOCR
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

def extract_awb_numbers(text: str) -> List[str]:
    pattern = r'\b(SF\d{8,20}[A-Z]{1,3}|R\d{8,21}[A-Z]{1,3}|SF\d{8}R)\b'
    matches = re.findall(pattern, text)
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique

def is_pdf(file_path: str) -> bool:
    return Path(file_path).suffix.lower() == '.pdf'

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=300)

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def process_image_with_ocr(ocr: PaddleOCR, img: np.ndarray, page_num: int = None):
    results = ocr.ocr(img, cls=False)
    extracted = []
    parts = []
    if results and len(results) > 0:
        for line in results[0]:
            if line and len(line) >= 2:
                bbox = line[0]
                info = line[1]
                if isinstance(info, tuple) and len(info) >= 2:
                    text = info[0]
                    conf = float(info[1])
                    extracted.append({
                        'text': text,
                        'confidence': round(conf, 4),
                        'bbox': [[float(c[0]), float(c[1])] for c in bbox],
                        'page': page_num
                    })
                    parts.append(text)
    return extracted, " ".join(parts)

def run_ocr(file_path: str, use_gpu: bool = False) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=use_gpu, show_log=False)
    all_data = []
    all_text = []
    total_pages = 1
    if is_pdf(file_path):
        images = pdf_to_images(file_path)
        total_pages = len(images)
        for i, pil_img in enumerate(images, 1):
            cv2_img = pil_to_cv2(pil_img)
            data, text = process_image_with_ocr(ocr, cv2_img, page_num=i)
            all_data.extend(data)
            all_text.append(text)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Unable to read image from: {file_path}")
        data, text = process_image_with_ocr(ocr, img)
        all_data.extend(data)
        all_text.append(text)
    full_text = " ".join(all_text)
    awb_numbers = extract_awb_numbers(full_text)
    return {
        'file_path': file_path,
        'file_type': 'PDF' if is_pdf(file_path) else 'Image',
        'total_pages': total_pages,
        'total_text_lines': len(all_data),
        'extracted_text': all_data,
        'full_text': full_text.strip(),
        'awb_numbers': awb_numbers,
        'awb_count': len(awb_numbers)
    }

def _download_to_tmp(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    ext = os.path.splitext(parsed.path)[1] or '.jpg'
    tmp_path = os.path.join(tempfile.gettempdir(), f"gcf_input{ext}")
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path

def process_image(request):
    if request.method != 'POST':
        return json.dumps({'error': 'Method not allowed'}), 405, {'Content-Type': 'application/json'}
    use_gpu = False
    try:
        qp = request.args
        if qp and 'use_gpu' in qp:
            use_gpu = qp.get('use_gpu') in ('1', 'true', 'True')
    except Exception:
        pass
    tmp_path = None
    if hasattr(request, 'files') and 'file' in request.files:
        f = request.files['file']
        filename = getattr(f, 'filename', '') or 'upload'
        ext = os.path.splitext(filename)[1] or '.jpg'
        tmp_path = os.path.join(tempfile.gettempdir(), f"gcf_input{ext}")
        f.save(tmp_path)
    else:
        data = None
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None
        if data and 'image_url' in data:
            tmp_path = _download_to_tmp(data['image_url'])
        elif data and 'file_path' in data:
            tmp_path = data['file_path']
        else:
            return json.dumps({'error': 'Provide file in form-data or image_url/file_path in JSON'}), 400, {'Content-Type': 'application/json'}
        if data and 'use_gpu' in data:
            use_gpu = bool(data.get('use_gpu'))
    try:
        result = run_ocr(tmp_path, use_gpu=use_gpu)
        return json.dumps(result), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}
