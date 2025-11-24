
import re
from typing import List, Dict, Any

def mock_extract_awb_numbers(text: str) -> List[str]:
    # This mimics the improved logic we want to implement
    # We will copy the logic from main.py once implemented, but for now let's define the expected behavior
    # actually, I should implement the logic here first to test it.
    
    pattern = r"\b(?:SF|R)[\s\-\/]*[A-Z0-9](?:[\s\-\/]*[A-Z0-9]){7,24}(?:[\s\-\/]*[A-Z]{0,4})?\b|\bSF[\s\-\/]*[A-Z0-9](?:[\s\-\/]*[A-Z0-9]){7}[\s\-\/]*R\b"
    matches = re.findall(pattern, text)
    
    def normalize_and_validate(t: str):
        t = re.sub(r"[\s\-/]+", "", t).upper()
        
        # Fix common OCR errors in prefix/suffix
        if t.startswith("5F"): t = "SF" + t[2:]
        
        if t.startswith("SF") and t.endswith("R") and len(t) >= 11:
            core = t[2:-1]
            map_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'Q': '9', 'D': '0'}
            fixed = ''.join(map_table.get(ch, ch) for ch in core)
            if re.fullmatch(r"\d{8}", fixed):
                return "SF" + fixed + "R"
            return None
            
        m = re.match(r"^(SF|R)([A-Z0-9]{8,24})([A-Z]{0,4})$", t)
        if not m:
            return None
        prefix, mid, suffix = m.groups()
        
        # If suffix is empty but mid ends with letters, try to shift them to suffix
        if not suffix and mid:
            # Find where the last digit is
            last_digit_idx = -1
            for i in range(len(mid) - 1, -1, -1):
                if mid[i].isdigit():
                    last_digit_idx = i
                    break
            
            if last_digit_idx < len(mid) - 1:
                # We have trailing non-digits
                potential_suffix = mid[last_digit_idx+1:]
                if len(potential_suffix) <= 4 and potential_suffix.isalpha():
                    mid = mid[:last_digit_idx+1]
                    suffix = potential_suffix

        map_table = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'Q': '9', 'D': '0'}
        fixed_mid = ''.join(map_table.get(ch, ch) for ch in mid)
        
        # Allow 1-2 non-digits in the middle if it looks like an AWB
        digit_count = sum(c.isdigit() for c in fixed_mid)
        if len(fixed_mid) - digit_count > 2:
             # Too many non-digits
             if not re.fullmatch(r"\d{8,21}", fixed_mid):
                return None

        # If mostly digits, try to force fix the rest
        final_mid = ""
        for ch in fixed_mid:
            if ch.isdigit():
                final_mid += ch
            elif ch in map_table:
                final_mid += map_table[ch]
            else:
                # If it's a very strong match otherwise, maybe ignore or treat as error?
                # For now, let's just keep it and see if regex matches
                final_mid += ch
        
        if not re.fullmatch(r"\d{8,21}", final_mid):
            return None
            
        if suffix and not re.fullmatch(r"[A-Z]{1,4}", suffix):
            return None
        return prefix + final_mid + (suffix or "")

    seen = set()
    out = []
    for m in matches:
        val = normalize_and_validate(m)
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out

def join_rows(extracted_data: List[Dict[str, Any]]) -> str:
    # Sort by Y then X
    # Simple clustering by Y
    rows = []
    current_row = []
    
    # Sort by top-left Y coordinate
    sorted_data = sorted(extracted_data, key=lambda x: x['bbox'][0][1])
    
    if not sorted_data:
        return ""
        
    current_y = sorted_data[0]['bbox'][0][1]
    row_threshold = 10 # pixels
    
    for item in sorted_data:
        y = item['bbox'][0][1]
        if abs(y - current_y) > row_threshold:
            # New row
            # Sort current row by X
            current_row.sort(key=lambda x: x['bbox'][0][0])
            rows.append(current_row)
            current_row = [item]
            current_y = y
        else:
            current_row.append(item)
            
    # Append last row
    if current_row:
        current_row.sort(key=lambda x: x['bbox'][0][0])
        rows.append(current_row)
    
    # Join text in rows
    full_text_parts = []
    for row in rows:
        row_text = "".join([item['text'] for item in row]) # Join without spaces to merge split tokens
        full_text_parts.append(row_text)
        
    return " ".join(full_text_parts)

def test_split_token_joining():
    # Case: "SF14" in one box, "00581262" in another, "AJI" in third
    data = [
        {'text': 'SF14', 'bbox': [[10, 10], [50, 10], [50, 30], [10, 30]]},
        {'text': '00581262', 'bbox': [[55, 12], [150, 12], [150, 32], [55, 32]]},
        {'text': 'AJI', 'bbox': [[155, 11], [180, 11], [180, 31], [155, 31]]},
        {'text': 'Other Text', 'bbox': [[10, 50], [100, 50], [100, 70], [10, 70]]}
    ]
    
    joined_text = join_rows(data)
    print(f"Joined Text: {joined_text}")
    
    awbs = mock_extract_awb_numbers(joined_text)
    print(f"Extracted AWBs: {awbs}")
    
    assert "SF1400581262AJI" in awbs

if __name__ == "__main__":
    test_split_token_joining()
