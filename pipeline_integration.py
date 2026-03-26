import pandas as pd
import cv2
import numpy as np
import re
from PIL import Image

def clean_text(text):
    """Normalize text for fuzzy matching by removing special characters and keeping alphanumeric + spaces"""
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).strip()

def flatten_json(y):
    """Flattens nested JSON into a single level dictionary of keys to scalar values."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def match_qwen_to_ocr(qwen_json, ocr_results):
    """
    Matches Qwen extracted values to PaddleOCR text blocks to recover their bounding boxes.
    """
    if isinstance(qwen_json, dict) and 'page_1' in qwen_json:
        # Assuming single page for testing based on prompt
        qwen_json = qwen_json['page_1']
        
    flattened_qwen = flatten_json(qwen_json)
    
    # Keep track of OCR blocks that have already been assigned (avoid double-matching)
    used_ocr_indices = set()
    matched_data = []

    # Iterate through every Qwen Key specifically checking the VALUE
    for key, qwen_value in flattened_qwen.items():
        if not qwen_value or str(qwen_value).strip() == "":
            continue
            
        target_clean = clean_text(qwen_value)
        if not target_clean or len(target_clean) < 2:
            continue
            
        # Keep track of what part of the Qwen string we STILL need to find bounding boxes for
        remaining_target = target_clean
            
        # Search through all OCR text strings to find pieces that make up the Qwen value string
        for idx, ocr_box in enumerate(ocr_results):
            
            # Stop if we already found the whole string for this Key!
            if len(remaining_target) < 2:
                break
                
            box_text_clean = clean_text(ocr_box['text'])
            if not box_text_clean:
                continue
            
            # Case 1: The OCR chunk is a clean substring of the Qwen target (e.g. OCR: "Ted's", Qwen: "Ted's Small Business")
            # We completely consume this OCR physical box.
            if idx not in used_ocr_indices and box_text_clean in remaining_target:
                matched_data.append({
                    "Key": key,
                    "Qwen_Value": str(qwen_value),
                    "OCR_Matched_Text": ocr_box['text'],
                    "Confidence": ocr_box['confidence'],
                    "BBox": ocr_box['bbox']
                })
                # Once uniquely matched, we don't assign this exact OCR chunk to any other Qwen field
                used_ocr_indices.add(idx)
                # Subtract this found string from our checklist
                remaining_target = remaining_target.replace(box_text_clean, "", 1)

            # Case 2: The REMAINING Qwen field is a smaller piece inside a huge OCR line
            # We must calculate a mathematical sub-bounding-box and we DO NOT consume the physical box!
            elif remaining_target in box_text_clean and len(remaining_target) >= 4:
                # Find character indices
                start_idx = box_text_clean.find(remaining_target)
                end_idx = start_idx + len(remaining_target)
                
                # Calculate character position ratios (0.0 to 1.0)
                start_ratio = start_idx / len(box_text_clean)
                end_ratio = end_idx / len(box_text_clean)
                
                # Fetch original BBox points
                x1, y1 = ocr_box['bbox'][0] # Top-Left
                x2, y2 = ocr_box['bbox'][1] # Top-Right
                x3, y3 = ocr_box['bbox'][2] # Bottom-Right
                x4, y4 = ocr_box['bbox'][3] # Bottom-Left
                
                # Calculate width of top and bottom edges
                top_width = x2 - x1
                bottom_width = x3 - x4
                
                # Interpolate new X coordinates
                new_x1 = x1 + (top_width * start_ratio)
                new_x2 = x1 + (top_width * end_ratio)
                new_x4 = x4 + (bottom_width * start_ratio)
                new_x3 = x4 + (bottom_width * end_ratio)
                
                # Generate new localized bounding box for just the target word(s)
                sub_bbox = [
                    [new_x1, y1], 
                    [new_x2, y2], 
                    [new_x3, y3], 
                    [new_x4, y4]
                ]
                
                matched_data.append({
                    "Key": key,
                    "Qwen_Value": str(qwen_value),
                    "OCR_Matched_Text": f"{str(qwen_value)} (Sub-extract)",
                    "Confidence": ocr_box['confidence'],
                    "BBox": sub_bbox
                })
                # We found everything we needed inside this huge box!
                remaining_target = ""
                break

    return matched_data

def highlight_matches_on_image(image_path, matched_data, output_path):
    """
    Draws light green highlights (no border) and labels the keys in red.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}. Check path.")
        return
        
    overlay = image.copy()
    
    # Group bounding boxes by their Qwen Key
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for m in matched_data:
        bbox = m['BBox']
        grouped[m['Key']].append(bbox)
        
        # Draw light green filled rectangle for the highlight (BGR: 144, 238, 144)
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (144, 238, 144))
        
    # Blend the green rectangles transparently (no border visible)
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw Key names in red text in the free space above the topmost box for each group
    for key, bboxes in grouped.items():
        # Find the topmost-left bounding box for this group to label it
        top_left = min([b[0] for b in bboxes], key=lambda x: (x[1], x[0]))
        x, y = int(top_left[0]), int(top_left[1])
        
        # Write the key name in red (BGR: 0, 0, 255) offset slightly above
        cv2.putText(image, str(key), (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    if output_path.lower().endswith('.pdf'):
        # Convert OpenCV BGR to Pillow RGB for PDF saving
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        pil_img.save(output_path, "PDF", resolution=100.0)
        print(f"✅ Highlighted PDF successfully saved to: {output_path}")
    else:
        cv2.imwrite(output_path, image)
        print(f"✅ Highlighted image successfully saved to: {output_path}")

def export_to_excel(matched_data, excel_output_path="matched_results.xlsx"):
    """
    Dumps the final matched data, coordinates, and confidences to an Excel file.
    """
    df = pd.DataFrame(matched_data)
    df.to_excel(excel_output_path, index=False)
    print(f"✅ Extracted data successfully exported to: {excel_output_path}")

if __name__ == "__main__":
    print("Integration pipeline loaded successfully.")
    # Here is an example of how you run it using your existing engines
    # 
    # ocr_output = extractor.extract_text_with_confidence("Document_1.pdf")
    # qwen_output = qwen_extractor.extract_data("Document_1.pdf")
    # 
    # matched_results = match_qwen_to_ocr(qwen_output, ocr_output)
    # highlight_matches_on_image("Document_1.jpg", matched_results, "output.pdf")
    # export_to_excel(matched_results, "matched_results.xlsx")
