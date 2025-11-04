import base64
import os
import re
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
from difflib import SequenceMatcher
from io import BytesIO
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from mmocr.apis import TextRecInferencer
import mmcv
import mmengine
import traceback
import concurrent.futures
import urllib.request
from pdf2image import convert_from_bytes
import fasttext
import json
import uuid
import hashlib
import datetime
import uuid
import hashlib
from typing import Dict, Any, Optional

if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128]
    }

url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
urllib.request.urlretrieve(url, "lid.176.ftz")
model_lang_detect = fasttext.load_model("lid.176.ftz")
print("Tải thành công lid.176.ftz")

print("mmcv version:", mmcv.__version__)
print("mmengine version:", mmengine.__version__)

print("MMOCR inference ready!")

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

SCHEMA_PATTERNS = {
    "company_name": [
        r"công\s*ty",
        r"doanh\s*nghiệp",
        r"tập\s*đoàn",
    ],
    "company_type": [
        r"trách\s*nhiệm\s*hữu\s*hạn|t\.?n\.?h\.?h",
        r"cổ\s*phần|c\.?p",
        r"một\s*thành\s*viên|m\.?t\.?v",
        r"hợp\s*tác|h\.?t\.?x",
    ],
    "personal_info.id_type": [
        r"cccd|cmnd|căn\s*cước",
        r"hộ\s*chiếu|passport",
    ],
    "personal_info.id_number": [
        r"\b\d{9,12}\b",
    ],
    "personal_info.full_name": [
        r"(ông|bà|chị|anh|nguyễn|trần|phạm|lê|vũ)\s+[A-ZÀÁẢÃẠÂĂĐÊÔƠƯ][\w\s]+",
    ],
    "appointment_date": [
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
    ],
    "signing_authority": [
        r"giám\s*đốc|tổng\s*giám\s*đốc|chủ\s*tịch|phó\s*giám\s*đốc",
    ],
}

def load_text_detection_model():
    """Load PaddleOCR (DBNet/DBNet++) for detection only."""
    ocr_detector = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)
   
    return ocr_detector
text_detector = load_text_detection_model()

def fasttext_detect_lang(text):
    """
    Detect language using FastText, return lang + confidence
    """
    if not text or not text.strip():
        return "unknown", 0.0

    text_norm = text.strip()

    try:
        labels, probs = model_lang_detect.predict(text_norm)
        lang = labels[0].replace("__label__", "")
        prob = probs[0]
    except Exception:
        lang, prob = "unknown", 0.0

    # Nếu confidence cao → tin tưởng kết quả
    if prob >= 0.5:
        return lang, prob

    # --- Fallback khi FastText không chắc chắn ---
    clean_text = re.sub(r'[^\w\s]', '', text_norm, flags=re.UNICODE).strip()
    if not clean_text:
        return "unknown", 0.0

    # Kiểm tra tiếng Việt
    if re.search(r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', clean_text, re.IGNORECASE):
        return "vi", 0.8  # giả định confidence trung bình 0.8

    # Nếu chỉ chứa ký tự Latin cơ bản (không dấu) → coi là tiếng Anh
    if re.fullmatch(r'[a-zA-Z0-9\s]+', clean_text):
        return "en", 0.8

    return "unknown", 0.0

def detect_text_regions(image):
    """Detect text boxes and recognition results using PaddleOCR predict()."""
    import tempfile
    tmp_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    start_time = time.time()
    results = text_detector.predict(tmp_path)
    runtime_total = round(time.time() - start_time, 3)

    df_records = []
    boxes = []
    box_times = []

    if not results:
        print("Không phát hiện vùng chữ nào.")
        return boxes, pd.DataFrame()

    for res in results:
        # Mỗi phần tử có thể là object hoặc dict
        if hasattr(res, "res"):
            res_data = res.res
        elif isinstance(res, dict):
            res_data = res
        elif hasattr(res, "__dict__"):
            res_data = res.__dict__
        else:
            print("Không nhận dạng được kiểu kết quả:", type(res))
            continue

        # Debug thử
        print("🔑 Keys trong res_data:", list(res_data.keys()))

        dt_polys = res_data.get("dt_polys")
        dt_scores = res_data.get("dt_scores", [])
        rec_texts = res_data.get("rec_texts", [])
        rec_scores = res_data.get("rec_scores", [])

        if dt_polys is None:
            print("Không có dt_polys trong res_data.")
            continue

        for i, poly in enumerate(dt_polys):
            try:
                box_start = time.time()
                points = np.array(poly, dtype=np.int32)
                x_min = int(np.min(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                x_max = int(np.max(points[:, 0]))
                y_max = int(np.max(points[:, 1]))
                score = float(dt_scores[i]) if i < len(dt_scores) else 1.0

                # Nếu có nhận dạng chữ
                text = rec_texts[i] if i < len(rec_texts) else ""
                rec_conf = float(rec_scores[i]) if i < len(rec_scores) else score

                # Phát hiện ngôn ngữ nếu có text
                if text.strip():
                    lang, lang_conf = fasttext_detect_lang(text)
                else:
                    lang, lang_conf = "unknown", 0.0
                
                box_runtime = round(time.time() - box_start, 4)

                boxes.append({
                    "bbox": (x_min, y_min, x_max, y_max),
                    "score": rec_conf,
                    "runtime": box_runtime
                })

                df_records.append({
                    "doc_preprocessor_res": {
                        "angle": 0,
                        "input_path": None,
                        "model_settings": {
                            "use_doc_orientation_classify": False,
                            "use_doc_unwarping": False
                        },
                        "page_index": None
                    },
                    "dt_polys": [poly],
                    "input_path": tmp_path,
                    "model_settings": {
                        "use_doc_preprocessor": False,
                        "use_textline_orientation": False
                    },
                    "page_index": None,
                    "rec_boxes": [list(map(int, [x_min, y_min, x_max, y_max]))],
                    "rec_polys": [poly],
                    "rec_scores": [rec_conf],
                    "rec_texts": [text],
                    "return_word_box": False,
                    "text_det_params": {
                        "box_thresh": 0.5,
                        "limit_side_len": 960,
                        "limit_type": "max",
                        "max_side_limit": 960,
                        "thresh": 0.3,
                        "unclip_ratio": 2.0
                    },
                    "text_rec_score_thresh": 0,
                    "text_type": "general",
                    "textline_orientation_angles": [],
                    "runtime": [box_runtime],
                    "lang": lang,
                    "lang_conf": float(lang_conf)
                })

            except Exception as e:
                print(f"❌ Lỗi khi xử lý polygon: {e}")
                continue

    df_paddle = pd.DataFrame(df_records)
    print(f"✅ PaddleOCR phát hiện {len(boxes)} vùng chữ, thời gian: {runtime_total:.3f}s")
    return boxes, df_paddle

def draw_unicode_text(img, text, pos, color=(0, 255, 0), font_path="Roboto-Black.ttf", font_size=18):
    """
    Vẽ chữ Unicode (tiếng Việt có dấu) lên ảnh OpenCV bằng Pillow.
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_text(image):
    annotated = image.copy()
    ocr_texts, raw_text_list = [], []

    # --- Load các mô hình OCR ---
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    rec_inferencer = TextRecInferencer(model='sar')
    
    def timed_run(func, crop, name):
        start = time.time()
        result = func(crop)
        end = time.time()
        result["runtime"] = round(end - start, 3)
        result["model"] = name
        return result

    # ========== HÀM OCR ==========
    def run_ocr_tesseract(crop):
        try:
            # gray = preprocess_for_ocr(crop)
            config = '--oem 3 --psm 6 -l vie+eng'
            text = pytesseract.image_to_string(crop, config=config).strip()
            text = re.sub(r'\s+', ' ', text)
            conf = 0.5 if text else 0.0
            lang, lang_conf = fasttext_detect_lang(text) if text else ("unknown", 0.0)
            return {"text": text, "conf": conf, "lang": lang, "lang_conf": lang_conf}
        except Exception as e:
            print(f"❌ Lỗi Tesseract: {e}")
            return {"text": "", "conf": 0.0, "lang": "unknown", "lang_conf": 0.0}

    def run_ocr_easyocr(crop):
        try:
            # gray = preprocess_for_ocr(crop)
            results = reader.readtext(crop)
            if not results:
                return {"text": "", "conf": 0.0, "lang": "unknown", "lang_conf": 0.0}
            _, text, conf = max(results, key=lambda x: x[2])
            lang, lang_conf = fasttext_detect_lang(text) if text else ("unknown", 0.0)
            return {"text": text, "conf": conf, "lang": lang, "lang_conf": lang_conf}
        except Exception as e:
            print(f"❌ Lỗi EasyOCR: {e}")
            return {"text": "", "conf": 0.0, "lang": "unknown", "lang_conf": 0.0}

    # ========== CÔNG CỤ HỖ TRỢ ==========
    def clean_text(t):
        t = re.sub(r'[^0-9A-Za-zÀ-ỹ\s.,:/%()+=-]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    
    def has_good_spacing(text):
        return bool(re.search(r'[A-Za-zÀ-ỹ]+\s+[A-Za-zÀ-ỹ]+', text))

    def normalize_case(text):
        if not text.strip():
            return text

        letters = re.findall(r'[A-Za-zÀ-ỹ]', text)
        if not letters:
            return text  

        upper_count = sum(1 for c in letters if c.isupper())
        lower_count = len(letters) - upper_count

        if upper_count >= 0.7 * len(letters):
            return text.upper()

        elif lower_count >= 0.7 * len(letters):
            return text.lower()
        else:
            return text
    
    def text_similarity(a, b):
        a_clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', a.lower())
        b_clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', b.lower())
        ratio = SequenceMatcher(None, a_clean, b_clean).ratio()
        return ratio * (1 - abs(len(a_clean) - len(b_clean)) / max(len(a_clean), len(b_clean), 1))
 
    def choose_best_text(results):
        # --- Lấy tất cả text không rỗng ---
        texts = [r for r in results.values() if r["text"]]
        for r in texts:
            r["text"] = normalize_case(clean_text(r["text"]))

        if not texts:
            return "(rỗng)", 0.0, "unknown"

        # --- Lọc text có lang hợp lệ ---
        valid_langs = [t for t in texts if t.get("lang") and t["lang"] != "unknown"]

        if valid_langs:
            # 1️⃣ Chọn candidate lang_conf cao nhất
            max_lang_conf = max(t.get("lang_conf", 0.0) for t in valid_langs)
            top_candidates = [t for t in valid_langs if t.get("lang_conf", 0.0) == max_lang_conf]
            
            if len(top_candidates) > 1:
                mean_conf = np.mean([t["conf"] for t in top_candidates])
                print(f"--- Mean conf toàn cục: {mean_conf:.3f} ---")

                # --- Bước 2: lọc text dưới mean_conf toàn cục ---
                filtered_candidates = []
                for t in top_candidates:
                    if t["conf"] >= mean_conf:
                        filtered_candidates.append(t)
                        print(f"Giữ text toàn cục: {t['text']} ({t['conf']:.2f}) >= {mean_conf:.2f}")
                    else:
                        print(f"Loại text toàn cục: {t['text']} ({t['conf']:.2f}) < {mean_conf:.2f}")
                if not filtered_candidates:
                    filtered_candidates = top_candidates
                    print("Không còn text nào sau lọc toàn cục, fallback top_candidates cũ")

                # --- Bước 3: phân nhóm theo text chuẩn hóa ---
                def normalize_text_for_grouping(t):
                    s = t.lower()
                    return s

                grouped = {}
                for t in filtered_candidates:
                    key = normalize_text_for_grouping(t["text"])
                    grouped.setdefault(key, []).append(t)
                
                print(f"Số nhóm sau phân nhóm: {len(grouped)}")
                for key, group in grouped.items():
                    group_mean_conf = np.mean([x["conf"] for x in group])
                    print(f" Nhóm '{key}' - mean conf nhóm: {group_mean_conf:.3f}")
                    for x in group:
                        if x["conf"] >= group_mean_conf:
                            print(f"   Giữ: {x['text']} ({x['conf']:.2f}) >= {group_mean_conf:.2f}")
                        else:
                            print(f"   Loại: {x['text']} ({x['conf']:.2f}) < {group_mean_conf:.2f}")

                # --- Bước 4: lọc trong mỗi nhóm theo group mean conf ---
                top_candidates = []
                for key, group in grouped.items():
                    group_mean_conf = np.mean([x["conf"] for x in group])
                    group_filtered = [x for x in group if x["conf"] >= group_mean_conf]
                    if group_filtered:
                        top_candidates.extend(group_filtered)

                print(f"Ngưỡng toàn cục: {mean_conf:.2f}")
                print(f"Số nhóm: {len(grouped)}, giữ lại {len(top_candidates)} text sau lọc nhóm")

            # else:
            #     continue

            # 2️⃣ Nếu nhiều candidate bằng nhau, lọc theo spacing
            spaced_candidates = [t for t in top_candidates if has_good_spacing(t["text"])]
            pool = spaced_candidates if spaced_candidates else top_candidates

            # 3️⃣ Trong pool, chọn những text đồng bộ về case (toàn upper hoặc toàn lower)
            case_candidates = []
            for t in pool:
                letters = re.findall(r'[A-Za-zÀ-ỹ]', t["text"])
                if not letters:
                    continue
                upper_count = sum(1 for c in letters if c.isupper())
                lower_count = len(letters) - upper_count
                if upper_count == len(letters) or lower_count == len(letters):
                    case_candidates.append(t)

            if case_candidates:
                # --- Đếm tần suất xuất hiện text ---
                text_counter = {}
                for t in case_candidates:
                    txt = t["text"]
                    if txt in text_counter:
                        text_counter[txt].append(t)
                    else:
                        text_counter[txt] = [t]

                # --- Tìm text xuất hiện nhiều nhất ---
                max_count = max(len(v) for v in text_counter.values())
                most_common_texts = [v for v in text_counter.values() if len(v) == max_count]

                if len(most_common_texts) == 1:
                    # Chỉ có 1 text phổ biến nhất
                    best = max(most_common_texts[0], key=lambda t: t["conf"])
                else:
                    # Nếu nhiều text cùng tần suất, chọn conf cao nhất
                    best = max([t for group in most_common_texts for t in group], key=lambda t: t["conf"])
            else:
                # Không có case_candidates → fallback conf cao nhất trong pool
                best = max(pool, key=lambda t: t["conf"])

            return best["text"], float(best["conf"]), best["lang"]

        # --- Nếu không có lang hợp lệ, fallback theo độ giống nhau ---
        groups = []
        for t in texts:
            found = False
            for g in groups:
                if text_similarity(g[0]["text"], t["text"]) > 0.88:
                    g.append(t)
                    found = True
                    break
            if not found:
                groups.append([t])

        best_group = max(groups, key=lambda g: len(g))
        best = max(best_group, key=lambda t: t["conf"])
        final_text = normalize_case(best["text"])

        try:
            lang, _ = fasttext_detect_lang(final_text)
        except:
            lang = "unknown"

        return final_text, float(best["conf"]), lang

    # ========== PHÁT HIỆN VÙNG CHỮ ==========
    boxes, df_paddle  = detect_text_regions(image)
    if not boxes:
        print(" Không phát hiện vùng chữ, fallback toàn ảnh.")
        boxes = [(0, 0, image.shape[1], image.shape[0])]

    # ========== CHẠY OCR TRÊN TỪNG VÙNG ==========
    for idx, box in enumerate(boxes, start=1):
        try:
            (x_min, y_min, x_max, y_max) = box if not isinstance(box, dict) else box["bbox"]
            pad = 0
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(image.shape[1], x_max + pad), min(image.shape[0], y_max + pad)
            crop = image[y_min:y_max, x_min:x_max]

            rec_text, rec_conf, rec_lang, rec_lang_conf = "", 0.0, "unknown", 0.0
            if not df_paddle.empty:
                # Tìm hàng có rec_boxes trùng bbox hiện tại
                match = df_paddle[
                    df_paddle["rec_boxes"].apply(
                        lambda b: list(map(int, [x_min, y_min, x_max, y_max])) in b
                    )
                ]
                if not match.empty:
                    rec_text = match.iloc[0]["rec_texts"][0]
                    rec_conf = float(match.iloc[0]["rec_scores"][0])
                    rec_lang = match.iloc[0].get("lang", "unknown")
                    rec_lang_conf = float(match.iloc[0].get("lang_conf", 0.0))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    "tesseract": executor.submit(timed_run, run_ocr_tesseract, crop, "tesseract"),
                    "easyocr": executor.submit(timed_run, run_ocr_easyocr, crop, "easyocr"),
                    # "mmocr": executor.submit(timed_run, run_ocr_mmocr, crop, "mmocr"),
                    # "trocr": executor.submit(timed_run, run_ocr_trocr, crop, "trocr")
                }
                results = {k: f.result() for k, f in futures.items()}

            # results["paddleocr"] = {
            #     "text": rec_text,
            #     "conf": rec_conf,
            #     "lang": rec_lang,
            #     "lang_conf": rec_lang_conf,
            #     "runtime": 0.0,
            #     "model": "paddleocr"
            # }
            # ---- Chọn text tốt nhất ----
            best_text, best_conf, best_lang = choose_best_text(results)
            best_text = corrector(best_text, max_length=MAX_LENGTH)

            # 🔧 Đảm bảo best_text luôn là chuỗi (string)
            if isinstance(best_text, list):
                # Trường hợp corrector trả về [{'generated_text': '...'}]
                if len(best_text) > 0 and isinstance(best_text[0], dict) and "generated_text" in best_text[0]:
                    best_text = best_text[0]["generated_text"]
                else:
                    best_text = " ".join(map(str, best_text))
            elif isinstance(best_text, dict) and "generated_text" in best_text:
                best_text = best_text["generated_text"]
            elif not isinstance(best_text, str):
                best_text = str(best_text)

            print(f"text đã chuẩn hóa: {best_text}")

            log_entry = {
                
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "region_index": idx,
                "bbox": [x_min, y_min, x_max, y_max],
                "models": results,
                "best_text": best_text,
                "best_conf": best_conf,
                "best_lang": best_lang
            }
            # append_log(log_entry)

            color = (0, 255, 0) if best_text != "(rỗng)" else (0, 200, 255)
            annotated = draw_unicode_text(annotated, best_text, (x_min, max(0, y_min - 18)), color)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

            ocr_texts.append({
                "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                "models": results,
                "final_text": best_text,
                "final_conf": best_conf,
                "lang": best_lang
            })
            raw_text_list.append(best_text)

            print(f"\nVùng {idx}:")
            for name, r in results.items():
                print(f"  {name:<12} → {r['text']} ({r['conf']:.2f}) [lang={r.get('lang','unknown')}, lang_conf={r.get('lang_conf',0.0):.2f}]")

            print(f" Chọn: {best_text} ({best_conf:.2f}) [lang={best_lang}]")

        except Exception as e:
            print(f" Lỗi xử lý vùng {idx}: {e}")
            traceback.print_exc()

    print(f"\nHoàn tất OCR: {len(ocr_texts)} vùng.")
    return " ".join(map(str, raw_text_list)), ocr_texts, annotated

MAX_LENGTH = 512
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")

# --- Hằng số Mapping ---
COMPANY_TYPE_MAPPING: Dict[str, str] = {
    # Từ đầy đủ
    "CÔNG TY": "CT",
    "CÔNGTY": "CT",
    "TRÁCH NHIỆM HỮU HẠN": "TNHH",
    "MỘT THÀNH VIÊN": "MTV",
    "HAI THÀNH VIÊN": "HTV",
    "CỔ PHẦN": "CP",
    "PHÁT TRIỂN": "PT",
    "THƯƠNG MẠI": "TM",
    "DỊCH VỤ": "DV",
    "ĐẦU TƯ": "ĐT",
    # Từ viết tắt
    "CT": "CT",
    "TNHH": "TNHH",
    "MTV": "MTV",
    "HTV": "HTV",
    "CP": "CP",
    "PT": "PT",
    "TM": "TM",
    "DV": "DV",
    "ĐT": "ĐT",
    "TV": "TV", # Tư Vấn (giả định)
    "XD": "XD", # Xây Dựng (giả định)
    "VT": "VT", # Vận Tải (giả định)
    "GP": "GP", # Giải Pháp (giả định)
    "JSC": "JSC", # Joint Stock Company
    "CO LTD": "Co LTD", # Company Limited
    "HKD": "HKD" # Hộ Kinh Doanh
}

# Ánh xạ cho ID Type (Giấy tờ tùy thân)
ID_TYPE_MAPPING: Dict[str, str] = {
    "CĂN CƯỚC CÔNG DÂN": "CCCD",
    "CCCD": "CCCD",
    "CHỨNG MINH NHÂN DÂN": "CMND",
    "CMND": "CMND",
    "HỘ CHIẾU": "Passport",
    "HỌ CHIẾU": "Passport", # Lỗi OCR thường gặp
    "PASSPORT": "Passport",
}

# --- Hàm normalize_appointment_text ---
def normalize_appointment_text(ocr_text: str) -> Dict[str, Any]:
    """
    Ánh xạ text thô từ OCR thành các trường chuẩn hóa theo schema.
    """
    normalized_data = {
        "company_name": "",
        "company_type": "",
        "personal_info": {
            "id_type": "",
            "id_number": "",
            "full_name": ""
        },
        "appointment_date": {
            "day": 0,
            "month": 0,
            "year": 0
        },
        "signing_authority": ""
    }
    
    # 1. Chuẩn hóa cơ bản toàn bộ text thô (chữ hoa và loại bỏ dấu/ký tự đặc biệt không cần thiết)
    # Loại bỏ các ký tự đặc biệt, giữ lại chữ cái, số, dấu tiếng Việt
    clean_text = re.sub(r'[^A-ZÀ-Ỹ0-9\s/]', ' ', ocr_text.upper()) 
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # 2. Xử lý ID Type (CCCD/CMND/Passport)
    
    # Tìm kiếm các từ khóa ID Type trong text thô
    id_type_found: Optional[str] = None
    for full_form, enum_val in ID_TYPE_MAPPING.items():
        # Tìm kiếm cụm từ đầy đủ/viết tắt/lỗi OCR, không phân biệt chữ hoa
        if full_form in clean_text:
            id_type_found = enum_val
            break
            
    # Xử lý trường hợp có nhiều loại giấy tờ liên tiếp (CCCD/CMND/HỘ CHIẾU)
    if id_type_found:
        normalized_data["personal_info"]["id_type"] = id_type_found
        
        # --- Tìm số ID (thủ công) ---
        # Tạm thời chỉ tìm kiếm một chuỗi số/ký tự sau từ khóa ID.
        # Đây là phần phức tạp nhất, cần regex phức tạp hơn trong thực tế.
        try:
            # Tìm kiếm: Từ khóa ID (CMND/CCCD/HỘ CHIẾU) + 0-5 khoảng trắng/ký tự/dấu hai chấm + số
            match_id = re.search(r'(' + '|'.join(ID_TYPE_MAPPING.keys()).replace(' ', '\s+') + r')[\s:\-\/]{0,5}([0-9A-Z]{7,15})', clean_text)
            if match_id:
                normalized_data["personal_info"]["id_number"] = match_id.group(2).strip()
        except:
            pass # Bỏ qua nếu không tìm được số ID
            
    # 3. Xử lý Company Type (CT/TNHH/CP/...)
    
    # Tìm kiếm cụm từ viết tắt hoặc đầy đủ trong text thô
    company_type_found: Optional[str] = None
    for full_form, enum_val in COMPANY_TYPE_MAPPING.items():
        # Kiểm tra exact match (sau khi upper)
        # Sử dụng ranh giới từ (\b) để tránh nhầm lẫn "CT" với "CHUTICH"
        if re.search(r'\b' + re.escape(full_form) + r'\b', clean_text):
            company_type_found = enum_val
            break
            
    if company_type_found:
        normalized_data["company_type"] = company_type_found
        
        # --- Xử lý Company Name ---
        # Tạm thời đặt tên công ty là phần còn lại của dòng có Company Type.
        # Đây cũng là phần phức tạp cần NLP mạnh hơn, tạm thời làm đơn giản.
        try:
            # Tìm kiếm dòng chứa Company Type và tách ra.
            company_type_regex = r'\b(' + '|'.join(COMPANY_TYPE_MAPPING.keys()).replace(' ', '\s+') + r')\b'
            
            # Tìm kiếm dòng chứa loại hình công ty
            lines = ocr_text.split('\n')
            for line in lines:
                if re.search(company_type_regex, line.upper()):
                    # Lấy text trước hoặc sau từ khóa loại hình công ty
                    
                    # Giữ nguyên case cho Company Name
                    parts = re.split(company_type_regex, line, flags=re.IGNORECASE)
                    
                    if len(parts) >= 3:
                        # Ví dụ: "CÔNG TY ABC TNHH" -> [ '', 'CÔNG TY', ' ABC ', 'TNHH', '']
                        # Ghép phần trước và sau loại hình (ví dụ: ' CÔNG TY ABC TNHH' -> 'CÔNG TY ABC')
                        
                        # Loại bỏ loại hình công ty và các ký tự đặc biệt
                        name_part = parts[0] + parts[2]
                        name_part = re.sub(r'\b(' + '|'.join(COMPANY_TYPE_MAPPING.keys()).replace(' ', '\s+') + r')\b', '', name_part, flags=re.IGNORECASE).strip()
                        
                        # Giả định company name là phần còn lại của dòng
                        if name_part:
                            normalized_data["company_name"] = name_part
                            break
        except:
            pass

    # 4. Xử lý Appointment Date
    
    # Tạm thời tìm kiếm theo format D/M/Y hoặc D-M-Y hoặc D.M.Y
    try:
        match_date = re.search(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})', clean_text)
        if match_date:
            day, month, year = map(int, match_date.groups())
            if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= datetime.date.today().year + 1:
                normalized_data["appointment_date"]["day"] = day
                normalized_data["appointment_date"]["month"] = month
                normalized_data["appointment_date"]["year"] = year
    except:
        pass
        
    # 5. Xử lý Full Name (thủ công) và Signing Authority
    
    # Phần này cực kỳ khó vì không có cấu trúc cố định. Cần mô hình NER.
    # Tạm thời bỏ qua phần tìm kiếm Full Name và Signing Authority cho đến khi có cấu trúc rõ ràng.
    
    return normalized_data

def extract_text_v2(image, user_id="user_001"):
    """
    Chạy OCR (Tesseract + EasyOCR + PaddleOCR) và trả về JSON theo schema AppointmentDecisionRaw.
    """
    # --- 1️⃣ Tính hash ảnh ---
    image_hash = "fake_hash"

    # --- 2️⃣ Chạy OCR ---
    
    # Giả định kết quả OCR thô cho ví dụ
    text_all = "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM. CÔNG TY TNHH Xây Dựng ABC. Người đại diện: NGUYỄN VĂN A. CCCD số 012345678901. Ngày ký: 15/09/2024. Chức danh: TỔNG GIÁM ĐỐC"
    ocr_regions = [] 
    annotated = None 

    # --- 3️⃣ Gom text theo model ---
    tesseract_texts = []
    easyocr_texts = []
    paddle_boxes = []

    # Giả định ocr_regions đã được điền...
    tesseract_text = " ".join(tesseract_texts).strip() or text_all
    easyocr_text = " ".join(easyocr_texts).strip() or text_all 

    # --- 4️⃣ Chuẩn hóa thông tin ---
    # GỌI HÀM CHUẨN HÓA MỚI
    normalized = normalize_appointment_text(text_all)

    # --- 5️⃣ Tạo JSON kết quả ---
    result = {
        "_id": f"dec_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
        "user_id": user_id,
        "image_hash": image_hash,
        "ocr_raw": {
            "paddle_boxes": paddle_boxes,
            "tesseract_text": tesseract_text,
            "easyocr_text": easyocr_text
        },
        "normalized": normalized,
        "status": "pending",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

    return result, annotated

def pdf_to_images(content: bytes, dpi=150):
    """
    Chuyển PDF bytes thành danh sách ảnh numpy (mỗi trang một ảnh)
    """
    pages = convert_from_bytes(content, dpi=dpi)
    images = [np.array(p.convert("RGB")) for p in pages]
    return images

def process_image(content: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        print("📄 Phát hiện file PDF, chuyển sang ảnh...")
        images = pdf_to_images(content)
        results = []

        for idx, img_array in enumerate(images, start=1):
            print(f"\n--- Xử lý trang {idx}/{len(images)} ---")
            text, details, annotated = extract_text(img_array)
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            results.append({
                "page": idx,
                "text": text,
                "details": details,
                "annotated_image": annotated_b64
            })
        return results

    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        print("🖼️ Phát hiện file ảnh, xử lý trực tiếp...")
        img = Image.open(BytesIO(content)).convert("RGB")
        img_array = np.array(img)
        text, details, annotated = extract_text(img_array)
        _, buffer = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        return [{
            "page": 1,
            "text": text,
            "details": details,
            "annotated_image": annotated_b64
        }]
    
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")

def compute_image_hash(image_array: np.ndarray) -> str:
    """Tính SHA256 hash của ảnh numpy array (RGB)."""
    # Đảm bảo định dạng nhất quán
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    # Chuyển sang bytes theo thứ tự cố định
    img_bytes = image_array.tobytes()
    return "sha256:" + hashlib.sha256(img_bytes).hexdigest()

def build_appointment_decision_json(
    image_array: np.ndarray,
    ocr_results: list,  # danh sách từ `extract_text()`: mỗi phần tử là dict có "models", "bbox", ...
    user_id: str = "user_001",
    doc_id: str = None
) -> dict:
    """
    Xây dựng JSON theo schema collection_appointment_decisions.
    Chỉ lưu OCR raw, normalized để trống (sẽ được điền sau bởi BERT).
    """
    # 1. Tạo ID
    if doc_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"dec_{timestamp}_{str(uuid.uuid4())[:8]}"

    # 2. Tính image hash
    img_hash = compute_image_hash(image_array)

    # 3. Trích xuất text từ từng engine
    tesseract_texts = []
    easyocr_texts = []
    paddle_boxes = []

    for region in ocr_results:
        bbox = region["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        paddle_boxes.append({
            "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            "bbox": [x, y, w, h]
        })

        models = region.get("models", {})
        tess = models.get("tesseract", {}).get("text", "")
        easy = models.get("easyocr", {}).get("text", "")
        if tess:
            tesseract_texts.append(tess)
        if easy:
            easyocr_texts.append(easy)

    ocr_raw = {
        "paddle_boxes": paddle_boxes,
        "tesseract_text": "\n".join(tesseract_texts),
        "easyocr_text": "\n".join(easyocr_texts)
    }

    # 4. Xây dựng JSON
    output = {
        "_id": doc_id,
        "user_id": user_id,
        "image_hash": img_hash,
        "ocr_raw": ocr_raw,
        "normalized": {},  # để trống, sẽ được BERT điền sau
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    return output
 
def main():
    path = r"./QDBN1.pdf"

    if not os.path.exists(path):
        print(f"❌ File không tồn tại: {path}")
        return

    with open(path, "rb") as f:
        content = f.read()

    ext = os.path.splitext(path)[1].lower()
    base_name = os.path.splitext(os.path.basename(path))[0]

    if ext == ".pdf":
        print("📄 Phát hiện file PDF, chuyển sang ảnh...")
        images = pdf_to_images(content)

        for idx, img in enumerate(images, start=1):
            print(f"\n--- Xử lý trang {idx}/{len(images)} ---")

            # chạy OCR trực tiếp
            text, details, annotated = extract_text(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # lưu annotate
            preview_path = f"annotated_page_{idx}.jpg"
            cv2.imwrite(preview_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✅ Ảnh chú thích OCR trang {idx} đã lưu tại: {preview_path}")

            # build JSON
            json_data = build_appointment_decision_json(
                image_array=np.array(img),
                ocr_results=details,
                user_id="user_001",
                doc_id=f"dec_{base_name}_page{idx}"
            )
            json_path = f"appointment_decision_{base_name}_page{idx}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã lưu JSON schema trang {idx} vào: {json_path}")
            
    else:
        print(f"⚠️ Không hỗ trợ định dạng: {ext}")

if __name__ == "__main__":
    main()
