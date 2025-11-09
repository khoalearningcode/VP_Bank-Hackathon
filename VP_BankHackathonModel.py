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
from typing import Dict, Any
import uuid

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

    def run_ocr_mmocr(crop):
        try:
            # crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = rec_inferencer(crop)
            preds = result['predictions'][0]
            text, conf = preds['text'], preds['scores']
            lang, lang_conf = fasttext_detect_lang(text) if text else ("unknown", 0.0)
            return {"text": text, "conf": float(conf), "lang": lang, "lang_conf": lang_conf}
        except Exception as e:
            print(f"❌ Lỗi MMOCR: {type(e).__name__} - {e}")
            traceback.print_exc()
            return {"text": "", "conf": 0.0, "lang": "unknown", "lang_conf": 0.0}

    def run_ocr_trocr(crop):
        try:
            # image_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pixel_values = trocr_processor(images=crop, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = trocr_model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            text = trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            conf = 0.0
            if hasattr(outputs, "scores") and len(outputs.scores) > 0:
                probs = []
                sequence = outputs.sequences[0][1:]
                for score_tensor, token_id in zip(outputs.scores, sequence):
                    token_id = int(token_id)
                    probs.append(F.softmax(score_tensor.squeeze(0), dim=-1)[token_id].item())
                if probs:
                    conf = float(np.mean(probs))
            lang, lang_conf = fasttext_detect_lang(text) if text else ("unknown", 0.0)
            return {"text": text, "conf": conf, "lang": lang, "lang_conf": lang_conf}
        except Exception as e:
            print(f"❌ Lỗi TrOCR: {type(e).__name__} - {e}")
            traceback.print_exc()
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
        """
        Chọn text tốt nhất từ nhiều mô hình OCR,
        Ưu tiên tiếng Việt và lang_conf cao.
        """
        texts = [r for r in results.values() if r.get("text")]

        if not texts:
            return "(rỗng)", 0.0, "unknown"

        # Làm sạch và chuẩn hóa chữ
        for r in texts:
            r["text"] = normalize_case(clean_text(r["text"]))
            if not r.get("lang") or r["lang"] == "unknown":
                try:
                    lang, lang_conf = fasttext_detect_lang(r["text"])
                    r["lang"], r["lang_conf"] = lang, lang_conf
                except:
                    r["lang"], r["lang_conf"] = "unknown", 0.0

        # --- Ưu tiên text tiếng Việt ---
        vi_texts = [t for t in texts if t["lang"] == "vi" and t["lang_conf"] >= 0.6]

        if vi_texts:
            candidates = vi_texts
            print(f"🟩 Ưu tiên {len(candidates)} text tiếng Việt có lang_conf ≥ 0.6")
        else:
            # Nếu không có tiếng Việt đáng tin, chọn ngôn ngữ có lang_conf cao nhất
            max_lang_conf = max(t["lang_conf"] for t in texts)
            candidates = [t for t in texts if t["lang_conf"] >= max_lang_conf * 0.9]
            print(f"⚠️ Không có tiếng Việt rõ ràng → fallback top {len(candidates)} ngôn ngữ khác")

        has_space = [t for t in candidates if " " in t["text"]]
        if has_space:
            candidates = has_space
            print(f"🟦 Ưu tiên {len(candidates)} text có dấu cách rõ ràng")

        # --- Ưu tiên thêm theo confidence ---
        max_conf = max(t["conf"] for t in candidates)
        top_conf = [t for t in candidates if t["conf"] >= max_conf * 0.9]

        # --- Ưu tiên text có spacing hợp lý ---
        spaced = [t for t in top_conf if has_good_spacing(t["text"])]
        pool = spaced if spaced else top_conf

        # --- Ưu tiên text dài hơn (có nhiều từ hơn) ---
        pool = sorted(pool, key=lambda t: (len(t["text"].split()), t["conf"]), reverse=True)

        best = pool[0]
        print(f"✅ Chọn: {best['text']} ({best['conf']:.2f}) lang={best['lang']} lang_conf={best['lang_conf']:.2f}")

        return best["text"], float(best["conf"]), best["lang"]

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
                    "mmocr": executor.submit(timed_run, run_ocr_mmocr, crop, "mmocr"),
                    "trocr": executor.submit(timed_run, run_ocr_trocr, crop, "trocr")
                }
                results = {k: f.result() for k, f in futures.items()}

            results["paddleocr"] = {
                "text": rec_text,
                "conf": rec_conf,
                "lang": rec_lang,
                "lang_conf": rec_lang_conf,
                "runtime": 0.0,
                "model": "paddleocr"
            }
            # ---- Chọn text tốt nhất ----
            best_text, best_conf, best_lang = choose_best_text(results)
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
            best_text, best_conf, best_lang = choose_best_text(results)
            corr = corrector(best_text, max_length=MAX_LENGTH)
            best_text = corr[0]["generated_text"]
            best_text = map_vietnamese_to_schema(best_text)

            # color = (0, 255, 0) if best_text != "(rỗng)" else (0, 200, 255)
            # annotated = draw_unicode_text(annotated, best_text, (x_min, max(0, y_min - 18)), color)
            # cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

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
    # return " ".join(raw_text_list), ocr_texts, annotated
    return " ".join(map(str, raw_text_list)), ocr_texts

MAX_LENGTH = 512
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")

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

ID_TYPE_MAPPING: Dict[str, str] = {
    "CĂN CƯỚC CÔNG DÂN": "CCCD",
    "CCCD": "CCCD",
    "CHỨNG MINH NHÂN DÂN": "CMND",
    "CMND": "CMND",
    "HỘ CHIẾU": "Passport",
    "HỌ CHIẾU": "Passport", 
    "PASSPORT": "Passport",
}

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
        r"(ông|bà)\s+[A-ZÀÁẢÃẠÂĂĐÊÔƠƯ][\w\s]+",
    ],
    "appointment_date": [
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
    ],
    "signing_authority": [
        r"giám\s*đốc|tổng\s*giám\s*đốc|chủ\s*tịch|phó\s*giám\s*đốc",
    ],
}

def map_vietnamese_to_schema(best_text: str) -> Dict[str, Any]:
    """
    Hàm chuẩn hóa text OCR của Quyết định Bổ Nhiệm sang schema chuẩn hóa.
    - Nhận đầu vào: chuỗi OCR đã ghép toàn bộ văn bản.
    - Trích xuất thông tin: công ty, số quyết định, loại, người ký, người được bổ nhiệm, ngày tháng, hiệu lực, dấu, v.v.
    """
    text = best_text.upper()

    # ======================
    # 1️⃣ Khởi tạo cấu trúc schema rỗng
    # ======================
    normalized = {
        "company_info": {
            "company_name": "",
            "company_type": "",
            "decision_number": "",
            "decision_type": "",
            "issue_date": {"day": 0, "month": 0, "year": 0},      # ngày ký
            "effective_date": {"day": 0, "month": 0, "year": 0},  # ngày hiệu lực
            "has_valid_stamp": False
        },
        "signing_person": {
            "full_name": "",
            "position": "",
            "signature_detected": False,
            "authorization_rule": ""
        },
        "appointee": {
            "full_name": "",
            "id_number": "",
            "position": "",
            "document_ref": ""
        }
    }

    # ======================
    # 2️⃣ Nhận dạng tên công ty
    # ======================
    m_company = re.search(r"CÔNG\s*TY\s+([A-ZÀ-Ỹ0-9\s]+)", text)
    if m_company:
        normalized["company_info"]["company_name"] = "CÔNG TY " + m_company.group(1).strip()

    # Nhận dạng loại hình công ty (TNHH, CP, MTV,...)
    m_type = re.search(r"(TNHH|CỔ\s*PHẦN|MTV|MỘT\s*THÀNH\s*VIÊN|HTV|JSC|CO\s*LTD)", text)
    if m_type:
        normalized["company_info"]["company_type"] = m_type.group(1).replace(" ", "")

    # ======================
    # 3️⃣ Số và loại quyết định
    # ======================
    # Ví dụ: “Số: 15/QĐ-CTYABC”
    m_number = re.search(r"SỐ\s*[:\-]?\s*([0-9A-Z\/\-\_]+)", text)
    if m_number:
        normalized["company_info"]["decision_number"] = m_number.group(1).strip()

    # Loại quyết định: BỔ NHIỆM / MIỄN NHIỆM / ĐIỀU ĐỘNG
    m_decision_type = re.search(r"(BỔ NHIỆM|MIỄN NHIỆM|ĐIỀU ĐỘNG|PHÂN CÔNG|BỔ SUNG)", text)
    if m_decision_type:
        normalized["company_info"]["decision_type"] = m_decision_type.group(1)

    # ======================
    # 4️⃣ Ngày ký quyết định
    # ======================
    # Mẫu phổ biến: “Ngày 15 tháng 10 năm 2025”
    m_issue = re.search(r"NGÀY\s+(\d{1,2})\s+THÁNG\s+(\d{1,2})\s+NĂM\s+(\d{4})", text)
    if m_issue:
        normalized["company_info"]["issue_date"] = {
            "day": int(m_issue.group(1)),
            "month": int(m_issue.group(2)),
            "year": int(m_issue.group(3))
        }

    # ======================
    # 5️⃣ Ngày hiệu lực (nếu có)
    # ======================
    # Mẫu: “Có hiệu lực từ ngày 20/10/2025” hoặc “Hiệu lực kể từ ngày ...”
    m_effect = re.search(r"(HIỆU LỰC|CÓ HIỆU LỰC)\s*(KỂ TỪ)?\s*NGÀY\s*(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", text)
    if m_effect:
        normalized["company_info"]["effective_date"] = {
            "day": int(m_effect.group(3)),
            "month": int(m_effect.group(4)),
            "year": int(m_effect.group(5))
        }

    # ======================
    # 6️⃣ Người được bổ nhiệm
    # ======================
    # Ví dụ: “Bổ nhiệm Ông Nguyễn Văn A giữ chức vụ Kế toán trưởng”
    m_app = re.search(r"BỔ NHIỆM\s+(ÔNG|BÀ)\s+([A-ZÀ-Ỹ\s]+?)\s+(GIỮ|LÀM)\s+CHỨC\s*VỤ\s*[:\-]?\s*([A-ZÀ-Ỹ\s]+)", text)
    if m_app:
        normalized["appointee"]["full_name"] = m_app.group(2).strip()
        normalized["appointee"]["position"] = m_app.group(4).strip()

    # Nếu tách riêng hai bước (không có “giữ chức vụ”)
    else:
        m_app_name = re.search(r"(ÔNG|BÀ)\s+([A-ZÀ-Ỹ\s]{3,100})", text)
        if m_app_name:
            normalized["appointee"]["full_name"] = m_app_name.group(2).strip()

        m_app_pos = re.search(r"CHỨC\s*VỤ\s*[:\-]?\s*([A-ZÀ-Ỹ\s]{3,100})", text)
        if m_app_pos:
            normalized["appointee"]["position"] = m_app_pos.group(1).strip()

    # Lấy số giấy tờ / CCCD nếu có
    m_app_id = re.search(r"(CCCD|CMND|SỐ)\s*[:\-]?\s*(\d{9,12})", text)
    if m_app_id:
        normalized["appointee"]["id_number"] = m_app_id.group(2)

    # ======================
    # 7️⃣ Người ký quyết định
    # ======================
    # Ví dụ: “GIÁM ĐỐC TRẦN VĂN B” hoặc “Chủ tịch HĐQT Nguyễn Thị C”
    m_sign = re.search(r"(GIÁM ĐỐC|TỔNG GIÁM ĐỐC|CHỦ TỊCH|PHÓ GIÁM ĐỐC)\s+([A-ZÀ-Ỹ\s]+)", text)
    if m_sign:
        normalized["signing_person"]["position"] = m_sign.group(1).strip()
        normalized["signing_person"]["full_name"] = m_sign.group(2).strip()
        normalized["signing_person"]["signature_detected"] = True
        normalized["signing_person"]["authorization_rule"] = f"Người ký là {m_sign.group(1)}, có thẩm quyền ký văn bản hành chính."

    # ======================
    # 8️⃣ Kiểm tra dấu mộc (chỉ flag logic, chưa nhận diện hình ảnh)
    # ======================
    if "DẤU" in text or "CÔNG TY" in text and "TRÒN" in text:
        normalized["company_info"]["has_valid_stamp"] = True

    # ======================
    # ✅ Kết quả cuối cùng
    # ======================
    return normalized


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

def compute_image_hash(image_array):
    # Đảm bảo image_array là np.ndarray
    if isinstance(image_array, np.ndarray):
        # Chuyển ảnh sang bytes (encode PNG hoặc JPEG)
        success, img_bytes = cv2.imencode(".png", image_array)
        if success:
            img_bytes = img_bytes.tobytes()
        else:
            img_bytes = b""
    elif isinstance(image_array, (bytes, bytearray)):
        img_bytes = image_array
    else:
        raise TypeError(f"Không thể hash kiểu dữ liệu: {type(image_array)}")

    return "sha256:" + hashlib.sha256(img_bytes).hexdigest()

def build_appointment_decision_json(image_array, ocr_results, user_id, doc_id, collection_id):
    collection_id = str(uuid.uuid4())
    joined_text = " ".join(
        str(t.get("final_text", "")) if isinstance(t.get("final_text"), (str, int, float)) else ""
        for t in ocr_results if isinstance(t, dict)
    )
    print("🧩 Chuỗi OCR đã ghép:\n", joined_text)
    normalized = map_vietnamese_to_schema(joined_text)

    json_data = {
        "_id": f"dec_{doc_id}",
        "user_id": user_id,
        "image_hash": compute_image_hash(image_array),
        "ocr_raw": {...},
        "normalized": map_vietnamese_to_schema(joined_text),
        "status": "pending",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat()
    }
    return json_data
 
def main():
    path = r"/home/caokhoa/Documents/VP_Bank-Hackathon/QDBN/QDBN1.pdf"
    schema_path = "./schema.json"

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
            text, details = extract_text(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # lưu annotate
            # preview_path = f"annotated_page_{idx}.jpg"
            # cv2.imwrite(preview_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            # print(f"✅ Ảnh chú thích OCR trang {idx} đã lưu tại: {preview_path}")

            # build JSON
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
                    if isinstance(schema_data, list) and len(schema_data) > 0:
                        schema_data = schema_data[0]  # lấy phần tử đầu
                    collection_id = schema_data.get("_id", "collection_appointment_decisions")
                    print(f"📦 Đọc schema thành công: _id = {collection_id}")
            except Exception as e:
                print(f"⚠️ Không đọc được schema.json ({e}), dùng mặc định 'collection_appointment_decisions'")
                collection_id = "collection_appointment_decisions"

            json_data = build_appointment_decision_json(
                image_array=np.array(img),
                ocr_results=details,
                user_id="user_001",
                doc_id=f"dec_{base_name}_page{idx}",
                collection_id=collection_id
            )

            json_path = f"appointment_decision_{base_name}_page{idx}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã lưu JSON schema trang {idx} vào: {json_path}")
            
    else:
        print(f"⚠️ Không hỗ trợ định dạng: {ext}")

if __name__ == "__main__":
    main()
