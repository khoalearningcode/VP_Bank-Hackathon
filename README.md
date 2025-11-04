
# 📄 Multi-OCR Vietnamese Document Processor

> **Dự án OCR đa mô hình cho văn bản hành chính tiếng Việt**, tích hợp nhiều engine OCR (PaddleOCR, EasyOCR, Tesseract, MMOCR, TrOCR), tự động phát hiện vùng văn bản, chọn kết quả tốt nhất, sửa lỗi chính tả tiếng Việt và trích xuất thông tin theo schema (tên công ty, CCCD, người ký, ngày bổ nhiệm,...).

---

## 🔧 Yêu cầu hệ thống

- **Hệ điều hành**: Linux / macOS / Windows (khuyến khích Linux cho hiệu năng tốt)
- **Python**: `>= 3.8, < 3.11`
- **GPU (tùy chọn)**: CUDA-enabled GPU để tăng tốc (EasyOCR, TrOCR, MMOCR hỗ trợ GPU)

---

## 🚀 Cài đặt nhanh (Quick Setup)

### 1. Clone mã nguồn

```bash
git clone https://github.com/your-username/vietnamese-ocr-processor.git
cd vietnamese-ocr-processor
```


---

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc
venv\Scripts\activate           # Windows
```

---

### 3. Cài đặt các phụ thuộc

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý quan trọng**: Một số thư viện yêu cầu cài đặt thủ công trước (xem phần "Cài đặt nâng cao" bên dưới nếu gặp lỗi).

---

### 4. Tải mô hình ngôn ngữ (FastText)

Khi chạy lần đầu, script sẽ tự **tải mô hình phát hiện ngôn ngữ** `lid.176.ftz` từ Facebook Research (~100MB). Đảm bảo máy có kết nối Internet.

---

### 5. Chuẩn bị font hỗ trợ tiếng Việt (tuỳ chọn nhưng khuyến nghị)

Tải font `Roboto-Black.ttf` và đặt vào thư mục gốc của dự án để hiển thị chữ có dấu khi vẽ kết quả OCR:

```bash
wget -O Roboto-Black.ttf https://github.com/googlefonts/roboto/raw/main/fonts/static/Roboto-Black.ttf
```

> Nếu không có, hệ thống sẽ dùng font mặc định (có thể bị lỗi hiển thị tiếng Việt).

---

### 6. Chạy thử

Đặt file PDF hoặc ảnh vào thư mục, ví dụ: `./QDBN1.pdf`

```bash
python main.py
```

- Output: Tạo file JSON như `appointment_decision_QDBN1_page1.json`
- Log: Hiển thị tiến trình xử lý từng trang, kết quả OCR từ từng mô hình

---

## 📦 Cấu trúc dự án

```
.
├── VP_BankHackathonModel.py                  # Entry point
├── requirements.txt         # Danh sách phụ thuộc
├── Roboto-Black.ttf         # Font hỗ trợ tiếng Việt (optional)
├── QDBN1.pdf                # File mẫu để test
└── schema.json              # Schema mẫu (nếu có)
```

---

## ⚙️ Cài đặt nâng cao (nếu gặp lỗi)

Một số thư viện yêu cầu cài đặt hệ thống trước:

### Trên Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev poppler-utils libgl1 libglib2.0-0
```

### Trên macOS (dùng Homebrew):

```bash
brew install tesseract poppler
```

### Trên Windows:

- Cài [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) và thêm vào `PATH`
- Cài [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) và thêm `bin/` vào `PATH`

---

### Cài đặt thủ công một số package (nếu `pip install` thất bại):

```bash
# OpenCV
pip install opencv-python-headless

# MMOCR (rất quan trọng!)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmengine
pip install mmocr

# PaddleOCR
pip install paddlepaddle  # hoặc paddlepaddle-gpu nếu có GPU

# PDF -> Image
pip install pdf2image

# Mô hình ngôn ngữ
pip install fasttext

# TrOCR
pip install transformers torch

# EasyOCR
pip install easyocr

# Tesseract Python wrapper
pip install pytesseract
```

> Thay `cu118` bằng version CUDA phù hợp (xem [MMCV official](https://mmcv.readthedocs.io/en/latest/get_started/installation.html))

---

## 📝 Cách tùy chỉnh

### 1. Thay đổi file đầu vào

Sửa dòng sau trong `main.py`:

```python
path = r"./QDBN1.pdf"
```

→ Thay bằng đường dẫn file PDF hoặc ảnh của bạn.

### 2. Thay đổi schema đầu ra

Chỉnh sửa hàm `map_vietnamese_to_schema()` trong `main.py` để phù hợp với business logic của bạn.

### 3. Tắt một số mô hình OCR

Trong hàm `extract_text()`, comment dòng tương ứng trong `futures`:

```python
# "mmocr": executor.submit(...),
```

→ Giúp tăng tốc nếu không cần mô hình đó.

---

## 📤 Output mẫu

File JSON sẽ có cấu trúc như sau:

```json
{
  "_id": "uuid-generated",
  "public": {
    "node_data": {
      "jsonSchema": {
        "normalized": {
          "company_name": "CÔNG TY TNHH ABC",
          "personal_info": {
            "id_type": "CCCD",
            "id_number": "012345678901",
            "full_name": "NGUYỄN VĂN A"
          },
          "appointment_date": { "day": 15, "month": 6, "year": 2024 },
          "signing_authority": "GIÁM ĐỐC",
          "signing_person": { ... }
        },
        "user_id": "user_001",
        "doc_id": "dec_QDBN1_page1",
        "created_at": "2024-06-15T10:30:00.000Z"
      }
    }
  }
}
```

---
