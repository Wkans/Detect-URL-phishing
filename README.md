# Detect-URL-phishing

Phát hiện URL giả mạo (phishing) bằng mô hình lai **CNN-LSTM + Random Forest**.

- Nhánh học sâu: CNN-LSTM đọc chuỗi tối đa 512 ký tự của URL và cho vector đặc trưng 64 chiều.
- Nhánh thủ công: 41 đặc trưng tính từ cấu trúc URL.
- Hai vector được nối thành 105 chiều và đưa vào Random Forest (500 cây). URL bị gắn nhãn giả mạo khi điểm ≥ 0,545.
- Lớp luật bổ trợ (`typosquat.py`) nâng điểm lên tối thiểu 0,9 khi phát hiện giả mạo thương hiệu và hạ điểm xuống tối đa 0,2 với host chính chủ tin cậy.

Đồ án chỉ phân tích chuỗi URL, không truy cập nội dung trang web, không dùng WHOIS hay thông tin chuyển hướng.

## Kết quả (mô hình `runs/url_only_v10`)

Trên tập kiểm tra gồm 125.934 URL (tách theo tên miền):

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 93,69% | 96,05% | 87,85% | 91,77% | 0,97718 |

Trên tập độc lập 600 URL (`test_urls_public2.csv`, 300 hợp lệ và 300 giả mạo), F1-score khoảng 0,80 và FPR khoảng 18%, thấp hơn rõ so với tập kiểm tra.

## Cấu trúc

| Tệp | Vai trò |
|---|---|
| `common.py` | Nạp và làm sạch dữ liệu, chia tập theo tên miền, từ điển ký tự, chọn ngưỡng, chỉ số |
| `features.py` | Chuẩn hóa URL, phân tách thành phần, 41 đặc trưng thủ công |
| `model.py` | Mạng CNN-LSTM và bộ trích xuất `deep_features` |
| `train.py` | Huấn luyện mạng và Random Forest, lưu vào `runs/` |
| `tune.py` | Tìm siêu tham số bằng Optuna |
| `predict.py` | Dự đoán một URL hoặc tệp CSV từ dòng lệnh |
| `typosquat.py`, `brands.json`, `trusted.json` | Lớp luật giả mạo thương hiệu và host tin cậy |
| `build_brands.py` | Tạo `brands.json` từ danh sách thương hiệu và dữ liệu |
| `api.py` | Dịch vụ REST (FastAPI) |
| `app.py` | Giao diện web (Streamlit) |
| `demo.py` | Ví dụ gọi API bằng `requests` |
| `test_pipeline.py` | 11 kiểm thử đơn vị chạy offline |
| `runs/url_only_v10/` | Mô hình đã huấn luyện và kết quả |
| `runs/tune_v10/` | Kết quả tinh chỉnh siêu tham số |

## Cài đặt

Khuyến nghị Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tệp mô hình `runs/url_only_v10/TN4_Hybrid_RF.joblib` (khoảng 188 MB) được lưu bằng **Git LFS**. Cài [Git LFS](https://git-lfs.com) và chạy `git lfs install` trước khi `git clone` (hoặc `git lfs pull` sau khi clone).

## Dự đoán

```powershell
# một URL
.\.venv\Scripts\python.exe predict.py --model-dir runs/url_only_v10 --url "https://example.com" --out prediction_example.csv

# nhiều URL từ tệp CSV có cột url
.\.venv\Scripts\python.exe predict.py --model-dir runs/url_only_v10 --csv input_urls.csv --out predictions_batch.csv
```

## Dịch vụ API và giao diện web

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app          # http://127.0.0.1:8000/docs
.\.venv\Scripts\python.exe -m streamlit run app.py     # giao diện web
```

Các đường dẫn API: `GET /health`, `GET /models`, `POST /predict`, `POST /predict/batch`. Ví dụ gọi từ Python: xem `demo.py`. Dịch vụ chưa có xác thực và chưa bật CORS.

## Huấn luyện lại

Cần tệp dữ liệu có hai cột `url`, `label` (nhãn 0 là hợp lệ, nhãn 1 là giả mạo). Tệp `data/urls.csv` dùng trong đồ án không nằm trong repo; `data/urls_raw.csv` là dữ liệu thô đã có sẵn.

```powershell
.\.venv\Scripts\python.exe train.py --data data/urls.csv --out runs/url_only_v10 --max-per-domain 200 --embedding-dim 128 --filters 128 --kernel-size 3 --lstm-units 64 --deep-dim 64 --batch-size 128 --learning-rate 0.00138711 --dropout 0.3315 --trees 500 --rf-max-depth 24 --rf-min-leaf 2 --rf-max-features log2
```

Các tham số khác (`--split domain`, `--seed 42`, `--max-len 512`, `--epochs 15`) dùng giá trị mặc định. Giá trị tốt nhất của bước tinh chỉnh nằm trong `runs/tune_v10/best_params.json`. Tinh chỉnh siêu tham số bằng `tune.py` cần cài thêm `optuna` (đã có trong `requirements.txt`).

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest test_pipeline -v
```
