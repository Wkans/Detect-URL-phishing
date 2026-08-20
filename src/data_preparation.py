"""
data_preparation.py
-------------------
Đọc và làm sạch dataset URL:
- Kiểm tra file và cột dữ liệu
- Loại missing
- Kiểm tra label 0/1
- Chuẩn hóa URL
- Kiểm tra URL hợp lệ
- Loại URL trùng
- Kiểm tra phân bố lớp
- Chia Train / Validation / Test
"""

import os
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DATA_PATH, RANDOM_STATE


def normalize_url(url: str) -> str:
    """
    Chuẩn hóa URL ở mức cơ bản.
    Không thay đổi cấu trúc URL quá mạnh vì URL là dữ liệu đầu vào của mô hình.
    """
    url = str(url).strip()

    # Loại khoảng trắng đầu/cuối
    url = url.strip()

    # Chuyển về chữ thường
    url = url.lower()

    # Nếu chưa có scheme thì thêm http:// để kiểm tra
    if "://" not in url:
        url = "http://" + url

    return url


def is_valid_url(url: str) -> bool:
    """
    Kiểm tra URL có cấu trúc cơ bản hợp lệ hay không.
    Có bắt lỗi ValueError, đặc biệt với URL IPv6 lỗi.
    """
    try:
        parsed = urlparse(url)

        # Phải có hostname
        if not parsed.hostname:
            return False

        return True

    except (ValueError, TypeError):
        return False


def load_and_clean_data(path=RAW_DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. "
            f"Hãy đặt file urls_raw.csv vào thư mục data/ "
            f"với 2 cột: url,label"
        )

    # =========================
    # 1. Đọc dữ liệu
    # =========================
    df = pd.read_csv(path)

    # Kiểm tra cột
    required_columns = {"url", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "File CSV phải có đúng các cột cần thiết: url, label"
        )

    print("===== DATASET BAN ĐẦU =====")
    print("Tổng số mẫu:", len(df))

    # =========================
    # 2. Loại missing
    # =========================
    before = len(df)

    df = df.dropna(subset=["url", "label"])

    print(
        f"Loại {before - len(df)} mẫu bị thiếu URL hoặc label"
    )

    # =========================
    # 3. Chuẩn hóa URL
    # =========================
    df["url"] = df["url"].astype(str).str.strip()

    # Loại URL rỗng
    df = df[df["url"].str.len() > 0]

    # Chuẩn hóa
    df["url"] = df["url"].apply(normalize_url)

    # =========================
    # 4. Kiểm tra label
    # =========================

    # Chuyển label sang số
    try:
        df["label"] = pd.to_numeric(df["label"])
    except Exception:
        raise ValueError("Label không thể chuyển sang số.")

    # Kiểm tra label chỉ được 0 hoặc 1
    invalid_labels = ~df["label"].isin([0, 1])

    if invalid_labels.any():

        print("\nCác label không hợp lệ:")
        print(df.loc[invalid_labels, "label"].value_counts())

        raise ValueError(
            "Dataset chứa label không hợp lệ. "
            "Label chỉ được phép là 0 hoặc 1."
        )

    df["label"] = df["label"].astype(int)

    # =========================
    # 5. Kiểm tra URL hợp lệ
    # =========================

    print("\nĐang kiểm tra URL...")

    valid_mask = df["url"].apply(is_valid_url)

    invalid_count = (~valid_mask).sum()

    print("Số URL không hợp lệ:", invalid_count)

    if invalid_count > 0:

        print("\nMột số URL lỗi:")

        print(
            df.loc[~valid_mask, "url"]
            .head(10)
            .to_string(index=False)
        )

        # Loại URL lỗi
        df = df[valid_mask].copy()

    # =========================
    # 6. Loại URL trùng
    # =========================

    before = len(df)

    df = df.drop_duplicates(subset=["url"])

    duplicate_count = before - len(df)

    print("Số URL trùng đã loại:", duplicate_count)

    # =========================
    # 7. Kiểm tra dữ liệu
    # =========================

    if len(df) == 0:
        raise ValueError(
            "Không còn dữ liệu sau khi làm sạch."
        )

    print("\n===== DATASET SAU KHI LÀM SẠCH =====")

    print("Tổng số mẫu:", len(df))

    print("\nPhân bố nhãn:")

    print(df["label"].value_counts())

    print("\nTỷ lệ nhãn (%):")

    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size=0.2,
    val_size=0.15
):
    """
    Chia dữ liệu thành:
        Train / Validation / Test

    Sử dụng stratify để giữ tỷ lệ benign/phishing.
    """

    # =========================
    # Train + Validation / Test
    # =========================

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # =========================
    # Train / Validation
    # =========================

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label"],
        random_state=RANDOM_STATE
    )

    print("\n===== CHIA DATASET =====")

    print(f"Train: {len(train)}")
    print(f"Val  : {len(val)}")
    print(f"Test : {len(test)}")

    print("\nPhân bố Train:")
    print(train["label"].value_counts())

    print("\nPhân bố Val:")
    print(val["label"].value_counts())

    print("\nPhân bố Test:")
    print(test["label"].value_counts())

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True)
    )


if __name__ == "__main__":

    df = load_and_clean_data()

    train, val, test = split_data(df)