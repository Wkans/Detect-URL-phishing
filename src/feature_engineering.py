"""
feature_engineering.py
-----------------------
Trích xuất bộ đặc trưng thủ công (hand-crafted features) từ URL.
Đây là "Nhánh 2" trong kiến trúc lai: cung cấp thông tin thống kê
mà bản thân chuỗi ký tự thô không thể hiện đầy đủ.

Nhóm đặc trưng:
- Lexical: độ dài, số ký tự đặc biệt, tỷ lệ số/chữ, entropy...
- Domain-based: số subdomain, có dùng IP thay domain không, độ dài domain...
(Đặc trưng WHOIS/domain age được tách riêng vì cần gọi mạng, xem whois_features.py)
"""

import re
import math
from urllib.parse import urlparse
import pandas as pd
import tldextract


SPECIAL_CHARS = ['-', '_', '.', '/', '@', '?', '&', '=', '%', '#', '~']


def shannon_entropy(s: str) -> float:
    """Tính entropy Shannon của chuỗi - URL độc hại thường có entropy bất thường
    do chèn ký tự ngẫu nhiên để né phát hiện."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def has_ip_address(url: str) -> int:
    """Kiểm tra URL có dùng địa chỉ IP thay vì tên miền không (dấu hiệu đáng ngờ)."""
    pattern = r'(\d{1,3}\.){3}\d{1,3}'
    return 1 if re.search(pattern, url) else 0


def count_char(s: str, ch: str) -> int:
    return s.count(ch)


def extract_features(url: str) -> dict:
    """
    Trích xuất toàn bộ đặc trưng thủ công cho một URL.

    Nếu URL có lỗi cú pháp, hàm không làm chương trình crash
    mà trả về các feature cơ bản.
    """

    url = str(url).strip()

    # Thêm scheme nếu chưa có
    normalized_url = (
        url if "://" in url
        else "http://" + url
    )

    # =========================
    # Parse URL an toàn
    # =========================

    try:

        parsed = urlparse(normalized_url)

    except (ValueError, TypeError):

        # URL lỗi
        parsed = None

    # =========================
    # Nếu parse thất bại
    # =========================

    if parsed is None:

        return {
            "url_length": len(url),
            "hostname_length": 0,
            "path_length": 0,

            "num_dots": count_char(url, "."),
            "num_hyphens": count_char(url, "-"),
            "num_underscore": count_char(url, "_"),
            "num_slash": count_char(url, "/"),
            "num_question": count_char(url, "?"),
            "num_equal": count_char(url, "="),
            "num_at": count_char(url, "@"),
            "num_percent": count_char(url, "%"),

            "num_digits": sum(
                c.isdigit() for c in url
            ),

            "num_letters": sum(
                c.isalpha() for c in url
            ),

            "digit_letter_ratio": (
                sum(c.isdigit() for c in url)
                /
                max(
                    sum(c.isalpha() for c in url),
                    1
                )
            ),

            "num_special_chars": sum(
                url.count(c)
                for c in SPECIAL_CHARS
            ),

            "url_entropy": shannon_entropy(url),

            "has_ip": has_ip_address(url),

            "has_https": (
                1 if url.lower().startswith("https://")
                else 0
            ),

            "has_at_symbol": (
                1 if "@" in url
                else 0
            ),

            "has_double_slash_redirect": (
                1 if url.rfind("//") > 6
                else 0
            ),

            "domain_length": 0,
            "num_subdomains": 0,
            "is_shortened": 0,
        }

    # =========================
    # TLD Extract
    # =========================

    try:

        ext = tldextract.extract(url)

        domain = (
            ext.domain + "." + ext.suffix
            if ext.suffix
            else ext.domain
        )

        subdomain = ext.subdomain

    except Exception:

        domain = ""
        subdomain = ""

    # =========================
    # Hand-crafted features
    # =========================

    features = {

        # ----- Lexical features -----

        "url_length": len(url),

        "hostname_length": len(
            parsed.netloc
        ),

        "path_length": len(
            parsed.path
        ),

        "num_dots": count_char(
            url, "."
        ),

        "num_hyphens": count_char(
            url, "-"
        ),

        "num_underscore": count_char(
            url, "_"
        ),

        "num_slash": count_char(
            url, "/"
        ),

        "num_question": count_char(
            url, "?"
        ),

        "num_equal": count_char(
            url, "="
        ),

        "num_at": count_char(
            url, "@"
        ),

        "num_percent": count_char(
            url, "%"
        ),

        "num_digits": sum(
            c.isdigit() for c in url
        ),

        "num_letters": sum(
            c.isalpha() for c in url
        ),

        "digit_letter_ratio": (
            sum(c.isdigit() for c in url)
            /
            max(
                sum(c.isalpha() for c in url),
                1
            )
        ),

        "num_special_chars": sum(
            url.count(c)
            for c in SPECIAL_CHARS
        ),

        "url_entropy": shannon_entropy(
            url
        ),

        "has_ip": has_ip_address(
            url
        ),

        "has_https": (
            1
            if parsed.scheme.lower() == "https"
            else 0
        ),

        "has_at_symbol": (
            1
            if "@" in url
            else 0
        ),

        "has_double_slash_redirect": (
            1
            if url.rfind("//") > 6
            else 0
        ),

        # ----- Domain features -----

        "domain_length": len(
            domain
        ),

        "num_subdomains": (
            len(subdomain.split("."))
            if subdomain
            else 0
        ),

        "is_shortened": (
            1
            if ext.domain in [
                "bit",
                "tinyurl",
                "goo",
                "ow",
                "t",
                "is",
                "buff",
                "adf"
            ]
            else 0
        ),
    }

    return features


def build_feature_dataframe(urls: pd.Series) -> pd.DataFrame:
    """Áp dụng extract_features cho toàn bộ danh sách URL, trả về DataFrame."""
    records = [extract_features(u) for u in urls]
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Test nhanh module
    test_urls = [
        "http://192.168.1.1/login/verify-account.php",
        "https://www.google.com/search?q=test",
        "http://secure-paypal-login.tk/update-info?user=abc",
    ]
    df = build_feature_dataframe(pd.Series(test_urls))
    print(df.T)
