"""Rule layer on top of the model: typosquatting and brand-impersonation checks (string only, no network).

check_url(url) returns None or {'rule', 'brand', 'reason'}. Callers raise the phishing score to RULE_SCORE on a hit.
Brands come from brands.json (build it with build_brands.py). Official hosts that the model over-flags come from
trusted.json; for them the score is capped at TRUSTED_SCORE ('trusted' rule).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from features import EXTRACT, normalize_url, parse_url

BRANDS_FILE = Path(__file__).with_name('brands.json')
RULE_SCORE = 0.9
TRUSTED_SCORE = 0.2
TRUSTED_FILE = Path(__file__).with_name('trusted.json')
FUZZY_MIN_LEN = 5       # edit-distance matching only for brands/labels with at least this many letters
SHARED_HOSTS = {'github'}  # sites whose subdomains are routinely used by the brand owners themselves
_LEET = str.maketrans({'0': 'o', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '$': 's', '@': 'a'})
_IP = re.compile(r'[\d.:\[\]]+')
_EMBEDS_URL = re.compile(r'https?(:|%3a)|%2f%2f|[?&](url|u|q|redirect|continue|next|target)=', re.I)


@lru_cache(maxsize=1)
def default_brands():
    data = json.loads(BRANDS_FILE.read_text(encoding='utf-8'))
    trusted = json.loads(TRUSTED_FILE.read_text(encoding='utf-8')) if TRUSTED_FILE.exists() else None
    return make_brands(data['brands'], data.get('legit_labels', []), trusted)


def make_brands(brands, legit_labels=(), trusted=None):
    brands = set(brands)
    by_len = {}
    for b in brands:
        by_len.setdefault(len(b), []).append(b)
    t = trusted or {}
    return {'brands': brands, 'legit': set(legit_labels), 'by_len': by_len,
            'trusted_hosts': set(t.get('hosts', [])), 'trusted_root': set(t.get('root_only_hosts', [])),
            'trusted_domains': tuple(t.get('domains', []))}


def _trusted(host, p, b):
    """True for official hosts on the trusted list, unless the URL embeds another URL (redirect abuse)."""
    rest = (p.path or '') + '?' + (p.query or '')
    if _EMBEDS_URL.search(rest) or '//' in (p.path or ''):
        return False
    if host in b['trusted_root']:
        return not (p.path.strip('/') or p.query or p.fragment)
    if host in b['trusted_hosts']:
        return True
    return any(host == d or host.endswith('.' + d) for d in b['trusted_domains'])


def _variants(label):
    """Look-alike spellings of a label: digits/symbols to letters, rn->m, vv->w ('1' may be l or i)."""
    base = label.translate(_LEET)
    out = set()
    for v in (base.replace('1', 'l'), base.replace('1', 'i')):
        out.update({v, v.replace('rn', 'm'), v.replace('vv', 'w')})
    out.discard(label)
    return out


def _one_edit(a, b):
    """True if a and b differ by exactly one insertion, deletion or substitution."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] if len(a) == len(b) else a[i:] == b[i + 1:]


def check_url(url, brands=None):
    b = brands or default_brands()
    try:
        p, host = parse_url(normalize_url(url))
    except ValueError:
        return None
    if _IP.fullmatch(host):
        return None
    if _trusted(host, p, b):
        return {'rule': 'trusted', 'brand': host,
                'reason': f'Host "{host}" nằm trong danh sách tên miền chính chủ đáng tin cậy'}
    e = EXTRACT(host)
    label = e.domain.lower()
    if not label:
        return None
    sub_tokens = {t for part in e.subdomain.lower().split('.') for t in part.split('-') if t}
    domain_tokens = [t for t in label.split('-') if t]

    # 1) brand name used in a subdomain of someone else's domain (login.paypal.com.evil.xyz)
    if label not in SHARED_HOSTS and label not in b['brands']:
        for brand in sorted(sub_tokens & b['brands']):
            if label != brand:
                return {'rule': 'subdomain', 'brand': brand,
                        'reason': f'Tên thương hiệu "{brand}" nằm ở subdomain của tên miền khác ({e.top_domain_under_public_suffix})'}
    if label in b['brands'] or label in b['legit']:
        return None
    # 2) look-alike spelling of a brand (goog1e, paypa1, rnicrosoft)
    for v in _variants(label):
        if v in b['brands']:
            return {'rule': 'homoglyph', 'brand': v,
                    'reason': f'Tên miền "{label}" giả dạng thương hiệu "{v}" bằng cách đổi ký tự giống nhau'}
    # 3) one-character edit of a brand (gooogle, paypall)
    if len(label) >= FUZZY_MIN_LEN:
        for n in (len(label) - 1, len(label), len(label) + 1):
            for brand in b['by_len'].get(n, ()):
                if len(brand) >= FUZZY_MIN_LEN and _one_edit(label, brand):
                    return {'rule': 'edit1', 'brand': brand,
                            'reason': f'Tên miền "{label}" chỉ khác thương hiệu "{brand}" một ký tự'}
    # 4) brand name glued to other words in the domain (paypal-secure-login.com)
    if len(domain_tokens) >= 2:
        for token in domain_tokens:
            for brand in sorted(({token} | _variants(token)) & b['brands']):
                return {'rule': 'brand_token', 'brand': brand,
                        'reason': f'Tên miền "{label}" ghép tên thương hiệu "{brand}" với từ khác'}
    return None


def apply_rule(score, hit):
    """Final phishing score: raised to RULE_SCORE when a phishing rule fires, capped at TRUSTED_SCORE for trusted hosts."""
    if not hit:
        return score
    return min(score, TRUSTED_SCORE) if hit['rule'] == 'trusted' else max(score, RULE_SCORE)
