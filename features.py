"""Offline features extracted entirely from URL strings."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit, parse_qsl

import numpy as np
import tldextract

# Bundled PSL only: reproducible offline extraction, no automatic network update.
EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
KEYWORDS = ('login', 'signin', 'verify', 'account', 'secure', 'update', 'bank', 'password')
LEXICAL_NAMES = [
    'url_length', 'host_length', 'domain_length', 'path_length', 'query_length',
    'fragment_length', 'dot_count', 'hyphen_count', 'underscore_count',
    'slash_count', 'at_count', 'percent_count', 'question_count', 'equal_count',
    'ampersand_count', 'digit_ratio', 'letter_ratio', 'digit_letter_ratio',
    'non_ascii_ratio', 'entropy', 'host_entropy', 'subdomain_count',
    'path_depth', 'query_param_count', 'host_is_ip', 'has_https_scheme',
    'has_explicit_port', 'has_userinfo', 'has_punycode', 'percent_encoded_count',
    'longest_digit_run', 'host_digit_ratio', 'host_hyphen_count',
] + ['keyword_' + k for k in KEYWORDS]


_SCHEME = re.compile(r'^(?:https?:)?//', re.IGNORECASE)  # other schemes stay and are rejected by parse_url
_WWW = re.compile(r'^www\d{0,3}\.', re.IGNORECASE)


def normalize_url(url):
    """Drop the scheme and a leading www. so models cannot learn them as shortcuts."""
    s = str(url).strip()
    out = _WWW.sub('', _SCHEME.sub('', s, count=1), count=1)
    return out or s


def parse_url(url):
    url = str(url).strip()
    # Decide on a leading scheme only; an embedded "://" in the query must not count as one.
    if re.match(r'^[A-Za-z][A-Za-z0-9+.\-]*://', url):
        candidate = url
    else:
        candidate = 'http:' + url if url.startswith('//') else 'http://' + url
    try:
        p = urlsplit(candidate)
        host = (p.hostname or '').rstrip('.').lower()
        host = host.encode('idna').decode('ascii')
        if not host or any(c.isspace() for c in host):
            raise ValueError('Invalid hostname')
        _ = p.port  # Reject malformed ports.
        if p.scheme.lower() not in ('http', 'https'):
            raise ValueError('Only HTTP(S) URLs supported')
        return p, host
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Cannot parse URL: {url[:120]}') from exc


def domain_of(url):
    _, host = parse_url(url)
    e = EXTRACT(host)
    return e.top_domain_under_public_suffix or host


def canonical_key(url):
    p, host = parse_url(url)
    # Preserve case-sensitive path/query/fragment and user information.
    return (p.scheme.lower(), host, p.port, p.username, p.password,
            p.path, p.query, p.fragment)


def entropy(s):
    if not s:
        return 0.0
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in Counter(s).values())


def lexical_features(url):
    url = str(url).strip()
    p, host = parse_url(url)
    e = EXTRACT(host)
    n = max(1, len(url))
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    vals = [len(url), len(host), len(e.domain), len(p.path), len(p.query), len(p.fragment)]
    vals += [url.count(c) for c in '.-_/ @%?=&'.replace(' ', '')]
    vals += [digits / n, letters / n, digits / max(1, letters),
             sum(ord(c) > 127 for c in url) / n, entropy(url), entropy(host),
             len(e.subdomain.split('.')) if e.subdomain else 0,
             len([x for x in p.path.split('/') if x]),
             len(parse_qsl(p.query, keep_blank_values=True)), is_ip,
             int(p.scheme.lower() == 'https' and '://' in url),
             int(p.port is not None), int(p.username is not None),
             int(any(x.startswith('xn--') for x in host.split('.'))),
             len(re.findall(r'%[0-9a-fA-F]{2}', url)),
             max((len(x) for x in re.findall(r'\d+', url)), default=0),
             sum(c.isdigit() for c in host) / max(1, len(host)), host.count('-')]
    vals += [int(k in url.lower()) for k in KEYWORDS]
    assert len(vals) == len(LEXICAL_NAMES)
    return vals


def feature_matrix(urls):
    rows = [lexical_features(u) for u in urls]
    return np.asarray(rows, dtype=np.float32), LEXICAL_NAMES
