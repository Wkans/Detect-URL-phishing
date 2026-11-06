"""Build brands.json: commonly impersonated brands (curated) + domain labels known to be legitimate (from the data).
Run: python build_brands.py --data data/urls.csv
"""
import argparse
import json
from pathlib import Path

from common import load_data

# Curated list of brands that phishers often imitate. Edit this list to add or remove brands.
BRANDS = """
google gmail youtube facebook instagram whatsapp messenger twitter linkedin pinterest snapchat tiktok telegram discord
reddit microsoft outlook office365 onedrive sharepoint skype hotmail windows apple icloud itunes amazon netflix spotify
paypal ebay aliexpress alibaba taobao baidu yahoo dropbox adobe docusign zoom steam steampowered roblox epicgames github
dhl fedex usps canadapost royalmail wetransfer binance coinbase kraken metamask blockchain trustwallet opensea uniswap
kucoin bybit wellsfargo bankofamerica citibank hsbc barclays santander lloyds natwest halifax americanexpress mastercard
visa capitalone usbank scotiabank commerzbank deutschebank postbank sparkasse
rakuten mufg smbc mizuho saison sagawa yamato docomo softbank aeon epos nicos orico mercari
vietcombank techcombank vietinbank agribank bidv mbbank sacombank tpbank vpbank momo zalopay shopee lazada tiki viettel
vnpt mobifone
shopify wordpress godaddy namecheap cloudflare booking airbnb ikea walmart costco bestbuy homedepot verizon comcast
xfinity tmobile vodafone telstra
""".split()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/urls.csv')
    ap.add_argument('--out', default='brands.json')
    args = ap.parse_args()
    df, _ = load_data(args.data)
    df = df[~df.domain.str.fullmatch(r'[\d.:\[\]]+')].copy()
    df['name'] = df.domain.str.split('.').str[0]
    stats = df.groupby('name').label.agg(['size', 'mean'])
    legit_n = df[df.label == 0].groupby('name').size()
    # A label is "known legitimate" if it has several legitimate rows and is not mostly phishing.
    legit_labels = sorted(n for n in legit_n.index if legit_n[n] >= 3 and stats.loc[n, 'mean'] < 0.5)
    brands = sorted(set(BRANDS))
    Path(args.out).write_text(json.dumps({'brands': brands, 'legit_labels': legit_labels}, ensure_ascii=False), encoding='utf-8')
    print(f'brands: {len(brands)}, known legitimate labels: {len(legit_labels)} -> {args.out}')


if __name__ == '__main__':
    main()
