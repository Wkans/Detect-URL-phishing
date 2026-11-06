import requests

API = 'http://127.0.0.1:8000'

r = requests.post(f'{API}/predict', json={'url': 'https://goog1e.com'}, timeout=30)
print(r.status_code, r.json()['class_name'], r.json()['phishing_score'])

urls = ['https://goog1e.com', 'https://docs.github.com/en/actions']
r = requests.post(f'{API}/predict/batch', json=urls and {'urls': urls}, timeout=30)
for x in r.json():
    print(x['url'], x['class_name'], x['phishing_score'])
