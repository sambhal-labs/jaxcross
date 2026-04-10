# REST API Patterns Reference

## Authentication

### API Key (query param)
```python
params = {"api_key": os.environ["API_KEY"], "format": "json"}
response = requests.get(url, params=params)
```

### API Key (header)
```python
headers = {"X-API-Key": os.environ["API_KEY"]}
response = requests.get(url, headers=headers)
```

### Bearer Token
```python
headers = {"Authorization": f"Bearer {os.environ['TOKEN']}"}
response = requests.get(url, headers=headers)
```

### OAuth 2.0 Client Credentials
```python
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

client = BackendApplicationClient(client_id=os.environ["CLIENT_ID"])
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(
    token_url="https://api.example.com/oauth/token",
    client_id=os.environ["CLIENT_ID"],
    client_secret=os.environ["CLIENT_SECRET"],
)
response = oauth.get("https://api.example.com/data")
```

## Pagination Patterns

### Page Number
```python
all_data = []
page = 1
while True:
    resp = requests.get(url, params={"page": page, "per_page": 100})
    data = resp.json()
    if not data["results"]:
        break
    all_data.extend(data["results"])
    page += 1
```

### Cursor-Based
```python
all_data = []
cursor = None
while True:
    params = {"limit": 100}
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(url, params=params)
    data = resp.json()
    all_data.extend(data["results"])
    cursor = data.get("next_cursor")
    if not cursor:
        break
```

### Link Header (GitHub-style)
```python
import re

all_data = []
next_url = url
while next_url:
    resp = requests.get(next_url)
    all_data.extend(resp.json())
    link = resp.headers.get("Link", "")
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    next_url = match.group(1) if match else None
```

### Offset-Based
```python
all_data = []
offset = 0
limit = 100
while True:
    resp = requests.get(url, params={"offset": offset, "limit": limit})
    data = resp.json()
    if not data:
        break
    all_data.extend(data)
    offset += limit
```

## Rate Limiting

### Exponential Backoff
```python
import time

def fetch_with_retry(url, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        response = requests.get(url, **kwargs)
        if response.status_code == 429:
            wait = 2 ** attempt
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait = int(retry_after)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response
    raise Exception(f"Max retries ({max_retries}) exceeded for {url}")
```

### Respect Rate Limit Headers
```python
remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
if remaining < 5:
    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
    sleep_time = max(0, reset_time - time.time()) + 1
    time.sleep(sleep_time)
```

## Flattening Nested JSON

```python
# Nested JSON like {"user": {"name": "Alice", "address": {"city": "NYC"}}, "amount": 100}
df = pd.json_normalize(records, sep="_")
# Result columns: user_name, user_address_city, amount

# Deeply nested with arrays
df = pd.json_normalize(
    records,
    record_path=["items"],        # Explode this array
    meta=["order_id", "date"],    # Keep these parent fields
    sep="_",
)
```

## Error Handling

```python
def safe_fetch(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Timeout fetching {url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error {e.response.status_code}: {e}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error for {url}")
        return None
    except ValueError:
        print(f"Invalid JSON response from {url}")
        return None
```
