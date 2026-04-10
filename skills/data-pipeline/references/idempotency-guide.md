# Idempotency Guide

## What is idempotency?

An idempotent pipeline produces the same output regardless of how many times it runs. This is critical for production pipelines where reruns are common (crash recovery, scheduling overlap, manual re-execution).

## Config Hash Caching

The simplest approach: hash the config file and skip if unchanged.

```python
import hashlib
import json

def config_hash(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

def is_cached(config: dict, output_dir: str) -> bool:
    meta_path = os.path.join(output_dir, "prepared_metadata.json")
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("config_hash") == config_hash(config)
```

## Source Data Hashing

For local files, hash the source data to detect changes:

```python
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

## API Source Freshness

For API sources, use HTTP caching headers:

```python
import requests

def fetch_if_modified(url: str, cache_path: str) -> bool:
    headers = {}
    if os.path.exists(cache_path + ".etag"):
        with open(cache_path + ".etag") as f:
            headers["If-None-Match"] = f.read()
    
    response = requests.get(url, headers=headers)
    if response.status_code == 304:
        return False  # Not modified
    
    # Save response
    with open(cache_path, "wb") as f:
        f.write(response.content)
    if "ETag" in response.headers:
        with open(cache_path + ".etag", "w") as f:
            f.write(response.headers["ETag"])
    return True
```

## Atomic Writes

Prevent partial outputs on crash:

```python
import tempfile
import shutil

def atomic_write(data_path: str, write_fn):
    """Write to a temp file, then rename atomically."""
    dir_name = os.path.dirname(data_path)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".tmp") as tmp:
        write_fn(tmp.name)
        tmp_path = tmp.name
    os.rename(tmp_path, data_path)  # Atomic on POSIX
```

## Deduplication

When fetching from APIs that may return overlapping data:

```python
# By unique key
df = df.drop_duplicates(subset=["id"], keep="last")

# By content hash
df["_hash"] = df.apply(lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1)
df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
```
