---
name: data-fetch
description: Fetch tabular data from REST APIs, public datasets (World Bank, Kaggle, UCI, FRED), databases (SQL/BigQuery), URLs (CSV/JSON/Parquet), or HTML tables. Use when the user needs to acquire data from any external source before analysis or modeling.
version: "1.0.0"
license: Apache-2.0
---

# Data Fetch

Acquire tabular data from any source and write it to a local file ready for downstream analysis.

Usage: `/data-fetch <source_description>`

Examples:
- `/data-fetch World Bank GDP data for all countries 2010-2023`
- `/data-fetch https://example.com/dataset.csv`
- `/data-fetch Kaggle titanic dataset`
- `/data-fetch PostgreSQL query: SELECT * FROM transactions WHERE date > '2024-01-01'`

## Step 1: Identify source type

Determine which acquisition path to use:

| Source | Indicators | Path |
|--------|-----------|------|
| REST API | URL with `/api/`, mentions "endpoint", JSON response | Step 2A |
| Public dataset | Mentions World Bank, Kaggle, UCI, FRED, Census, WHO | Step 2B |
| Database | Mentions SQL, PostgreSQL, MySQL, BigQuery, Snowflake | Step 2C |
| Direct URL | URL ending in .csv, .json, .parquet, .xlsx, .tsv | Step 2D |
| HTML table | URL to a webpage with visible tables | Step 2E |

## Step 2A: REST API

```python
import requests
import pandas as pd
import time

def fetch_api(base_url, params=None, headers=None, max_pages=100):
    """Fetch paginated API data into a DataFrame."""
    all_records = []
    page = 1
    
    while page <= max_pages:
        response = requests.get(
            base_url,
            params={**(params or {}), "page": page},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        # Adapt to API response structure
        records = data if isinstance(data, list) else data.get("results", data.get("data", []))
        if not records:
            break
        all_records.extend(records)
        
        # Rate limiting
        time.sleep(0.5)
        page += 1
    
    return pd.json_normalize(all_records)
```

**Authentication patterns:**
- API key: `headers={"Authorization": "Bearer YOUR_KEY"}` or `params={"api_key": "YOUR_KEY"}`
- OAuth: use `requests_oauthlib` for token flow
- No auth: many public APIs (World Bank, FRED) require no authentication

**Pagination patterns:**
- Page number: increment `page` param until empty response
- Cursor: use `next_cursor` from response for next request
- Link header: parse `response.headers["Link"]` for next URL
- Offset: increment by `limit` each request

**Rate limiting:** Always add `time.sleep(0.5)` between requests. If you get 429, implement exponential backoff:
```python
for attempt in range(5):
    response = requests.get(url)
    if response.status_code != 429:
        break
    time.sleep(2 ** attempt)
```

See [api-patterns.md](references/api-patterns.md) for advanced patterns.

## Step 2B: Public datasets

### World Bank (WDI)
```python
import requests
import pandas as pd

# Fetch indicator data via World Bank API v2
indicator = "NY.GDP.MKTP.CD"  # GDP current USD
url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
params = {"format": "json", "per_page": 1000, "date": "2010:2023"}

response = requests.get(url, params=params)
data = response.json()[1]  # [0] is metadata, [1] is data
df = pd.DataFrame(data)
df = df[["country", "countryiso3code", "date", "value"]].dropna(subset=["value"])
```

Common indicators: `NY.GDP.MKTP.CD` (GDP), `SP.POP.TOTL` (population), `SL.UEM.TOTL.ZS` (unemployment), `FP.CPI.TOTL.ZG` (inflation). See [public-datasets.md](references/public-datasets.md) for full list.

### Kaggle
```bash
# Requires ~/.kaggle/kaggle.json with API credentials
pip install kaggle
kaggle datasets download -d <owner>/<dataset-name> --unzip -p ./data/
```

### UCI ML Repository
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=53)  # Iris
df = dataset.data.original
```
Or direct download: `pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")`

### FRED (Federal Reserve Economic Data)
```python
import pandas as pd
# Direct CSV download (no API key needed for basic access)
series_id = "GDP"
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
df = pd.read_csv(url, parse_dates=["DATE"])
```

### Census / Government Open Data
```python
# US Census API
url = "https://api.census.gov/data/2020/acs/acs5"
params = {"get": "NAME,B01001_001E", "for": "state:*"}
response = requests.get(url, params=params)
df = pd.DataFrame(response.json()[1:], columns=response.json()[0])
```

See [public-datasets.md](references/public-datasets.md) for 15+ more sources.

## Step 2C: Database

```python
import pandas as pd

# PostgreSQL
import psycopg2
conn = psycopg2.connect(host="localhost", dbname="mydb", user="user", password="pass")
df = pd.read_sql("SELECT * FROM table_name WHERE condition", conn)
conn.close()

# MySQL
import mysql.connector
conn = mysql.connector.connect(host="localhost", database="mydb", user="user", password="pass")
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()

# SQLite
import sqlite3
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()

# BigQuery
from google.cloud import bigquery
client = bigquery.Client()
df = client.query("SELECT * FROM `project.dataset.table`").to_dataframe()

# Snowflake
import snowflake.connector
conn = snowflake.connector.connect(user="user", password="pass", account="account")
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()
```

See [database-connectors.md](references/database-connectors.md) for connection string templates and credential management.

**Security:** Never hardcode credentials. Use environment variables:
```python
import os
password = os.environ["DB_PASSWORD"]
```

## Step 2D: Direct URL download

```python
import pandas as pd

# CSV
df = pd.read_csv("https://example.com/data.csv")

# JSON (records format)
df = pd.read_json("https://example.com/data.json")

# Parquet
df = pd.read_parquet("https://example.com/data.parquet")

# Excel
df = pd.read_excel("https://example.com/data.xlsx")

# TSV
df = pd.read_csv("https://example.com/data.tsv", sep="\t")

# Compressed files
df = pd.read_csv("https://example.com/data.csv.gz", compression="gzip")
```

For large files, stream the download:
```python
import requests
response = requests.get(url, stream=True)
with open("data.csv", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
df = pd.read_csv("data.csv")
```

## Step 2E: HTML table scraping

```python
import pandas as pd

# Simple: pandas read_html (extracts all tables from a page)
tables = pd.read_html("https://example.com/page-with-tables")
df = tables[0]  # First table on the page

# Complex: BeautifulSoup for specific tables
from bs4 import BeautifulSoup
import requests
response = requests.get("https://example.com/page")
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table", {"class": "data-table"})
df = pd.read_html(str(table))[0]
```

See [web-scraping-guide.md](references/web-scraping-guide.md) for handling JavaScript-rendered pages and multi-page scraping.

## Step 3: Validate download

After fetching, always verify:

```python
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:")
print(df.head())
```

**Fail if:**
- 0 rows returned (empty dataset)
- All values in any column are null
- Unexpected column names (API schema changed)

## Step 4: Save to local file

```python
# Preferred: Parquet (preserves types, compressed, fast)
df.to_parquet("data/raw_data.parquet", index=False)

# Alternative: CSV (universal compatibility)
df.to_csv("data/raw_data.csv", index=False)

# Record metadata
metadata = {
    "source": "<original URL or query>",
    "fetched_at": pd.Timestamp.now().isoformat(),
    "rows": len(df),
    "columns": list(df.columns),
}
import json
with open("data/raw_data_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

Print the output path and summary so downstream skills (`/data-quality`, `/data-transform`) know where to find it.

## Common Pitfalls

- **API rate limits**: Always add delays between requests. Check API docs for rate limit headers (`X-RateLimit-Remaining`).
- **Character encoding**: If you see garbled text, try `pd.read_csv(url, encoding="latin-1")` or `encoding="utf-8-sig"`.
- **Large downloads**: For files >1GB, use streaming download and chunked reading (`pd.read_csv(..., chunksize=10000)`).
- **API key security**: Never commit API keys to git. Use `.env` files or environment variables.
