# Web Scraping Guide

## Simple HTML Tables

```python
import pandas as pd

# Extract all tables from a page
tables = pd.read_html("https://example.com/page")
print(f"Found {len(tables)} tables")
for i, t in enumerate(tables):
    print(f"Table {i}: {t.shape}")

# Select the right table
df = tables[0]
```

## BeautifulSoup for Specific Tables

```python
from bs4 import BeautifulSoup
import requests

response = requests.get("https://example.com/page")
soup = BeautifulSoup(response.text, "html.parser")

# Find by class
table = soup.find("table", {"class": "data-table"})

# Find by ID
table = soup.find("table", {"id": "main-table"})

# Parse to DataFrame
df = pd.read_html(str(table))[0]
```

## Multi-Page Scraping

```python
all_dfs = []
for page in range(1, 11):
    url = f"https://example.com/data?page={page}"
    tables = pd.read_html(url)
    all_dfs.append(tables[0])
    time.sleep(1)  # Be polite

df = pd.concat(all_dfs, ignore_index=True)
```

## Handling JavaScript-Rendered Pages

If `pd.read_html()` returns empty, the table is likely rendered by JavaScript. Options:

1. **Check for an API**: Inspect browser Network tab for XHR/fetch requests — the data often comes from a JSON API
2. **Use Selenium** (last resort):
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://example.com/page")
html = driver.page_source
df = pd.read_html(html)[0]
driver.quit()
```

## Ethical Scraping

- Check `robots.txt` before scraping
- Add delays between requests (`time.sleep(1)`)
- Set a User-Agent header: `headers={"User-Agent": "DataFetch/1.0 (research)"}`
- Respect rate limits and terms of service
- Cache responses locally to avoid re-fetching
