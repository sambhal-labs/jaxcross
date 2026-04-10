# Public Dataset Sources Reference

## World Bank (WDI)

**API**: `https://api.worldbank.org/v2/`
**Auth**: None required
**Format**: JSON or XML

Common indicators:
| Indicator Code | Description |
|---------------|-------------|
| NY.GDP.MKTP.CD | GDP (current US$) |
| NY.GDP.PCAP.CD | GDP per capita (current US$) |
| SP.POP.TOTL | Population, total |
| SL.UEM.TOTL.ZS | Unemployment (% of labor force) |
| FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) |
| SE.XPD.TOTL.GD.ZS | Government education expenditure (% GDP) |
| SH.XPD.CHEX.GD.ZS | Health expenditure (% GDP) |
| EN.ATM.CO2E.PC | CO2 emissions (metric tons per capita) |
| IT.NET.USER.ZS | Internet users (% of population) |
| SI.POV.GINI | Gini index |

```python
indicator = "NY.GDP.MKTP.CD"
url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
params = {"format": "json", "per_page": 1000, "date": "2000:2023"}
```

## Kaggle

**CLI**: `kaggle datasets download`
**Auth**: `~/.kaggle/kaggle.json` with `{"username":"...","key":"..."}`

Popular datasets:
- `titanic`: `kaggle competitions download -c titanic`
- `house-prices`: `kaggle competitions download -c house-prices-advanced-regression-techniques`
- `credit-card-fraud`: `kaggle datasets download -d mlg-ulb/creditcardfraud`
- `airline-passengers`: `kaggle datasets download -d chirag19/air-passengers`

## UCI ML Repository

**URL**: `https://archive.ics.uci.edu/`
**Auth**: None
**Python package**: `ucimlrepo`

Popular datasets:
| ID | Name | Rows | Cols | Types |
|----|------|------|------|-------|
| 53 | Iris | 150 | 5 | Continuous + Categorical |
| 2 | Adult (Census) | 48842 | 14 | Mixed |
| 17 | Breast Cancer | 699 | 10 | Integer |
| 73 | Mushroom | 8124 | 22 | Categorical |
| 186 | Wine Quality | 4898 | 12 | Continuous |
| 320 | Student Performance | 649 | 33 | Mixed |

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=53)
df = dataset.data.original
```

## FRED (Federal Reserve Economic Data)

**URL**: `https://fred.stlouisfed.org/`
**Auth**: API key for full access, CSV download free
**API**: `https://api.stlouisfed.org/fred/`

Popular series:
| Series ID | Description |
|-----------|-------------|
| GDP | Gross Domestic Product |
| UNRATE | Unemployment Rate |
| CPIAUCSL | Consumer Price Index |
| DFF | Federal Funds Rate |
| T10YIE | 10-Year Breakeven Inflation Rate |
| DEXUSEU | USD/EUR Exchange Rate |
| SP500 | S&P 500 Index |
| MORTGAGE30US | 30-Year Fixed Mortgage Rate |

```python
# No API key needed for CSV
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
df = pd.read_csv(url, parse_dates=["DATE"])
```

## WHO (World Health Organization)

**API**: `https://ghoapi.azureedge.net/api/`
**Auth**: None

```python
url = "https://ghoapi.azureedge.net/api/WHOSIS_000001"  # Life expectancy
response = requests.get(url)
data = response.json()["value"]
df = pd.DataFrame(data)
```

## US Census Bureau

**API**: `https://api.census.gov/data/`
**Auth**: API key (free, register at census.gov)

```python
url = "https://api.census.gov/data/2020/acs/acs5"
params = {"get": "NAME,B01001_001E,B19013_001E", "for": "county:*", "in": "state:*"}
response = requests.get(url, params=params)
df = pd.DataFrame(response.json()[1:], columns=response.json()[0])
```

## Our World in Data

**URL**: `https://github.com/owid/`
**Auth**: None (direct CSV on GitHub)

```python
# COVID-19 data
df = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")

# Energy data
df = pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")
```

## Eurostat

**API**: `https://ec.europa.eu/eurostat/api/`
**Auth**: None

```python
import eurostat  # pip install eurostat
df = eurostat.get_data_df("nama_10_gdp")  # GDP and main components
```

## OpenWeather

**API**: `https://api.openweathermap.org/data/2.5/`
**Auth**: API key (free tier available)

## Yahoo Finance

```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
```

## Data.gov (US Government)

**URL**: `https://catalog.data.gov/`
**Auth**: None for most datasets
**Format**: CSV, JSON, various

Search at https://catalog.data.gov/dataset and download CSV directly.
