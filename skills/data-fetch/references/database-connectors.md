# Database Connector Reference

## Connection String Templates

### PostgreSQL
```python
import psycopg2
conn = psycopg2.connect(
    host=os.environ.get("PG_HOST", "localhost"),
    port=os.environ.get("PG_PORT", 5432),
    dbname=os.environ["PG_DATABASE"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
)
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()
```

### MySQL
```python
import mysql.connector
conn = mysql.connector.connect(
    host=os.environ.get("MYSQL_HOST", "localhost"),
    port=os.environ.get("MYSQL_PORT", 3306),
    database=os.environ["MYSQL_DATABASE"],
    user=os.environ["MYSQL_USER"],
    password=os.environ["MYSQL_PASSWORD"],
)
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()
```

### SQLite
```python
import sqlite3
conn = sqlite3.connect("path/to/database.db")
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()
```

### BigQuery
```python
from google.cloud import bigquery
# Auth: set GOOGLE_APPLICATION_CREDENTIALS env var to service account JSON
client = bigquery.Client(project="my-project")
query = "SELECT * FROM `project.dataset.table` LIMIT 10000"
df = client.query(query).to_dataframe()
```

### Snowflake
```python
import snowflake.connector
conn = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
)
df = pd.read_sql("SELECT * FROM table_name", conn)
conn.close()
```

### SQLAlchemy (generic)
```python
from sqlalchemy import create_engine

# PostgreSQL
engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")

# MySQL
engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}")

# SQLite
engine = create_engine(f"sqlite:///path/to/database.db")

df = pd.read_sql("SELECT * FROM table_name", engine)
engine.dispose()
```

## Credential Management

**Environment variables** (preferred):
```bash
export PG_HOST=localhost
export PG_PASSWORD=secret
```

**`.env` file** (for development):
```bash
pip install python-dotenv
```
```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
password = os.environ["DB_PASSWORD"]
```

**Never commit credentials to git.** Add `.env` to `.gitignore`.

## Large Query Patterns

### Chunked reading
```python
chunks = pd.read_sql("SELECT * FROM large_table", conn, chunksize=10000)
df = pd.concat(chunks, ignore_index=True)
```

### Server-side cursors (PostgreSQL)
```python
conn = psycopg2.connect(dsn)
with conn.cursor(name="server_cursor") as cursor:
    cursor.execute("SELECT * FROM large_table")
    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break
        # Process batch
```
