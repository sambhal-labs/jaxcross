# Pipeline Scheduling Guide

## Cron (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Daily at 6am
0 6 * * * cd /path/to/project && /path/to/python pipeline.py --force >> pipeline_cron.log 2>&1

# Every Monday at midnight
0 0 * * 1 cd /path/to/project && /path/to/python pipeline.py --force >> pipeline_cron.log 2>&1

# Every 6 hours
0 */6 * * * cd /path/to/project && /path/to/python pipeline.py --force >> pipeline_cron.log 2>&1
```

## Systemd Timer (Linux)

```ini
# /etc/systemd/system/data-pipeline.service
[Unit]
Description=Data Pipeline

[Service]
Type=oneshot
WorkingDirectory=/path/to/project
ExecStart=/path/to/python pipeline.py --force
User=myuser
```

```ini
# /etc/systemd/system/data-pipeline.timer
[Unit]
Description=Run Data Pipeline Daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable data-pipeline.timer
sudo systemctl start data-pipeline.timer
```

## Apache Airflow

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    "data_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

run_pipeline = BashOperator(
    task_id="run_pipeline",
    bash_command="cd /path/to/project && python pipeline.py --force",
    dag=dag,
)
```

## Prefect

```python
from prefect import flow, task
from prefect.deployments import Deployment

@task
def run_pipeline():
    import subprocess
    subprocess.run(["python", "pipeline.py", "--force"], check=True)

@flow
def data_pipeline_flow():
    run_pipeline()

# Schedule
deployment = Deployment.build_from_flow(
    flow=data_pipeline_flow,
    name="daily-data-pipeline",
    schedule={"cron": "0 6 * * *"},
)
deployment.apply()
```

## GitHub Actions

```yaml
name: Data Pipeline
on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python pipeline.py --force
      - uses: actions/upload-artifact@v4
        with:
          name: pipeline-output
          path: data/
```
