# jaxcross Agent Skills

Production-grade agent skills for building end-to-end probabilistic modeling workflows with [jaxcross](https://github.com/sambhal-labs/jaxcross). Compatible with Claude Code, Cursor, Copilot, Gemini CLI, and any [Agent Skills](https://agentskills.io)-compatible platform.

## Installation

**Claude Code** (recommended):
```bash
# From the jaxcross repo (skills auto-discovered)
npx skills add sambhal-labs/jaxcross

# Or copy manually
cp -r skills/ ~/.claude/skills/jaxcross-skills/
```

**Cursor**: Copy each `SKILL.md` to `.cursor/rules/` in your project.

**Any Agent Skills-compatible tool**: Point at this `skills/` directory.

## Skills Overview

### Tier 1: Data Preparation (general-purpose)

| Skill | Command | What it does |
|-------|---------|-------------|
| [data-fetch](data-fetch/) | `/data-fetch` | Fetch data from APIs, databases, public datasets, URLs |
| [data-quality](data-quality/) | `/data-quality` | Profile and validate tabular data quality |
| [data-transform](data-transform/) | `/data-transform` | Encode, normalize, and prepare data for modeling |
| [data-pipeline](data-pipeline/) | `/data-pipeline` | Generate reusable fetch-validate-transform pipeline script |

### Tier 2: jaxcross Workflow

| Skill | Command | What it does |
|-------|---------|-------------|
| [jxc-quickstart](jxc-quickstart/) | `/jxc-quickstart` | Raw file to first insights in 5 minutes |
| [jxc-model](jxc-model/) | `/jxc-model` | Production model training with convergence monitoring |
| [jxc-anomaly](jxc-anomaly/) | `/jxc-anomaly` | Detect and rank anomalous rows |
| [jxc-impute](jxc-impute/) | `/jxc-impute` | Fill missing values with Bayesian confidence |
| [jxc-discover](jxc-discover/) | `/jxc-discover` | Discover variable dependencies and structure |
| [jxc-predict](jxc-predict/) | `/jxc-predict` | Classify, forecast, and compute credible intervals |
| [jxc-segment](jxc-segment/) | `/jxc-segment` | Discover and profile natural data segments |

### Tier 3: Production Operations

| Skill | Command | What it does |
|-------|---------|-------------|
| [jxc-serve](jxc-serve/) | `/jxc-serve` | Deploy model for online inference |
| [jxc-monitor](jxc-monitor/) | `/jxc-monitor` | Monitor model health and detect drift |

## Typical Workflows

**Quick exploration:**
```
/data-fetch → /data-quality → /data-transform → /jxc-quickstart
```

**Production anomaly detection:**
```
/data-fetch → /data-quality → /data-transform → /data-pipeline
→ /jxc-model → /jxc-anomaly → /jxc-serve → /jxc-monitor
```

**Missing data imputation:**
```
/data-fetch → /data-quality → /data-transform
→ /jxc-model → /jxc-impute → /jxc-discover
```

**Customer segmentation:**
```
/data-fetch → /data-quality → /data-transform
→ /jxc-model → /jxc-segment → /jxc-predict
```

## Requirements

- Python 3.11+
- jaxcross (`pip install -e .` from source)
- JAX with GPU support (recommended): `pip install jax[cuda12]`
- For data skills: `pandas`, `pyarrow`, `requests` (standard data science stack)
