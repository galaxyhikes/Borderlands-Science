# Borderlands Science — Player Engagement Analysis

This project analyzes player engagement and churn patterns in the Borderlands Science citizen‑science minigame using feature engineering, survival analysis (Cox models), and clustering. This repository contains a data‑preparation pipeline and a research notebook for modeling and segmentation.

## What the project does

This project transforms raw Borderlands Science gameplay logs into player‑level behavioral features, then applies survival analysis and clustering to characterize engagement archetypes and churn risk.

Key artifacts:
- [player_profile.py](player_profile.py): End‑to‑end feature engineering pipeline that outputs processed player profiles.
- [Cluster cox.ipynb](Cluster%20cox.ipynb): Research notebook for Cox regression and K‑means clustering.
- [Independent_study_final_report.pdf](Independent_study_final_report.pdf): Written report about the project context, methods, results and suggestions.

## Why the project is useful

- **Scalable engagement insights**: Convert raw play sessions into durable player‑level metrics.
- **Churn modeling**: Cox proportional hazards modeling for survival analysis.
- **Behavioral archetypes**: K‑means clustering to identify player profiles within the dataset.
- **Reproducible pipeline**: Scripted feature engineering that can be re‑run on new datasets.

## Key Findings

This analysis identified **three distinct player archetypes** in Borderlands Science:

- **Speedrunners**: Players focused on rapid puzzle completion with lower consistency.
- **Completionists**: Players committed to thorough puzzle exploration and high performance standards.
- **Explorers**: Players with moderate engagement and variable session intensity.

Performance consistency and session intensity influence player churn differently depending on archetype—challenging one‑size‑fits‑all engagement models. These insights provide a foundation for targeted interventions to reduce churn and improve citizen science platform sustainability.

For detailed methods and results, see [Independent_study_final_report.pdf](Independent_study_final_report.pdf).

## How users can get started

### Install dependencies with Poetry (recommended)

This repo includes [pyproject.toml](pyproject.toml) for reproducible installs. Install Poetry, then run: `poetry install`.

If you plan to use the notebook, run `poetry run jupyter lab` (or open the notebook from your editor with the Poetry environment selected).

### Install dependencies with pip (alternative)

Install the common libraries with pip if you prefer not to use Poetry: `pip install pandas numpy scipy statsmodels scikit-learn lifelines matplotlib seaborn`.

### Run the feature pipeline

**Data requirement:** The input CSV should contain raw Borderlands Science gameplay logs. This file is not included in the repository due to size and is expected to be sourced from Gearbox or a research data repository.

Expected CSV columns:

| Column | Description |
|--------|-------------|
| `player_id` | Unique player identifier |
| `timestamp` | UNIX time (seconds) of the puzzle attempt |
| `difficulty` | Puzzle difficulty level (1–9) |
| `puzzle_id` | Unique puzzle identifier |
| `par_score` | Target score for the puzzle |
| `score` | Final score achieved by player |
| `duration` | Time spent on puzzle (milliseconds) |
| `progress` | Semicolon-separated scoring steps (format: `score1:time1;score2:time2;...`) |

To run the pipeline:

1. Obtain the raw Borderlands Science gameplay CSV from your data source.
2. Open [player_profile.py](player_profile.py) and set `file_path` to your CSV location.
3. Run the script to generate:
	- `player_profile.csv` — player-level aggregated features
	- `player_profile_processed.csv` — transformed features ready for modeling
	- `transformation_log.csv` — record of applied transformations

### Explore modeling and clustering

Open [Cluster cox.ipynb](Cluster%20cox.ipynb) and update the absolute paths in the first cells to point to your local copies of:

- [player_profile.py](player_profile.py)
- `player_profile.csv`
- `player_profile_processed.csv`

Then run the notebook top‑to‑bottom to reproduce the Cox modeling and K‑means analysis.

## Where users can get help

- Questions or issues: open a GitHub issue at https://github.com/galaxyhikes/Borderlands-Science/issues
- Project context and methods: see [Independent_study_final_report.pdf](Independent_study_final_report.pdf)
- Pipeline details: review docstrings in [player_profile.py](player_profile.py)

## Who maintains and contributes

Maintainer: https://github.com/galaxyhikes

Contributions are welcome via pull requests. There is no formal contributing guide yet; please open an issue to discuss changes or propose improvements before submitting larger updates.
