# Borderlands Science — Player Engagement Analysis

This project analyzes player engagement and churn patterns in the Borderlands Science citizen‑science minigame using feature engineering, survival analysis (Cox models), and clustering. This repository contains a data‑preparation pipeline and a research notebook for modeling and segmentation.

## Dataset and Scope

**Data Selection:** This analysis focuses on a subset of players (5000 of over 4 million players) who successfully completed all 9 difficulty levels (i.e., played and completed at least one puzzle at level 9). This filtering criterion was applied to ensure analysis on experienced, committed players with sufficient gameplay data across the full difficulty spectrum. This subset provides the most reliable signal for engagement patterns and churn behavior while minimizing bias from casual players.

**Timeframe:** The play logs analyzed span from April 2020 through June 2025, capturing player behavior across a significant period of citizen science participation.

**Reproducibility:** The feature engineering pipeline can be readily applied to other player subsets or datasets by adjusting the filtering criteria in the initial data preprocessing steps. See the [Run the feature pipeline](#run-the-feature-pipeline) section for guidance on customizing the analysis for different player cohorts.

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
- **Reproducible pipeline**: Fully scripted feature engineering that can be re‑run on new datasets or alternative player cohorts, enabling comparative analysis across different player subsets and time periods.
- **Transparent methodology**: Clear filtering and preprocessing steps documented in code, facilitating validation and extension of results to other engagement contexts.

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

**Data requirement:** The input CSV should contain raw Borderlands Science gameplay logs. This file is not included in the repository due to privacy concerns but the expected format is described below.

**Data filtering for reproducibility:** Note that the current analysis uses players who completed at least one level 9 puzzle (i.e., reached the final difficulty). To adapt this pipeline for a different player subset:

1. Open [player_profile.py](player_profile.py) and locate the data filtering logic
2. Modify the filtering criteria in the preprocessing or post-processing steps (e.g., filter by `level_9_puzzles > 0` in the notebook, or adjust timeframe filters like the COVID flag)
3. Follow the pipeline steps below with your modified dataset
4. Document your filtering rationale to maintain reproducibility

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
