# Medicine Prediction Models

Stage 2 course workflow for predicting ICU in-hospital mortality from the WiDS Datathon 2020 data.

The model output is mortality probability:

```text
P(hospital_death = 1)
```

## Setup

Use the course conda environment:

```powershell
conda activate education
python -m pip install -r requirements.txt
```

## Train All Models

```powershell
python -m src.train
```

This trains:

- `logistic_baseline`
- `logistic_ridge`
- `logistic_lasso`
- `logistic_elastic_net`

Outputs are written to:

```text
outputs/models/
outputs/reports/
outputs/figures/
outputs/predictions/
```

## Tune Regularized Models

Run nested grid search for ridge, lasso, and elastic-net logistic regression:

```powershell
python -m src.train --tune
```

This uses 5 outer folds for performance reporting and 3 inner folds for
hyperparameter selection. Tuned outputs are prefixed with `tuned_`.

## Train Tree-Based Models

Run only random forest and gradient boosted decision trees:

```powershell
python -m src.train --config-dir configs/tree_models
```

This keeps the same raw-data preprocessing pipeline used by the logistic models.
The gradient boosted tree config uses scikit-learn's histogram gradient boosting
implementation for practical runtime on this dataset.

## Modeling Notes

- Target: `hospital_death = 1`.
- Cohort filtering excludes known pediatric patients (`age < 18`) and keeps missing-age records for imputation.
- The stage-1 “less than 4 ICU hours” exclusion is not applied by default because the dataset contains `pre_icu_los_days`, not ICU length of stay.
- Models use all raw training columns except the target, row/patient identifiers, and APACHE prediction probability columns. Raw categorical variables and categorical numeric codes are one-hot encoded.
- APACHE prediction columns are not training features because they are already mortality model outputs. `apache_4a_hospital_death_prob` is reported separately as a benchmark.
- Poisson regression is skipped because the outcome is binary, not a count.
