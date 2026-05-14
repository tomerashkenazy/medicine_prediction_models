"""Cohort filtering rules based on the stage 1 answer."""

from __future__ import annotations

import pandas as pd


def apply_stage1_cohort(
    df: pd.DataFrame,
    *,
    exclude_pediatric: bool = True,
    keep_first_patient_record: bool = True,
    apply_min_pre_icu_los_filter: bool = False,
    min_pre_icu_los_hours: float = 4.0,
) -> pd.DataFrame:
    """Apply conservative stage-1 cohort rules.

    The assignment text mentions excluding ICU length of stay under 4 hours, but
    this dataset exposes pre-ICU length of stay rather than ICU length of stay.
    That filter is therefore disabled by default.
    """
    cohort = df.copy()

    if exclude_pediatric and "age" in cohort.columns:
        cohort = cohort[(cohort["age"].isna()) | (cohort["age"] >= 18)]

    if keep_first_patient_record and "patient_id" in cohort.columns:
        sort_columns = [c for c in ["patient_id", "encounter_id"] if c in cohort.columns]
        if sort_columns:
            cohort = cohort.sort_values(sort_columns)
        cohort = cohort.drop_duplicates(subset=["patient_id"], keep="first")

    if apply_min_pre_icu_los_filter and "pre_icu_los_days" in cohort.columns:
        min_days = min_pre_icu_los_hours / 24.0
        cohort = cohort[cohort["pre_icu_los_days"] >= min_days]

    return cohort.reset_index(drop=True)

