"""
Data Cleaning & Preprocessing Module for Credit Risk Scoring Project.
Project: Gimme Some Creds (Credit Default Prediction)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class CreditDataCleaner:
    """
    Production-grade data cleaning pipeline for credit risk datasets.
    Handles credit bureau error codes (96/98), zero-age anomalies,
    extreme utilization capping, and grouped missing income imputation.
    """

    def __init__(self, util_cap_quantile: float = 0.99, debt_cap_quantile: float = 0.99):
        self.util_cap_quantile = util_cap_quantile
        self.debt_cap_quantile = debt_cap_quantile
        self.util_cap_val_: Optional[float] = None
        self.debt_cap_val_: Optional[float] = None
        self.median_age_: Optional[float] = None
        self.median_income_overall_: Optional[float] = None
        self.median_income_by_age_: Optional[pd.Series] = None
        self.median_dependents_: Optional[float] = None

    def _create_age_groups(self, ages: pd.Series) -> pd.Series:
        """Categorize ages into business cohorts."""
        bins = [-np.inf, 29, 44, 59, np.inf]
        labels = ['<30', '30-44', '45-59', '60+']
        return pd.cut(ages, bins=bins, labels=labels)

    def fit(self, df: pd.DataFrame) -> 'CreditDataCleaner':
        """
        Learn statistical parameters (medians, quantiles) from training data
        to avoid data leakage into validation or test sets.
        """
        df_copy = df.copy()

        # Age median (ignoring age == 0)
        valid_ages = df_copy.loc[df_copy['age'] > 0, 'age']
        self.median_age_ = valid_ages.median()

        # Quantile caps for extreme values
        self.util_cap_val_ = df_copy['RevolvingUtilizationOfUnsecuredLines'].quantile(self.util_cap_quantile)
        self.debt_cap_val_ = df_copy['DebtRatio'].quantile(self.debt_cap_quantile)

        # Dependents median
        self.median_dependents_ = df_copy['NumberOfDependents'].median()

        # Monthly income medians (overall and cohort-based)
        temp_age = df_copy['age'].replace(0, self.median_age_)
        age_groups = self._create_age_groups(temp_age)
        self.median_income_overall_ = df_copy['MonthlyIncome'].median()
        self.median_income_by_age_ = df_copy.groupby(age_groups, observed=False)['MonthlyIncome'].median()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning and imputation transformations to dataframe.
        """
        data = df.copy()

        # Drop index column if present
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

        # 1. Clean Age Anomaly (age == 0)
        data['age'] = data['age'].replace(0, np.nan)
        data['age'] = data['age'].fillna(self.median_age_).astype(int)

        # 2. Handle Delinquency Bureau Error Codes (96 and 98)
        delinq_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]
        # Flag bureau error before imputation
        data['HasPastDueRecordError'] = (
            (data[delinq_cols[0]].isin([96, 98])) |
            (data[delinq_cols[1]].isin([96, 98])) |
            (data[delinq_cols[2]].isin([96, 98]))
        ).astype(int)

        # Replace error codes with 0 (mode of delinquency)
        for col in delinq_cols:
            data[col] = data[col].replace([96, 98], 0)

        # 3. Handle Revolving Utilization (Over-limit & Outliers)
        data['RevolvingUtilizationOverLimit'] = (
            data['RevolvingUtilizationOfUnsecuredLines'] > 1.0
        ).astype(int)
        data['RevolvingUtilizationOfUnsecuredLines'] = data[
            'RevolvingUtilizationOfUnsecuredLines'
        ].clip(upper=self.util_cap_val_)

        # 4. Handle MonthlyIncome Missing Values
        data['MonthlyIncome_is_missing'] = data['MonthlyIncome'].isna().astype(int)
        age_groups = self._create_age_groups(data['age'])
        
        # Impute by age cohort, fallback to overall median
        cohort_medians = age_groups.map(self.median_income_by_age_)
        data['MonthlyIncome'] = data['MonthlyIncome'].fillna(cohort_medians).fillna(self.median_income_overall_)

        # 5. Handle DebtRatio Extremes (often linked to missing income)
        data['DebtRatio_is_extreme'] = (data['DebtRatio'] > self.debt_cap_val_).astype(int)
        data['DebtRatio'] = data['DebtRatio'].clip(upper=self.debt_cap_val_)

        # 6. Handle NumberOfDependents Missing Values
        data['NumberOfDependents_is_missing'] = data['NumberOfDependents'].isna().astype(int)
        data['NumberOfDependents'] = data['NumberOfDependents'].fillna(self.median_dependents_)

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit parameters and transform training dataframe."""
        return self.fit(df).transform(df)


def run_pipeline(
    train_path: str = 'datasets/cs-training.csv',
    test_path: str = 'datasets/cs-test.csv',
    output_dir: str = 'data/processed'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute end-to-end data cleaning and persist processed files.
    """
    print(f"Loading raw datasets from {train_path} and {test_path}...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    cleaner = CreditDataCleaner()
    print("Fitting cleaner on training data...")
    train_cleaned = cleaner.fit_transform(train_raw)
    
    print("Transforming test data with learned statistics...")
    test_cleaned = cleaner.transform(test_raw)

    print(f"Saving cleaned datasets to {output_dir}...")
    # Save CSV
    train_cleaned.to_csv(f"{output_dir}/train_cleaned.csv", index=False)
    test_cleaned.to_csv(f"{output_dir}/test_cleaned.csv", index=False)

    # Save Parquet for fast loading (if pyarrow/fastparquet available)
    try:
        train_cleaned.to_parquet(f"{output_dir}/train_cleaned.parquet", index=False)
        test_cleaned.to_parquet(f"{output_dir}/test_cleaned.parquet", index=False)
        print("Successfully exported Parquet files.")
    except Exception as e:
        print(f"Parquet export skipped ({e}). CSV files are successfully saved.")

    print(f"Done! Cleaned Train Shape: {train_cleaned.shape}, Test Shape: {test_cleaned.shape}")
    return train_cleaned, test_cleaned


if __name__ == '__main__':
    run_pipeline()
