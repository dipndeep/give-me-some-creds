"""
Scoring Engine & Underwriting Logic Module
==========================================
CreditRiskAI: End-to-End Delinquency Prediction & Underwriting Simulator
Author: Ganendra Pradipa

This module provides the production scoring engine, FICO score calibration,
automated policy rules (approval/review/decline), and on-the-fly local SHAP
explainability for individual and batch credit underwriting.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

# Human-readable labels and descriptions for credit risk features
FEATURE_DICTIONARY = {
    'RevolvingUtilizationOfUnsecuredLines': {
        'label': 'Revolving Credit Utilization',
        'unit': '%',
        'desc': 'Total balance on credit cards & personal lines divided by total credit limit'
    },
    'age': {
        'label': 'Borrower Age',
        'unit': 'years',
        'desc': 'Age of borrower in years'
    },
    'NumberOfTime30-59DaysPastDueNotWorse': {
        'label': '30–59 Days Delinquency Count',
        'unit': 'times',
        'desc': 'Number of times borrower has been 30-59 days past due in the past 2 years'
    },
    'DebtRatio': {
        'label': 'Debt-to-Income Ratio (DTI)',
        'unit': '%',
        'desc': 'Monthly debt payments and living costs divided by monthly gross income'
    },
    'MonthlyIncome': {
        'label': 'Gross Monthly Income',
        'unit': 'USD',
        'desc': 'Gross monthly earnings of the borrower'
    },
    'NumberOfOpenCreditLinesAndLoans': {
        'label': 'Open Credit Lines & Loans',
        'unit': 'accounts',
        'desc': 'Number of open credit cards and consumer loan facilities'
    },
    'NumberOfTimes90DaysLate': {
        'label': '90+ Days Severe Default Count',
        'unit': 'times',
        'desc': 'Frequency of 90+ days past due delinquency (critical default indicator)'
    },
    'NumberRealEstateLoansOrLines': {
        'label': 'Real Estate & Mortgage Loans',
        'unit': 'properties',
        'desc': 'Count of mortgages and secured real estate lines of credit'
    },
    'NumberOfTime60-89DaysPastDueNotWorse': {
        'label': '60–89 Days Delinquency Count',
        'unit': 'times',
        'desc': 'Number of times borrower has been 60-89 days past due in the past 2 years'
    },
    'NumberOfDependents': {
        'label': 'Family Dependents Count',
        'unit': 'persons',
        'desc': 'Number of dependents in family excluding the borrower'
    },
    'HasPastDueRecordError': {
        'label': 'Past-Due Data Anomaly Flag',
        'unit': 'flag',
        'desc': 'Indicator for bureau reporting anomalies (e.g. code 96/98)'
    },
    'RevolvingUtilizationOverLimit': {
        'label': 'Over-Limit Utilization Flag (>100%)',
        'unit': 'flag',
        'desc': 'Credit utilization exceeded 100% of sanctioned limit'
    },
    'MonthlyIncome_is_missing': {
        'label': 'Imputed Missing Income Flag',
        'unit': 'flag',
        'desc': 'Income was unverified or missing at underwriting application'
    },
    'DebtRatio_is_extreme': {
        'label': 'Extreme Debt Ratio Flag (>500%)',
        'unit': 'flag',
        'desc': 'Debt ratio exceeds 5x monthly income or income missing'
    },
    'NumberOfDependents_is_missing': {
        'label': 'Missing Dependents Flag',
        'unit': 'flag',
        'desc': 'Dependents count was not declared at application'
    }
}

# Standard Default Personas for rapid testing & demonstration
PRESET_PERSONAS = {
    'prime': {
        'name': 'Persona A: Prime Executive',
        'badge': 'Low Risk (FICO ~810)',
        'description': 'Mature executive with verified mortgage, low credit utilization, and flawless payment history.',
        'data': {
            'age': 48,
            'MonthlyIncome': 12500.0,
            'DebtRatio': 0.22,
            'RevolvingUtilizationOfUnsecuredLines': 0.12,
            'NumberOfOpenCreditLinesAndLoans': 8,
            'NumberRealEstateLoansOrLines': 2,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfTimes90DaysLate': 0,
            'NumberOfDependents': 2
        }
    },
    'stretched': {
        'name': 'Persona B: Stretched Young Borrower',
        'badge': 'Moderate Risk (FICO ~610)',
        'description': 'Young adult with high credit card utilization (78%) and 1 minor late payment, seeking loan without collateral.',
        'data': {
            'age': 26,
            'MonthlyIncome': 3200.0,
            'DebtRatio': 0.48,
            'RevolvingUtilizationOfUnsecuredLines': 0.78,
            'NumberOfOpenCreditLinesAndLoans': 5,
            'NumberRealEstateLoansOrLines': 0,
            'NumberOfTime30-59DaysPastDueNotWorse': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfTimes90DaysLate': 0,
            'NumberOfDependents': 0
        }
    },
    'distressed': {
        'name': 'Persona C: Distressed Multi-Default',
        'badge': 'High Risk (FICO ~460)',
        'description': 'Borrower over limit (>100% utilization) with multiple 90+ day severe delinquencies and high debt burden.',
        'data': {
            'age': 39,
            'MonthlyIncome': 2100.0,
            'DebtRatio': 0.85,
            'RevolvingUtilizationOfUnsecuredLines': 1.15,
            'NumberOfOpenCreditLinesAndLoans': 12,
            'NumberRealEstateLoansOrLines': 0,
            'NumberOfTime30-59DaysPastDueNotWorse': 2,
            'NumberOfTime60-89DaysPastDueNotWorse': 1,
            'NumberOfTimes90DaysLate': 2,
            'NumberOfDependents': 3
        }
    }
}


class ScoringEngine:
    """
    Central scoring and explainability engine for retail credit risk underwriting.
    Loads champion model, scaler, and metadata, providing inference, FICO calibration,
    and fast SHAP local explanations.
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to relative models/ directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')

        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.explainer = None
        self.feature_schema = []
        self.optimal_threshold = 0.56

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, scaler, and metadata from disk."""
        model_path = os.path.join(self.models_dir, 'best_credit_model.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        meta_path = os.path.join(self.models_dir, 'model_metadata.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Champion model not found at {model_path}")

        self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.feature_schema = self.metadata.get('feature_schema', [])
            self.optimal_threshold = self.metadata.get('operational_policy', {}).get('cost_optimal_threshold', 0.56)

        # Initialize SHAP TreeExplainer for XGBoost model
        # TreeExplainer is extremely fast for tree ensembles
        self.explainer = shap.TreeExplainer(self.model)

    def prepare_features(self, raw_input: dict) -> pd.DataFrame:
        """
        Transforms raw applicant input into exactly 15 schema-aligned features,
        automatically calculating the 5 engineered risk flags.
        """
        # Extract inputs with defaults
        utilization = float(raw_input.get('RevolvingUtilizationOfUnsecuredLines', 0.15))
        age = float(raw_input.get('age', 52))
        past_due_30 = int(raw_input.get('NumberOfTime30-59DaysPastDueNotWorse', 0))
        debt_ratio = float(raw_input.get('DebtRatio', 0.35))
        income = raw_input.get('MonthlyIncome')
        open_lines = int(raw_input.get('NumberOfOpenCreditLinesAndLoans', 8))
        times_90_late = int(raw_input.get('NumberOfTimes90DaysLate', 0))
        real_estate = int(raw_input.get('NumberRealEstateLoansOrLines', 1))
        past_due_60 = int(raw_input.get('NumberOfTime60-89DaysPastDueNotWorse', 0))
        dependents = raw_input.get('NumberOfDependents')

        # Compute Flags
        has_error = 1 if (past_due_30 >= 96 or past_due_60 >= 96 or times_90_late >= 96) else 0
        util_overlimit = 1 if utilization > 1.0 else 0
        
        income_missing = 1 if (income is None or pd.isna(income) or income <= 0) else 0
        final_income = 5400.0 if income_missing else float(income)

        debt_extreme = 1 if (debt_ratio > 5.0 or (income_missing and debt_ratio > 10.0)) else 0

        dependents_missing = 1 if (dependents is None or pd.isna(dependents)) else 0
        final_dependents = 0.0 if dependents_missing else float(dependents)

        # Build feature dictionary aligned to schema
        features = {
            'RevolvingUtilizationOfUnsecuredLines': utilization,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': past_due_30,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': final_income,
            'NumberOfOpenCreditLinesAndLoans': open_lines,
            'NumberOfTimes90DaysLate': times_90_late,
            'NumberRealEstateLoansOrLines': real_estate,
            'NumberOfTime60-89DaysPastDueNotWorse': past_due_60,
            'NumberOfDependents': final_dependents,
            'HasPastDueRecordError': has_error,
            'RevolvingUtilizationOverLimit': util_overlimit,
            'MonthlyIncome_is_missing': income_missing,
            'DebtRatio_is_extreme': debt_extreme,
            'NumberOfDependents_is_missing': dependents_missing
        }

        # Convert to DataFrame ensuring identical column order
        df = pd.DataFrame([features])
        if self.feature_schema:
            df = df[self.feature_schema]

        return df

    def predict_risk(self, features_df: pd.DataFrame) -> dict:
        """
        Runs XGBoost inference and computes probability of default, FICO score,
        and automated underwriting policy tier.
        """
        # Ensure DataFrame has exact feature columns
        X = features_df[self.feature_schema] if self.feature_schema else features_df

        # Predict probability of default (Class 1)
        prob_default = float(self.model.predict_proba(X)[0, 1])

        # Calibrate FICO score (300 to 850)
        fico_score = self.calculate_fico_score(prob_default)

        # Underwriting policy tier
        policy = self.evaluate_policy_tier(prob_default, fico_score)

        return {
            'prob_default': prob_default,
            'prob_default_pct': round(prob_default * 100, 2),
            'fico_score': fico_score,
            'fico_rating': self.get_fico_rating(fico_score),
            'policy': policy,
            'features_df': features_df
        }

    @staticmethod
    def calculate_fico_score(prob_default: float) -> int:
        """
        Calibrates model default probability to the global FICO score scale (300–850).
        Uses log-odds scaling calibrated to:
        - Baseline 6.68% default rate -> ~720 Score
        - Optimal Cost Threshold 56%  -> 550 Score (Fair/Decline boundary)
        - 1% default rate             -> ~835 Score (Excellent)
        - >90% default rate           -> 300–380 Score (Subprime/Poor)
        """
        # Clamp probability to prevent log domain errors
        p = np.clip(prob_default, 0.0001, 0.9999)
        odds = (1.0 - p) / p

        # Optimal cutoff odds at p = 0.56
        odds_cutoff = (1.0 - 0.56) / 0.56 # ≈ 0.7857

        # Base score at cutoff is 550, scaling factor calibrated to baseline portfolio odds
        # Baseline p=0.0668, odds=13.97, delta_log_odds ≈ 2.878 -> gives 720 score
        scaling_factor = 59.07
        raw_score = 550.0 + scaling_factor * (np.log(odds) - np.log(odds_cutoff))

        # Clamp strictly between 300 and 850
        return int(np.clip(np.round(raw_score), 300, 850))

    @staticmethod
    def get_fico_rating(fico_score: int) -> dict:
        """Categorizes FICO score into standard consumer credit tiers."""
        if fico_score >= 750:
            return {'grade': 'Excellent', 'color': '#10b981', 'badge': 'Super Prime'}
        elif fico_score >= 700:
            return {'grade': 'Good', 'color': '#0ea5e9', 'badge': 'Prime'}
        elif fico_score >= 650:
            return {'grade': 'Fair / Acceptable', 'color': '#3b82f6', 'badge': 'Near Prime'}
        elif fico_score >= 550:
            return {'grade': 'Marginal / Review', 'color': '#f59e0b', 'badge': 'Subprime'}
        else:
            return {'grade': 'Poor / Distressed', 'color': '#ef4444', 'badge': 'Deep Subprime'}

    def evaluate_policy_tier(self, prob_default: float, fico_score: int) -> dict:
        """
        Determines the automated underwriting decision based on optimal asymmetric
        loss cutoff (0.56) and risk grade tiers.
        """
        cutoff = self.optimal_threshold

        if prob_default < 0.10 and fico_score >= 750:
            return {
                'decision': 'INSTANT APPROVAL',
                'tier': 'Tier 1: Prime (Low Risk)',
                'status': 'approved',
                'color': '#10b981',
                'icon': '',
                'interest_rate': '8.25% – 9.75% p.a. (Prime Rate)',
                'credit_limit_multiplier': 'Hingga 3.0× Batas Standar',
                'collateral_required': False,
                'action_recommendation': (
                    "Persetujuan otomatis (Straight-Through Processing). "
                    "Risiko default sangat rendah. Tawarkan fasilitas limit maksimal dan program bunga loyalitas."
                )
            }
        elif prob_default < 0.35 and fico_score >= 650:
            return {
                'decision': 'STANDARD APPROVAL',
                'tier': 'Tier 2: Standard Performing',
                'status': 'approved_standard',
                'color': '#0ea5e9',
                'icon': '',
                'interest_rate': '11.50% – 13.50% p.a. (Regular Rate)',
                'credit_limit_multiplier': '1.0× – 1.5× Batas Standar',
                'collateral_required': False,
                'action_recommendation': (
                    "Aplikasi disetujui secara otomatis melalui sistem standar. "
                    "Profil risiko terpantau sehat dalam parameter selera risiko portofolio bank."
                )
            }
        elif prob_default < cutoff and fico_score >= 550:
            return {
                'decision': 'CONDITIONAL / MANUAL REVIEW',
                'tier': 'Tier 3: Moderate Risk (Borderline)',
                'status': 'review',
                'color': '#f59e0b',
                'icon': '',
                'interest_rate': '15.50% – 18.00% p.a. (Risk-Adjusted)',
                'credit_limit_multiplier': '0.5× – 0.75× Batas Standar',
                'collateral_required': True,
                'action_recommendation': (
                    "Aplikasi tidak dapat disetujui otomatis. Wajib dialihkan ke Credit Underwriter "
                    "untuk verifikasi dokumen fisik (rekening koran 3 bulan, bukti kepemilikan aset/agunan, atau penjamin)."
                )
            }
        else:
            return {
                'decision': 'AUTOMATED DECLINE',
                'tier': 'Tier 4: Subprime / High Risk',
                'status': 'declined',
                'color': '#ef4444',
                'icon': '',
                'interest_rate': 'N/A (Ditolak)',
                'credit_limit_multiplier': '0× (Aplikasi Ditolak)',
                'collateral_required': True,
                'action_recommendation': (
                    "Aplikasi ditolak otomatis demi memitigasi potensi Non-Performing Loan (NPL). "
                    "Probabilitas default melampaui batas toleransi risiko asimetris (Threshold 0.56). "
                    "Terbitkan dokumen resmi Adverse Action Notice dengan 3 alasan objektif SHAP."
                )
            }

    def explain_applicant_shap(self, features_df: pd.DataFrame) -> dict:
        """
        Computes local SHAP attribution values for a single applicant record on the fly.
        Extracts top 3 adverse action risk drivers and top protective factors.
        """
        X = features_df[self.feature_schema] if self.feature_schema else features_df

        # Compute SHAP explanation object
        shap_explanation = self.explainer(X)
        shap_values_array = shap_explanation.values[0]
        base_value = float(shap_explanation.base_values[0])

        # Create sorted Series of feature contributions
        contrib_series = pd.Series(shap_values_array, index=self.feature_schema)

        # Features that increase risk (positive SHAP in log-odds)
        risk_drivers = contrib_series[contrib_series > 0].sort_values(ascending=False)
        # Features that decrease risk (negative SHAP in log-odds)
        protective_factors = contrib_series[contrib_series < 0].sort_values(ascending=True)

        top_3_adverse = []
        for feat, val in risk_drivers.head(3).items():
            meta = FEATURE_DICTIONARY.get(feat, {'label': feat, 'unit': ''})
            actual_val = features_df.iloc[0][feat]
            top_3_adverse.append({
                'feature': feat,
                'label': meta['label'],
                'actual_value': actual_val,
                'unit': meta['unit'],
                'shap_value': round(float(val), 4),
                'narrative': self._format_adverse_narrative(feat, actual_val, val)
            })

        top_3_protective = []
        for feat, val in protective_factors.head(3).items():
            meta = FEATURE_DICTIONARY.get(feat, {'label': feat, 'unit': ''})
            actual_val = features_df.iloc[0][feat]
            top_3_protective.append({
                'feature': feat,
                'label': meta['label'],
                'actual_value': actual_val,
                'unit': meta['unit'],
                'shap_value': round(float(val), 4),
                'narrative': self._format_protective_narrative(feat, actual_val, val)
            })

        return {
            'shap_explanation': shap_explanation,
            'base_value': base_value,
            'shap_values': shap_values_array,
            'top_adverse_drivers': top_3_adverse,
            'top_protective_drivers': top_3_protective,
            'feature_contributions': contrib_series.to_dict()
        }

    @staticmethod
    def _format_adverse_narrative(feature: str, val: float, shap_val: float) -> str:
        """Formats regulatory Adverse Action reason in clear plain language."""
        if feature == 'RevolvingUtilizationOfUnsecuredLines':
            return f"Penggunaan limit kredit tinggi ({val*100:.1f}%), mengindikasikan ketergantungan likuiditas utang (+{shap_val:.2f} risk)."
        elif feature == 'NumberOfTimes90DaysLate':
            return f"Terdapat {int(val)} riwayat tunggakan parah 90+ hari dalam riwayat kredit (+{shap_val:.2f} risk)."
        elif feature == 'NumberOfTime30-59DaysPastDueNotWorse':
            return f"Terdapat {int(val)} riwayat keterlambatan 30–59 hari (+{shap_val:.2f} risk)."
        elif feature == 'NumberOfTime60-89DaysPastDueNotWorse':
            return f"Terdapat {int(val)} riwayat keterlambatan moderat 60–89 hari (+{shap_val:.2f} risk)."
        elif feature == 'DebtRatio':
            return f"Rasio beban utang terhadap pendapatan tinggi ({val*100:.1f}%) (+{shap_val:.2f} risk)."
        elif feature == 'age':
            return f"Profil usia peminjam ({int(val)} thn) berada pada kelompok statistik risiko tinggi (+{shap_val:.2f} risk)."
        elif feature == 'RevolvingUtilizationOverLimit':
            return f"Saldo fasilitas kredit melampaui 100% dari plafon limit resmi (+{shap_val:.2f} risk)."
        else:
            label = FEATURE_DICTIONARY.get(feature, {}).get('label', feature)
            return f"Nilai faktor '{label}' ({val}) berkontribusi menaikkan estimasi probabilitas gagal bayar (+{shap_val:.2f})."

    @staticmethod
    def _format_protective_narrative(feature: str, val: float, shap_val: float) -> str:
        """Formats protective credit factors in clear plain language."""
        if feature == 'RevolvingUtilizationOfUnsecuredLines':
            return f"Disiplin utilisasi limit kredit sangat sehat ({val*100:.1f}%), menekan risiko gagal bayar ({shap_val:.2f})."
        elif feature == 'NumberRealEstateLoansOrLines':
            return f"Kepemilikan {int(val)} fasilitas agunan KPR/properti memberikan buffer stabilitas aset ({shap_val:.2f})."
        elif feature == 'age':
            return f"Kematangan usia peminjam ({int(val)} thn) berkorelasi positif dengan stabilitas finansial ({shap_val:.2f})."
        elif feature == 'MonthlyIncome':
            return f"Pendapatan bulanan (${val:,.0f}) memberikan kapasitas pembayaran cicilan yang memadai ({shap_val:.2f})."
        elif feature == 'NumberOfOpenCreditLinesAndLoans':
            return f"Diversifikasi portofolio pinjaman aktif ({int(val)} fasilitas) menunjukkan pengalaman kredit yang baik ({shap_val:.2f})."
        else:
            label = FEATURE_DICTIONARY.get(feature, {}).get('label', feature)
            return f"Nilai faktor '{label}' ({val}) membantu memperbaiki skor dan memperkecil risiko ({shap_val:.2f})."

    def score_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a batch of applicants from a DataFrame, adding PD, FICO Score,
        and Decision status columns.
        """
        results = []
        for _, row in df_raw.iterrows():
            feat_df = self.prepare_features(row.to_dict())
            pred = self.predict_risk(feat_df)
            res = {
                'Prob_Default_Pct': pred['prob_default_pct'],
                'FICO_Score': pred['fico_score'],
                'Credit_Grade': pred['fico_rating']['grade'],
                'Underwriting_Decision': pred['policy']['decision'],
                'Decision_Tier': pred['policy']['tier']
            }
            results.append(res)

        res_df = pd.DataFrame(results)
        return pd.concat([df_raw.reset_index(drop=True), res_df], axis=1)
