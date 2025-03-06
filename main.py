#!/usr/bin/env python3
"""
Credit Risk Analysis Pipeline

This module loads and processes credit risk data, imputes missing values,
engineers features, builds a preprocessing pipeline, trains several models,
performs hyperparameter tuning, and evaluates model performance.
Each step logs its runtime and any created plots are stored in self.plots.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")


def timed_step(step_name):
    """
    Decorator to log the start, end, and duration of a processing step.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            print(f"Starting step: {step_name}")
            start = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start
            print(f"Finished step: {step_name} in {elapsed:.2f} seconds\n")
            self.execution_times[step_name] = elapsed
            return result

        return wrapper

    return decorator


# -----------------------
# Custom Transformers
# -----------------------


class DecisionTreeClassifierImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values for a target column using a decision tree.
    """

    def __init__(self, target_column, unused_column=None):
        self.target_column = target_column
        self.unused_column = unused_column if unused_column is not None else []
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X, y=None):
        X_reduced = X.drop(self.unused_column, axis=1) if self.unused_column else X
        df_non_missing = X_reduced[X_reduced[self.target_column].notna()]
        y_train = df_non_missing[self.target_column]
        X_train = df_non_missing.drop(self.target_column, axis=1)
        self.model.fit(X_train, y_train)
        return self

    def transform(self, X):
        X = X.copy()
        if self.unused_column:
            unused_data = X[self.unused_column]
            X_reduced = X.drop(self.unused_column, axis=1)
        else:
            X_reduced = X

        df_missing = X_reduced[X_reduced[self.target_column].isna()]
        if not df_missing.empty:
            X_test = df_missing.drop(self.target_column, axis=1)
            y_pred = self.model.predict(X_test)
            X_reduced.loc[df_missing.index, self.target_column] = y_pred

        if self.unused_column:
            X_reduced[self.unused_column] = unused_data

        return X_reduced


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Label-encodes a list of categorical columns.
    """

    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.label_encoders.items():
            X[col] = le.transform(X[col])
        return X


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom preprocessing that creates new features and drops unnecessary columns.
    """

    def __init__(
        self, age_column, cnt_fam_members_cap, cnt_children_cap, columns_to_drop
    ):
        self.age_column = age_column
        self.cnt_fam_members_cap = cnt_fam_members_cap
        self.cnt_children_cap = cnt_children_cap
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Reset invalid DAYS_EMPLOYED values
        X.loc[X["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = np.nan

        # Create AGE and EXPERIENCE_LEVEL features
        X["AGE"] = -X[self.age_column] // 365
        bins = [0, 5, 10, 20, float("inf")]
        labels = [
            "Low Experience",
            "Medium Experience",
            "High Experience",
            "Very High Experience",
        ]
        X["EXPERIENCE_LEVEL"] = pd.cut(
            -X["DAYS_EMPLOYED"] // 365, bins=bins, labels=labels, right=True
        )
        # Cap family members and children counts
        X.loc[X["CNT_FAM_MEMBERS"] > self.cnt_fam_members_cap, "CNT_FAM_MEMBERS"] = (
            self.cnt_fam_members_cap
        )
        X.loc[X["CNT_CHILDREN"] > self.cnt_children_cap, "CNT_CHILDREN"] = (
            self.cnt_children_cap
        )

        # Drop unnecessary columns
        X = X.drop(columns=self.columns_to_drop, axis=1)
        return X


# -----------------------
# Main Pipeline Class
# -----------------------


class CreditRiskModel:
    """
    An object-oriented pipeline for credit risk analysis.
    """

    def __init__(self, app_record_path, credit_record_path):
        self.app_record_path = app_record_path
        self.credit_record_path = credit_record_path
        self.plots = {}  # dictionary to store figure objects
        self.execution_times = {}  # record time for each step
        self.application_data = None
        self.credit_data = None
        self.merged_data = None
        self.pipeline = None
        self.final_feature_names = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.label_encoder = LabelEncoder()
        self.results = None
        self.models = {}  # to store trained models

    @timed_step("Load and Merge Data")
    def load_and_merge_data(self):
        """Load datasets, calculate risk score and merge the data."""
        # Load datasets
        self.application_data = pd.read_csv(self.app_record_path)
        self.credit_data = pd.read_csv(self.credit_record_path)

        # Define status weights and calculate risk scores
        status_weights = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "5": 5,
            "C": 0,
            "X": 1,
        }
        self.credit_data["month_weight"] = self.credit_data["MONTHS_BALANCE"].apply(
            lambda x: 1 / (1 - x)
        )
        self.credit_data["status_weight"] = self.credit_data["STATUS"].map(
            status_weights
        )
        self.credit_data["risk_score"] = (
            self.credit_data["status_weight"] * self.credit_data["month_weight"]
        )

        # Aggregate risk scores by ID and define risk levels
        user_risk = self.credit_data.groupby("ID")["risk_score"].sum().reset_index()
        user_risk["risk_level"] = pd.cut(
            user_risk["risk_score"],
            bins=[-float("inf"), 1.5, 3.5, float("inf")],
            labels=["No Risk", "Medium Risk", "High Risk"],
        )
        # Merge with application data
        self.merged_data = pd.merge(
            self.application_data, user_risk, on="ID", how="inner"
        )

    @timed_step("Missing Value Imputation")
    def impute_missing_values(self):
        """
        Impute missing values for OCCUPATION_TYPE and DAYS_EMPLOYED using
        decision tree classifiers.
        """
        # --- Impute OCCUPATION_TYPE ---
        cols_occ = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "CNT_FAM_MEMBERS",
        ]
        df_relevant_occ = self.merged_data[cols_occ].copy()
        cat_cols_occ = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
        ]
        le_occ = LabelEncoder()
        for col in cat_cols_occ:
            df_relevant_occ[col] = le_occ.fit_transform(df_relevant_occ[col])
        df_non_missing_occ = self.merged_data[
            self.merged_data["OCCUPATION_TYPE"].notna()
        ]
        df_missing_occ = self.merged_data[self.merged_data["OCCUPATION_TYPE"].isna()]
        X_train_occ = df_relevant_occ.loc[df_non_missing_occ.index]
        y_train_occ = le_occ.fit_transform(df_non_missing_occ["OCCUPATION_TYPE"])
        dt_occ = DecisionTreeClassifier(random_state=42)
        dt_occ.fit(X_train_occ, y_train_occ)
        X_test_occ = df_relevant_occ.loc[df_missing_occ.index]
        y_pred_occ = dt_occ.predict(X_test_occ)
        y_pred_original_occ = le_occ.inverse_transform(y_pred_occ)
        self.merged_data.loc[df_missing_occ.index, "OCCUPATION_TYPE"] = (
            y_pred_original_occ
        )

        # --- Impute DAYS_EMPLOYED ---
        # Set DAYS_EMPLOYED > 0 to NaN
        self.merged_data.loc[self.merged_data["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = (
            np.nan
        )

        cols_days = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "OCCUPATION_TYPE",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "CNT_FAM_MEMBERS",
        ]
        df_relevant_days = self.merged_data[cols_days].copy()
        cat_cols_days = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "OCCUPATION_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
        ]
        le_days = LabelEncoder()
        for col in cat_cols_days:
            df_relevant_days[col] = le_days.fit_transform(df_relevant_days[col])
        df_non_missing_days = self.merged_data[
            self.merged_data["DAYS_EMPLOYED"].notna()
        ]
        df_missing_days = self.merged_data[self.merged_data["DAYS_EMPLOYED"].isna()]
        X_train_days = df_relevant_days.loc[df_non_missing_days.index]
        y_train_days = le_days.fit_transform(df_non_missing_days["DAYS_EMPLOYED"])
        dt_days = DecisionTreeClassifier(random_state=42)
        dt_days.fit(X_train_days, y_train_days)
        X_test_days = df_relevant_days.loc[df_missing_days.index]
        y_pred_days = dt_days.predict(X_test_days)
        y_pred_original_days = le_days.inverse_transform(y_pred_days)
        self.merged_data.loc[df_missing_days.index, "DAYS_EMPLOYED"] = (
            y_pred_original_days
        )

    @timed_step("Feature Engineering")
    def feature_engineering(self):
        """
        Create new features such as AGE and EXPERIENCE_LEVEL and apply
        capping for certain features.
        """
        self.merged_data["AGE"] = -self.merged_data["DAYS_BIRTH"] // 365
        exp_bins = [0, 5, 10, 20, float("inf")]
        exp_labels = [
            "Low Experience",
            "Medium Experience",
            "High Experience",
            "Very High Experience",
        ]
        self.merged_data["EXPERIENCE_LEVEL"] = pd.cut(
            -self.merged_data["DAYS_EMPLOYED"] // 365,
            bins=exp_bins,
            labels=exp_labels,
            right=True,
        )
        self.merged_data.loc[
            self.merged_data["CNT_FAM_MEMBERS"] > 6, "CNT_FAM_MEMBERS"
        ] = 6
        self.merged_data.loc[self.merged_data["CNT_CHILDREN"] > 5, "CNT_CHILDREN"] = 5

    @timed_step("Build Preprocessing Pipeline")
    def build_pipeline(self):
        """
        Build the sklearn Pipeline with custom preprocessing and imputation.
        """
        columns_to_drop = [
            "ID",
            "DAYS_BIRTH",
            "FLAG_MOBIL",
            "risk_score",
            "DAYS_EMPLOYED",
            "FLAG_PHONE",
        ]
        custom_preproc = CustomPreprocessor(
            age_column="DAYS_BIRTH",
            cnt_fam_members_cap=6,
            cnt_children_cap=5,
            columns_to_drop=columns_to_drop,
        )
        categorical_cols = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
        ]
        pipe_steps = [
            ("select_columns", "passthrough"),
            ("custom_preprocessor", custom_preproc),
            ("encode_categorical", Preprocessor(categorical_columns=categorical_cols)),
            (
                "impute_EXPERIENCE_LEVEL",
                DecisionTreeClassifierImputer(
                    target_column="EXPERIENCE_LEVEL", unused_column=["OCCUPATION_TYPE"]
                ),
            ),
            (
                "impute_OCCUPATION_TYPE",
                DecisionTreeClassifierImputer(
                    target_column="OCCUPATION_TYPE", unused_column=["EXPERIENCE_LEVEL"]
                ),
            ),
            (
                "categorical_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
        self.pipeline = Pipeline(steps=pipe_steps)

    @timed_step("Apply Pipeline")
    def apply_pipeline(self):
        """
        Prepare the features and apply the pipeline transformation.
        """
        # Separate features and target
        X = self.merged_data.drop("risk_level", axis=1)
        # Use custom preprocessor to capture final feature names
        temp_df = self.pipeline.named_steps["custom_preprocessor"].fit_transform(
            X.copy()
        )
        self.final_feature_names = temp_df.columns.tolist()

        # Apply full pipeline
        X_cleaned_array = self.pipeline.fit_transform(X)
        self.merged_data = self.merged_data.assign(
            **{
                col: X_cleaned_array[:, idx]
                for idx, col in enumerate(self.final_feature_names)
            }
        )

    @timed_step("Outlier Detection")
    def detect_outliers(self):
        """
        Detect and store outlier boxplot figures for all numerical columns.
        """
        # Use cleaned features from the pipeline
        df = self.merged_data[self.final_feature_names].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outlier Detection for {col}")
            # Instead of showing, store the figure in self.plots
            self.plots[f"outlier_{col}"] = fig
            plt.close(fig)

    @timed_step("Resample and Split Data")
    def resample_and_split(self):
        """
        Apply SMOTEENN resampling, split into training and testing sets,
        and label encode the target.
        """
        X_cleaned = self.merged_data[self.final_feature_names].copy()
        y = self.merged_data["risk_level"]
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X_cleaned, y)
        self.X_train, self.X_test, y_train, self.y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.2,
            random_state=42,
            stratify=y_resampled,
        )
        self.y_train = self.label_encoder.fit_transform(y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

    def evaluate_model(self, model, X_test, y_test, model_name, average="weighted"):
        """
        Evaluate the model by printing accuracy, plotting a confusion matrix,
        printing a classification report and ROC curve.
        """
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc:.4f}")

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_xlabel("Predicted Label")
        self.plots[f"confusion_matrix_{model_name}"] = fig_cm
        plt.close(fig_cm)

        # Classification Report
        report = classification_report(y_test, y_pred)
        print(f"\n{model_name} Classification Report:\n{report}")

        # ROC Curve and AUC
        roc_auc = None
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob, average=average, multi_class="ovr")
            print(f"{model_name} ROC-AUC Score: {roc_auc:.4f}")

            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            for i, class_label in enumerate(np.unique(y_test)):
                fpr, tpr, _ = roc_curve(y_test == class_label, y_prob[:, i])
                ax_roc.plot(
                    fpr,
                    tpr,
                    label=f"Class {self.label_encoder.inverse_transform([class_label])[0]} (AUC = {auc(fpr, tpr):.2f})",
                )
            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set_title("ROC Curve")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            self.plots[f"roc_curve_{model_name}"] = fig_roc
            plt.close(fig_roc)
        else:
            print(
                f"ROC Curve cannot be generated for {model_name} as predict_proba is not available."
            )

        return y_pred, report, roc_auc

    @timed_step("Train Models")
    def train_models(self):
        """
        Train Random Forest, AdaBoost, and XGBoost models and evaluate them.
        """
        # Random Forest
        rf = RandomForestClassifier(random_state=42, class_weight="balanced")
        rf.fit(self.X_train, self.y_train)
        print("Random Forest (untuned) performance:")
        y_pred_rf, report_rf, roc_auc_rf = self.evaluate_model(
            rf, self.X_test, self.y_test, "RF"
        )
        fl_score_rf = f1_score(self.y_test, y_pred_rf, average="weighted")
        self.models["RF"] = {
            "model": rf,
            "f1_score": fl_score_rf,
            "roc_auc": roc_auc_rf,
        }

        # AdaBoost
        ada = AdaBoostClassifier(random_state=42)
        ada.fit(self.X_train, self.y_train)
        print("AdaBoost performance:")
        y_pred_ada, report_ada, roc_auc_ada = self.evaluate_model(
            ada, self.X_test, self.y_test, "AdaBoost"
        )
        fl_score_ada = f1_score(self.y_test, y_pred_ada, average="weighted")
        self.models["AdaBoost"] = {
            "model": ada,
            "f1_score": fl_score_ada,
            "roc_auc": roc_auc_ada,
        }

        # XGBoost
        xgb = XGBClassifier(
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )
        xgb.fit(self.X_train, self.y_train)
        print("XGBoost performance:")
        y_pred_xgb, report_xgb, roc_auc_xgb = self.evaluate_model(
            xgb, self.X_test, self.y_test, "XGBoost"
        )
        fl_score_xgb = f1_score(self.y_test, y_pred_xgb, average="weighted")
        self.models["XGBoost"] = {
            "model": xgb,
            "f1_score": fl_score_xgb,
            "roc_auc": roc_auc_xgb,
        }

    @timed_step("Hyperparameter Tuning")
    def tune_models(self):
        """
        Tune Random Forest and XGBoost using RandomizedSearchCV.
        """
        # --- Tune Random Forest ---
        rf_param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": [None, 10, 15, 20, 25, 30, 40],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 5),
            "bootstrap": [True, False],
        }
        rf_base = RandomForestClassifier(random_state=42, class_weight="balanced")
        rf_tuned_cv = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=rf_param_dist,
            scoring="f1_weighted",
            n_iter=10,
            cv=5,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        start_tune = time.time()
        rf_tuned_cv.fit(self.X_train, self.y_train)
        rf_tune_time = time.time() - start_tune
        print(f"Tuned RF Time: {rf_tune_time:.2f} seconds")
        print("Tuned Random Forest performance:")
        y_pred_rf_tuned, report_rf_tuned, roc_auc_rf_tuned = self.evaluate_model(
            rf_tuned_cv.best_estimator_, self.X_test, self.y_test, "RF_Tuned"
        )
        fl_score_rf_tuned = f1_score(self.y_test, y_pred_rf_tuned, average="weighted")
        self.models["RF_Tuned"] = {
            "model": rf_tuned_cv.best_estimator_,
            "f1_score": fl_score_rf_tuned,
            "roc_auc": roc_auc_rf_tuned,
        }

        # --- Tune XGBoost ---
        xgb_param_dist = {
            "n_estimators": randint(50, 500),
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(3, 15),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 10),
            "gamma": uniform(0, 5),
        }
        xgb_base = XGBClassifier(
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )
        xgb_tuned_cv = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=xgb_param_dist,
            scoring="f1_weighted",
            n_iter=20,
            cv=5,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        xgb_tuned_cv.fit(self.X_train, self.y_train)
        print("Tuned XGBoost performance:")
        y_pred_xgb_tuned, report_xgb_tuned, roc_auc_xgb_tuned = self.evaluate_model(
            xgb_tuned_cv.best_estimator_, self.X_test, self.y_test, "XGB_Tuned"
        )
        fl_score_xgb_tuned = f1_score(self.y_test, y_pred_xgb_tuned, average="weighted")
        self.models["XGB_Tuned"] = {
            "model": xgb_tuned_cv.best_estimator_,
            "f1_score": fl_score_xgb_tuned,
            "roc_auc": roc_auc_xgb_tuned,
        }

    @timed_step("Feature Importance Visualization")
    def plot_feature_importances(self):
        """
        Plot feature importances for the tuned Random Forest.
        """
        tuned_rf = self.models["RF_Tuned"]["model"]
        importances = tuned_rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
        feature_names_sorted = [self.final_feature_names[i] for i in indices]
        sns.barplot(
            x=feature_names_sorted, y=importances[indices], palette="viridis", ax=ax_imp
        )
        ax_imp.set_title("Feature Importances for Credit Card Prediction Risk")
        ax_imp.set_xlabel("Features")
        ax_imp.set_ylabel("Importance")
        ax_imp.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        self.plots["feature_importances"] = fig_imp
        plt.close(fig_imp)

    @timed_step("Person Qualification Analysis")
    def analyze_person_qualification(self, person_record):
        """
        Given a person's record as a pandas Series, transform the record
        and predict the risk level using the tuned Random Forest.
        """
        # Remove target if present and transform record
        person_record = person_record.drop(labels=["risk_level"], errors="ignore")
        person_df = person_record.to_frame().T
        person_transformed = self.pipeline.transform(person_df)
        tuned_rf = self.models["RF_Tuned"]["model"]
        prediction = tuned_rf.predict(person_transformed)
        risk_level = self.label_encoder.inverse_transform(prediction)
        print("Person Risk Level:", risk_level[0])
        print("Person Details:\n", person_df)

    @timed_step("Summary and Model Comparison")
    def summarize_results(self):
        """
        Print model comparison results and plot F1 and ROC AUC scores.
        """
        model_names = []
        f1_scores = []
        roc_auc_scores = []
        for name, metrics in self.models.items():
            model_names.append(name)
            f1_scores.append(metrics["f1_score"])
            roc_auc_scores.append(metrics["roc_auc"])

        self.results = pd.DataFrame(
            {
                "Model": model_names,
                "F1 Score": f1_scores,
                "ROC AUC Score": roc_auc_scores,
            }
        )
        print("\nModel Comparison Results:")
        print(self.results)

        fig_comp, axes = plt.subplots(2, 1, figsize=(10, 12))
        axes[0].bar(
            self.results["Model"],
            self.results["F1 Score"],
            color="skyblue",
            edgecolor="black",
        )
        axes[0].set_title("Model Comparison - F1 Score", fontsize=14)
        axes[0].set_ylabel("F1 Score", fontsize=12)
        axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(
            self.results["Model"],
            self.results["ROC AUC Score"],
            color="lightgreen",
            edgecolor="black",
        )
        axes[1].set_title("Model Comparison - ROC AUC Score", fontsize=14)
        axes[1].set_ylabel("ROC AUC Score", fontsize=12)
        axes[1].set_ylim(0, 1.1)
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self.plots["model_comparison"] = fig_comp
        plt.close(fig_comp)

    def run_all(self):
        """Run all pipeline steps and print total execution time."""
        total_start = time.time()
        self.load_and_merge_data()
        self.impute_missing_values()
        self.feature_engineering()
        self.build_pipeline()
        self.apply_pipeline()
        self.detect_outliers()
        self.resample_and_split()
        self.train_models()
        self.tune_models()
        self.plot_feature_importances()

        # Example person record for analysis
        print("\n--- Person Qualification Analysis ---")
        default_input_data = {
            "ID": 123456,
            "DAYS_BIRTH": -20000,
            "FLAG_MOBIL": 1,
            "DAYS_EMPLOYED": -4000,
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 50000,
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_HOUSING_TYPE": "Rented apartment",
            "risk_score": 0.0,
            "OCCUPATION_TYPE": "Other",
            "FLAG_WORK_PHONE": "N",
            "FLAG_PHONE": "N",
            "FLAG_EMAIL": "Y",
            "CNT_FAM_MEMBERS": 1,
        }
        new_person_record = pd.Series(default_input_data)
        self.analyze_person_qualification(new_person_record)

        self.summarize_results()
        total_end = time.time()
        print(f"\nTotal execution time: {total_end - total_start:.2f} seconds")


if __name__ == "__main__":
    # Update these file paths as needed
    app_record_csv = "application_record.csv"
    credit_record_csv = "credit_record.csv"
    crm = CreditRiskModel(app_record_csv, credit_record_csv)
    crm.run_all()
