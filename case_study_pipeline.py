"""Train the final campaign-sales model and score messy campaign records.

The notebook is the canonical analysis. This module exposes its selected
production design as a small command-line workflow:

* four planning-time campaign inputs only;
* suffix-aware parsing and validation;
* fold-compatible clipping, median imputation, and scaling;
* Lasso regression with alpha=30;
* input-quality warnings and an approximate residual-based range.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = [
    "Followers",
    "EngagementRate (%)",
    "AdSpend (GBP)",
    "ContentQuality",
]

CLEANED_FEATURES = [
    "followers",
    "engagement_rate",
    "ad_spend",
    "content_quality",
]

TARGET = "Sales (Units)"
FINAL_ALPHA = 30.0
DEFAULT_INTERVAL_RADIUS = 3_252


def _is_grouped_integer(text: str, separator: str) -> bool:
    parts = text.split(separator)
    return (
        len(parts) > 1
        and 1 <= len(parts[0]) <= 3
        and parts[0].isdigit()
        and all(part.isdigit() and len(part) == 3 for part in parts[1:])
    )


def _normalise_number_separators(text: str) -> str | None:
    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        if text.count(decimal_separator) != 1:
            return None

        integer_part, decimal_part = text.rsplit(decimal_separator, 1)
        if not decimal_part.isdigit():
            return None
        if thousands_separator in integer_part:
            if not _is_grouped_integer(integer_part, thousands_separator):
                return None
            integer_part = integer_part.replace(thousands_separator, "")
        if not integer_part.isdigit():
            return None
        return f"{integer_part}.{decimal_part}"

    if "," in text:
        if text.count(",") == 1:
            integer_part, trailing_part = text.split(",", 1)
            if not integer_part.isdigit() or not trailing_part.isdigit():
                return None
            if len(trailing_part) == 3 and 1 <= len(integer_part) <= 3:
                return integer_part + trailing_part
            return f"{integer_part}.{trailing_part}"
        return text.replace(",", "") if _is_grouped_integer(text, ",") else None

    if text.count(".") > 1:
        return text.replace(".", "") if _is_grouped_integer(text, ".") else None
    return text


def parse_messy_number(value: Any) -> float:
    """Parse common currency, percentage, suffix, and separator formats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else np.nan

    text = str(value).strip()
    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "-",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return np.nan

    parenthesised_negative = text.startswith("(") and text.endswith(")")
    if parenthesised_negative:
        text = text[1:-1].strip()

    multiplier = 1.0
    suffix_match = re.search(r"([kKmMbB])\s*$", text)
    if suffix_match:
        multiplier = {
            "k": 1_000.0,
            "m": 1_000_000.0,
            "b": 1_000_000_000.0,
        }[suffix_match.group(1).lower()]
        text = text[: suffix_match.start()]

    text = re.sub(
        r"^(GBP|USD|EUR|INR)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if re.search(r"[A-Za-z]", text):
        return np.nan

    text = re.sub(r"[^0-9,.+\-]", "", text)
    if not text or not re.search(r"\d", text):
        return np.nan

    sign = ""
    if text[0] in "+-":
        sign, text = text[0], text[1:]
    if not text or "+" in text or "-" in text:
        return np.nan

    normalised = _normalise_number_separators(text)
    if normalised is None:
        return np.nan

    try:
        numeric_value = float(sign + normalised) * multiplier
    except ValueError:
        return np.nan

    if parenthesised_negative:
        numeric_value = -abs(numeric_value)
    return numeric_value if np.isfinite(numeric_value) else np.nan


class CampaignCleaner(BaseEstimator, TransformerMixin):
    """Convert messy campaign inputs into four validated numeric features."""

    def __init__(self, clip_lower: float = 0.01, clip_upper: float = 0.99):
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "CampaignCleaner":
        cleaned = self._basic_clean(X)
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        for column in CLEANED_FEATURES:
            values = cleaned[column].dropna()
            if values.empty:
                self.clip_bounds_[column] = (np.nan, np.nan)
            else:
                self.clip_bounds_[column] = (
                    float(values.quantile(self.clip_lower)),
                    float(values.quantile(self.clip_upper)),
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._basic_clean(X)
        for column, (lower, upper) in self.clip_bounds_.items():
            if not math.isnan(lower) and not math.isnan(upper):
                cleaned[column] = cleaned[column].clip(lower, upper)
        return cleaned[CLEANED_FEATURES]

    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        raw_followers = X.get(
            "Followers",
            pd.Series(index=X.index, dtype=object),
        )
        raw_follower_text = raw_followers.astype("string").str.strip()
        has_magnitude_suffix = raw_follower_text.str.contains(
            r"[kKmMbB]\s*$",
            regex=True,
            na=False,
        )
        is_plain_numeric_follower = raw_follower_text.str.fullmatch(
            r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)",
            na=False,
        )

        followers = raw_followers.map(parse_messy_number)
        engagement = X.get(
            "EngagementRate (%)",
            pd.Series(index=X.index, dtype=object),
        ).map(parse_messy_number)
        ad_spend = X.get(
            "AdSpend (GBP)",
            pd.Series(index=X.index, dtype=object),
        ).map(parse_messy_number)
        content_quality = X.get(
            "ContentQuality",
            pd.Series(index=X.index, dtype=object),
        ).map(parse_messy_number)

        small_plain_followers = (
            followers.between(1, 999, inclusive="both")
            & is_plain_numeric_follower
            & ~has_magnitude_suffix
        )
        followers = followers.where(
            ~small_plain_followers,
            followers * 1_000.0,
        )

        followers = followers.mask(followers <= 0)
        engagement = engagement.mask((engagement <= 0) | (engagement > 100))
        ad_spend = ad_spend.mask(ad_spend < 0)
        content_quality = content_quality.mask(
            (content_quality < 1) | (content_quality > 10)
        )

        cleaned = pd.DataFrame(
            {
                "followers": followers,
                "engagement_rate": engagement,
                "ad_spend": ad_spend,
                "content_quality": content_quality,
            },
            index=X.index,
        )
        return cleaned.replace([np.inf, -np.inf], np.nan)


def make_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, CLEANED_FEATURES)],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("cleaner", CampaignCleaner()),
            ("preprocess", preprocessor),
            (
                "model",
                Lasso(alpha=FINAL_ALPHA, max_iter=100_000, random_state=42),
            ),
        ]
    )


def train_model(train_data: pd.DataFrame) -> Pipeline:
    required = MODEL_FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in train_data]
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    target = pd.to_numeric(train_data[TARGET], errors="coerce")
    if target.isna().any():
        raise ValueError("Training target contains missing or non-numeric values.")

    model = make_model()
    model.fit(train_data[MODEL_FEATURES], target)
    return model


def _input_quality_messages(
    campaign_data: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    parsed_inputs = CampaignCleaner()._basic_clean(campaign_data)
    raw_to_parsed = {
        "Followers": "followers",
        "EngagementRate (%)": "engagement_rate",
        "AdSpend (GBP)": "ad_spend",
        "ContentQuality": "content_quality",
    }

    warnings: list[str] = []
    imputed_fields: list[str] = []
    reliability_flags: list[str] = []

    for row_index in campaign_data.index:
        row_warnings: list[str] = []
        row_imputed_fields: list[str] = []
        for raw_column, parsed_column in raw_to_parsed.items():
            raw_value = campaign_data.loc[row_index, raw_column]
            parsed_value = parse_messy_number(raw_value)
            cleaned_value = parsed_inputs.loc[row_index, parsed_column]
            if pd.isna(raw_value):
                reason = "was missing"
            elif pd.isna(parsed_value):
                reason = "could not be parsed"
            elif pd.isna(cleaned_value):
                reason = "was outside the accepted range"
            else:
                continue

            row_warnings.append(f"{raw_column} {reason} and was imputed")
            row_imputed_fields.append(raw_column)

        if row_warnings:
            warnings.append("Warning: " + "; ".join(row_warnings) + ".")
            imputed_fields.append(", ".join(row_imputed_fields))
            reliability_flags.append("Review - imputed input")
        else:
            warnings.append("None")
            imputed_fields.append("None")
            reliability_flags.append("Standard")

    return warnings, imputed_fields, reliability_flags


def predict_campaigns(
    model: Pipeline,
    campaign_data: pd.DataFrame,
    interval_radius_units: int = DEFAULT_INTERVAL_RADIUS,
) -> pd.DataFrame:
    missing_columns = [
        column for column in MODEL_FEATURES if column not in campaign_data
    ]
    if missing_columns:
        raise ValueError(f"Prediction data is missing columns: {missing_columns}")

    model_inputs = campaign_data[MODEL_FEATURES].copy()
    warnings, imputed_fields, reliability = _input_quality_messages(model_inputs)
    point_prediction = (
        model.predict(model_inputs).clip(min=0).round().astype(int)
    )

    output_columns = ["ID"] if "ID" in campaign_data else []
    output = campaign_data[output_columns + MODEL_FEATURES].copy()
    output["Input Warning"] = warnings
    output["Imputed Fields"] = imputed_fields
    output["Prediction Reliability Flag"] = reliability
    output["Predicted Sales (Units)"] = point_prediction
    output["Approx. 90% Lower (Units)"] = np.maximum(
        point_prediction - interval_radius_units,
        0,
    )
    output["Approx. 90% Upper (Units)"] = (
        point_prediction + interval_radius_units
    )
    return output


def illustrative_campaign() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Followers": "125k",
                "EngagementRate (%)": "3.2%",
                "AdSpend (GBP)": "GBP 5,000",
                "ContentQuality": 8,
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected Lasso model and predict campaign sales."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("messy_train_data.csv"),
        help="Labeled messy training CSV.",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        help="Campaign CSV to score. Omit to run the illustrative example.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("campaign_predictions.csv"),
        help="Destination CSV for predictions.",
    )
    parser.add_argument(
        "--interval-radius",
        type=int,
        default=DEFAULT_INTERVAL_RADIUS,
        help="Approximate residual-based interval radius in sales units.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_data = pd.read_csv(args.train_data)
    campaign_data = (
        pd.read_csv(args.input_data)
        if args.input_data
        else illustrative_campaign()
    )

    model = train_model(training_data)
    predictions = predict_campaigns(
        model,
        campaign_data,
        interval_radius_units=args.interval_radius,
    )
    predictions.to_csv(args.output, index=False)

    preview_columns = [
        "Predicted Sales (Units)",
        "Approx. 90% Lower (Units)",
        "Approx. 90% Upper (Units)",
        "Prediction Reliability Flag",
    ]
    print(predictions[preview_columns].head().to_string(index=False))
    print(f"\nPrediction file created: {args.output}")


if __name__ == "__main__":
    main()
