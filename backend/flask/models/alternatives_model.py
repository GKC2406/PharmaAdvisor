from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
app = Flask(__name__)
CORS(app)
def safe_str(s):
    return str(s).lower().strip() if pd.notna(s) else ""

def string_similarity(a, b):
    return SequenceMatcher(None, safe_str(a), safe_str(b)).ratio()

def build_comparison_dataset(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    comparison_rows = []

    for i, original in df.iterrows():
        for j, candidate in df.iterrows():
            if i == j:
                continue

            if not (
                candidate["Disease"] == original["Disease"]
                or candidate["Use"] == original["Use"]
                or candidate["ActionClass"] == original["ActionClass"]
            ):
                continue

            orig_ing = safe_str(original["Active_Ingredient"])
            cand_ing = safe_str(candidate["Active_Ingredient"])
            ing_similarity = string_similarity(orig_ing, cand_ing)

            same_ingredient = orig_ing == cand_ing
            same_dosage_unit = safe_str(candidate["DosageUnit1"]) == safe_str(original["DosageUnit1"])

            try:
                orig_dose = float(original["DosageQty1"])
                cand_dose = float(candidate["DosageQty1"])
            except:
                orig_dose = cand_dose = None

            dose_ratio = (cand_dose / orig_dose) if same_ingredient and cand_dose and orig_dose else 0
            lower_dose_same_ing = 1 if (same_ingredient and cand_dose and orig_dose and (cand_dose < orig_dose)) else 0

            severity_improved = candidate["Severity"] < original["Severity"]
            side_effect_changed = safe_str(candidate["Side_Effect"]) != safe_str(original["Side_Effect"])
            is_better = int(severity_improved or side_effect_changed)

            comparison = {
                "OriginalMed": original["Medicine"],
                "CandidateMed": candidate["Medicine"],
                "Disease": original["Disease"],
                "SameActionClass": int(candidate["ActionClass"] == original["ActionClass"]),
                "SameUse": int(candidate["Use"] == original["Use"]),
                "DifferentIngredient": int(orig_ing != cand_ing),
                "SameDosageForm": int(candidate["DosageForm"] == original["DosageForm"]),
                "SharedSideEffect": int(not side_effect_changed),
                "ActionUseMatchScore": int(
                    candidate["ActionClass"] == original["ActionClass"]
                    and candidate["Use"] == original["Use"]
                ),
                "IngredientSimilarity": ing_similarity,
                "SameDosageUnit": int(same_dosage_unit),
                "LowerDoseSameIngredient": lower_dose_same_ing,
                "DoseRatio": dose_ratio,
                "IsBetterAlternative": is_better
            }

            comparison_rows.append(comparison)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_csv_path, index=False)
    return len(comparison_df)


def train_and_evaluate_model(comparison_csv_path):
    if not os.path.exists(comparison_csv_path):
        return None, "Comparison dataset not found!"

    df = pd.read_csv(comparison_csv_path)

    pos = df[df["IsBetterAlternative"] == 1]
    neg = df[df["IsBetterAlternative"] == 0]
    min_class = min(len(pos), len(neg))

    pos_bal = pos.sample(min_class, random_state=42)
    neg_bal = neg.sample(min_class, random_state=42)
    balanced_df = pd.concat([pos_bal, neg_bal]).sample(frac=1.0, random_state=42)

    features = [
        "SameActionClass", "SameUse", "DifferentIngredient", "SameDosageForm",
        "SharedSideEffect", "ActionUseMatchScore", "IngredientSimilarity",
        "SameDosageUnit", "LowerDoseSameIngredient", "DoseRatio"
    ]

    X = balanced_df[features]
    y = balanced_df["IsBetterAlternative"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
        "f1_score": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, probas), 3)
    }

    return metrics, None
