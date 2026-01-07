import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_csv(path):
    """Load CSV and return a DataFrame."""
    # Some CSVs (like the image attributes file) contain a few metadata/comment rows
    # before the real header line that starts with 'Image_No'. Detect that header row
    # and read the CSV with the correct header index.
    # no extra stdlib imports required here

    header_row = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if "Image_No" in line:
                    header_row = i
                    break
    except Exception:
        # If file can't be opened in text mode for some reason, fall back to pandas
        try:
            return pd.read_csv(path)
        except Exception:
            # give up with a descriptive error
            raise

    if header_row is None:
        # no special header found; let pandas infer
        return pd.read_csv(path)

    # read using detected header row index
    try:
        df = pd.read_csv(path, header=header_row, encoding="utf-8")
    except Exception:
        # try BOM-friendly encoding
        df = pd.read_csv(path, header=header_row, encoding="utf-8-sig")
    return df


def build_feature_matrix(df, target_col="Valence_omnivore_Male", feature_set="all"):
    """Return X (DataFrame) and y (Series) with numeric predictors.

    feature_set controls which predictors to include:
      - "all": all numeric columns (except identifier-like columns and the target)
      - "nutrition": select nutritional/macro columns (protein_100g, fat_100g, carbs_100g,
                     kcal_100g, no_items_image, grams_total, protein_total, fat_total,
                     carbs_total, Kcal_total) if present in the CSV

    X is returned as a DataFrame (keeps column names) so GLM can operate on it.
    """
    numeric = df.select_dtypes(include=["number"]).copy()
    # drop identifier-like columns
    drop_cols = [c for c in numeric.columns if c.lower().startswith("image_no")]
    if target_col not in numeric.columns:
        raise ValueError(f"target {target_col} not found in numeric columns")
    if feature_set == "all":
        X = numeric.drop(columns=[target_col] + drop_cols)
    elif feature_set == "nutrition":
        # canonical set of nutrition columns present in this CSV
        # Match case-insensitively and accept common variants (Kcal vs kcal)
        candidate_cols = [
            "grams_total",
            "protein_100g",
            "fat_100g",
            "carbs_100g",
            "kcal_100g",
            "red",
            "green",
            "blue",
            "taste:1=sweet 2=tasty",
            "food: 1=whole foods, 2=processed foods",
        ]

        # Map lowercased column names to actual columns present in the original df
        lc_to_col = {c.lower(): c for c in df.columns}
        selected = []
        for cand in candidate_cols:
            if cand in numeric.columns:
                selected.append(cand)
            else:
                lc = cand.lower()
                if lc in lc_to_col:
                    selected.append(lc_to_col[lc])

        # Ensure the 'total' macros are coerced to numeric when present
        coerced = {}
        import re

        for col in selected:
            # Try coercing to numeric first
            try:
                num = pd.to_numeric(df[col], errors="coerce")
                # if coercion gives any numeric values, use it
                if num.notna().any():
                    coerced[col] = num
                    continue
            except Exception:
                num = None

            # If not numeric or coercion failed, treat as categorical and create dummies
            try:
                ser = df[col].astype(str).fillna("")
                # sanitize prefix for dummy column names
                prefix = re.sub(r"\W+", "_", col).strip("_")
                dummies = pd.get_dummies(ser, prefix=prefix, dummy_na=False).astype(int)
                # add each dummy column into coerced dict
                for dc in dummies.columns:
                    coerced[dc] = dummies[dc]
            except Exception:
                # final fallback: empty NA series
                coerced[col] = pd.Series([pd.NA] * len(df))

        if not coerced:
            raise ValueError("No nutritional columns found in the provided CSV")

        X = pd.DataFrame(coerced)
        # Debug: print selected nutrition columns so the runner can see what was used
        try:
            print("[build_feature_matrix] selected nutrition columns:", list(X.columns))
        except Exception:
            pass

    else:
        raise ValueError(f"unknown feature_set: {feature_set}")

    y = numeric[target_col]
    return X, y


def split_and_preprocess(X, y, test_size=0.2, random_state=0):
    """Impute and scale numeric DataFrame. Returns numpy arrays and the fitted transformers.

    Returns: X_train_np, X_test_np, y_train, y_test, preprocess_dict, feature_names
    """
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_df)
    X_test_imp = imputer.transform(X_test_df)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    preprocess = {
        "imputer": imputer,
        "scaler": scaler,
    }

    feature_names = list(X.columns)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.values,
        y_test.values,
        preprocess,
        feature_names,
    )


def load_trials_from_dir(participant_dir, question=None):
    """Read rating CSVs under participant directories and return combined DataFrame.

    Expects files like <participant_folder>/rating_data_*.csv with columns including
    'Participant ID', 'Image Name', 'Question', 'Rating', 'RT (rel)'.
    The returned DataFrame includes columns: 'Participant ID', 'Image Name', 'Question',
    'Rating', 'RT (rel)', and a numeric 'Image_No' extracted from Image Name.
    If question is provided, only rows matching that question are returned.
    """
    import glob
    import os

    pattern = os.path.join(participant_dir, "*", "rating_data*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No rating CSVs found with pattern: {pattern}")

    parts = []
    skipped = []
    for f in files:
        try:
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                # file exists but is empty
                skipped.append((f, "empty file"))
                continue
            except UnicodeDecodeError:
                # try with BOM-friendly encoding
                try:
                    df = pd.read_csv(f, encoding="utf-8-sig")
                except Exception as e:
                    skipped.append((f, f"decode error: {e}"))
                    continue
            except Exception:
                # final fallback to python engine
                try:
                    df = pd.read_csv(f, engine="python")
                except Exception as e:
                    skipped.append((f, f"unreadable: {e}"))
                    continue

            # if dataframe has no columns or is empty, skip
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                skipped.append((f, "empty_or_no_columns"))
                continue

        except Exception as e:
            skipped.append((f, f"unexpected error: {e}"))
            continue

        # keep expected columns if present
        cols = df.columns.tolist()
        # Normalize column names
        rename_map = {}
        lower_cols = [c.lower() for c in cols]
        if "participant id" in lower_cols and "Participant ID" not in cols:
            for c in cols:
                if c.lower() == "participant id":
                    rename_map[c] = "Participant ID"
        if "image name" in lower_cols and "Image Name" not in cols:
            for c in cols:
                if c.lower() == "image name":
                    rename_map[c] = "Image Name"
        df = df.rename(columns=rename_map)

        if question is not None and "Question" in df.columns:
            df = df[df["Question"] == question]

        # ensure Rating column exists
        if "Rating" not in df.columns:
            # try to find a column that looks like Rating
            found = False
            for c in df.columns:
                if c.lower() == "rating":
                    df = df.rename(columns={c: "Rating"})
                    found = True
                    break
            if not found:
                # no rating column -> skip this file
                skipped.append((f, "no rating column"))
                continue

        parts.append(df)

    if not parts:
        msg = f"No valid rating CSVs were loaded from {participant_dir}."
        if skipped:
            msg += " Skipped files:\n" + "\n".join(
                [f"{p}: {reason}" for p, reason in skipped]
            )
        raise FileNotFoundError(msg)

    combined = pd.concat(parts, ignore_index=True, sort=False)

    # extract numeric image id from Image Name
    if "Image Name" in combined.columns:
        combined["Image_No"] = (
            combined["Image Name"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float)
            .astype("Int64")
        )
    else:
        raise ValueError("Rating files do not contain 'Image Name' column")

    return combined
