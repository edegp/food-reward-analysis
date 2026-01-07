"""Quick training script to predict normative ratings from image metadata CSV.
Run: python train.py --csv PATH_TO_CSV
"""

import argparse
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm

from utils import (
    load_csv,
    build_feature_matrix,
    split_and_preprocess,
    load_trials_from_dir,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def train(csv_path, target, out_dir, train_on_full=False, feature_set="all"):
    df = load_csv(csv_path)
    X, y = build_feature_matrix(df, target_col=target, feature_set=feature_set)
    if train_on_full:
        # Preprocess entire dataset (no train/test split) and report in-sample metrics
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)
        X_train = X_scaled
        X_test = X_scaled
        y_train = y.values
        y_test = y.values
        preprocess = {"imputer": imputer, "scaler": scaler}
        feature_names = list(X.columns)
    else:
        X_train, X_test, y_train, y_test, preprocess, feature_names = (
            split_and_preprocess(X, y)
        )

    models = {
        "ridge": Ridge(random_state=0),
        "rf": RandomForestRegressor(n_estimators=100, random_state=0),
    }

    # GLM using statsmodels (ordinary least squares)
    glm_results = None

    results = {}
    os.makedirs(out_dir, exist_ok=True)

    # If trial-level modeling requested, this function will be called differently by main()

    for name, model in models.items():
        model.fit(X_train, y_train)
        # if training on full data, evaluate in-sample (predict on training set)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"mse": mse, "r2": r2}
        joblib.dump(model, os.path.join(out_dir, f"model_{name}.joblib"))
    # save sklearn preprocess
    joblib.dump(preprocess, os.path.join(out_dir, "preprocess.joblib"))

    # Fit GLM (OLS) on the training data preserving feature names
    try:
        X_train_df = sm.add_constant(
            sm.datasets.utils.recarray_fromarrays(
                [X_train[:, i] for i in range(X_train.shape[1])],
                names=["const"] + feature_names,
            )
        )
        # The recarray approach above is clunky; instead build DataFrame
    except Exception:
        import pandas as pd

        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_train_df = sm.add_constant(X_train_df)

    glm_model = sm.OLS(y_train, X_train_df)
    glm_results = glm_model.fit()
    results["glm"] = {
        "params": glm_results.params.to_dict(),
        "aic": float(glm_results.aic),
        "bic": float(glm_results.bic),
    }
    # save summary text
    with open(os.path.join(out_dir, "glm_summary.txt"), "w") as fh:
        fh.write(glm_results.summary().as_text())
    joblib.dump(glm_results, os.path.join(out_dir, "model_glm.joblib"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="Valence_omnivore_Male")
    parser.add_argument("--out", default="models")
    parser.add_argument(
        "--train-on-full",
        action="store_true",
        help="Fit models on the entire dataset (no train/test split)",
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "nutrition"],
        default="all",
        help="Which set of features to use: 'all' or 'nutrition'",
    )
    parser.add_argument(
        "--level",
        choices=["image", "trial"],
        default="image",
        help="Modeling level: 'image' uses image-level normative ratings; 'trial' uses trial-level subject ratings",
    )
    parser.add_argument(
        "--participant-dir",
        default="../../Food_Behavior",
        help="Path containing participant folders with rating CSVs (used when --level trial)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="If set, only use trial rows matching this Question (trial-level only)",
    )
    args = parser.parse_args()

    if args.level == "image":
        res = train(
            args.csv,
            args.target,
            args.out,
            train_on_full=args.train_on_full,
            feature_set=args.feature_set,
        )
    else:
        # trial-level pipeline: load trials, merge with image attributes, and run MixedLM
        import pandas as pd
        from statsmodels.regression.mixed_linear_model import MixedLM

        trials = load_trials_from_dir(args.participant_dir, question=args.question)
        # merge with image attributes (csv_path)
        images = load_csv(args.csv)
        # ensure Image_No exists in images numeric columns
        # images has Image_No column called Image_No
        if "Image_No" not in images.columns:
            # try variants
            matched = [c for c in images.columns if c.lower().startswith("image_no")]
            if matched:
                images = images.rename(columns={matched[0]: "Image_No"})
            else:
                raise ValueError("Image_No column not found in image attributes CSV")

        trials = trials.merge(
            images, left_on="Image_No", right_on="Image_No", how="left"
        )

        # Collapse RGB into a single color_mean (simple average) to avoid
        # multicollinearity between red/green/blue.
        color_channels = ["red", "green", "blue"]
        if all(c in images.columns for c in color_channels):
            images["color_mean"] = images[color_channels].astype(float).mean(axis=1)
        if all(c in trials.columns for c in color_channels):
            trials["color_mean"] = trials[color_channels].astype(float).mean(axis=1)

        # build feature matrix from images to decide which image-derived columns to use
        X_sample, _ = build_feature_matrix(
            images, target_col=args.target, feature_set=args.feature_set
        )
        image_feature_cols = list(X_sample.columns)
        # If RGB channels are present, replace them with color_mean
        if any(c in image_feature_cols for c in color_channels):
            # remove any existing RGB entries
            image_feature_cols = [
                c for c in image_feature_cols if c not in color_channels
            ]
            # prefer color_mean if available
            if "color_mean" in images.columns:
                image_feature_cols.append("color_mean")

        # Include trial-level predictors if present (e.g., RT (rel))
        trial_level_preds = []
        if "RT (rel)" in trials.columns:
            trials["RT (rel)"] = pd.to_numeric(trials["RT (rel)"], errors="coerce")
            trial_level_preds.append("RT (rel)")

        # Add a per-participant trial index (何トライアル目) starting at 1
        if "Participant ID" in trials.columns:
            trials = trials.copy()
            trials["trial_index"] = trials.groupby("Participant ID").cumcount() + 1
            trial_level_preds.append("trial_index")

        Xcols = image_feature_cols + trial_level_preds

        # Impute image-based features from image medians, and trial-level
        # predictors (like RT) from trial medians.
        image_cols_present = [c for c in image_feature_cols if c in images.columns]
        trial_cols_present = [c for c in trial_level_preds if c in trials.columns]

        medians_image = pd.Series(dtype=float)
        if image_cols_present:
            medians_image = (
                images[image_cols_present]
                .apply(lambda s: pd.to_numeric(s, errors="coerce"))
                .median()
            )

        medians_trial = pd.Series(dtype=float)
        if trial_cols_present:
            medians_trial = trials[trial_cols_present].median()

        # Fill image-derived columns in trials using image medians
        for c in image_cols_present:
            trials[c] = pd.to_numeric(trials.get(c), errors="coerce")
            trials[c] = trials[c].fillna(medians_image.get(c, pd.NA))

        # Fill trial-level predictors using trial medians
        for c in trial_cols_present:
            trials[c] = pd.to_numeric(trials[c], errors="coerce")
            trials[c] = trials[c].fillna(medians_trial.get(c, pd.NA))

        # drop rows missing Rating or Participant ID only
        trials = trials.dropna(subset=["Rating", "Participant ID"])

        # prepare endog and exog
        endog = trials["Rating"].astype(float)
        exog_df = trials[Xcols].astype(float)

        # Standard-scale predictors (important for model stability)
        try:
            scaler = StandardScaler()
            exog_scaled = scaler.fit_transform(exog_df)
            # rebuild with column names so summary is readable
            import pandas as _pd

            exog = _pd.DataFrame(exog_scaled, columns=Xcols)
        except Exception:
            # fallback: use raw values if scaler unavailable
            exog = exog_df

        # add constant after scaling
        exog = sm.add_constant(exog)

        print(exog.head())
        # save scaler/preprocess for reproducibility
        os.makedirs(args.out, exist_ok=True)
        try:
            joblib.dump(
                {"scaler": scaler, "feature_names": Xcols},
                os.path.join(args.out, "preprocess_trial.joblib"),
            )
        except Exception:
            pass

        groups = trials["Participant ID"].astype(str)

        # fit MixedLM with random intercept and random slopes per participant
        # Random slopes for: trial_index
        re_cols = ["const"]
        for col in ["trial_index"]:
            if col in exog.columns:
                re_cols.append(col)

        exog_re = exog[re_cols]
        model = MixedLM(endog, exog, groups=groups, exog_re=exog_re)
        mixed_res = model.fit(reml=False)
        # save summary and model
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "mixedlm_summary.txt"), "w") as fh:
            fh.write(mixed_res.summary().as_text())
        joblib.dump(mixed_res, os.path.join(args.out, "model_mixedlm.joblib"))
        # Compose an equation string from fixed effects
        # summary will show coefficients
        fixed_effects = mixed_res.summary()
        print(fixed_effects)

        equation = "Rating ~ " + " + ".join(Xcols)
        # print("MixedLM equation:", equation)
        print("MixedLM fixed effects:", fixed_effects)

        # Compute in-sample predictions and metrics
        try:
            # mixed_res.predict accepts exog with same columns used for fitting
            preds = mixed_res.predict(exog)
            from sklearn.metrics import mean_squared_error, r2_score

            mse = float(mean_squared_error(endog, preds))
            rmse = float(mse**0.5)
            r2 = float(r2_score(endog, preds))
            # Pearson correlation between observed and predicted
            try:
                import numpy as _np

                obs = _np.asarray(endog, dtype=float)
                pr = _np.asarray(preds, dtype=float)
                # compute Pearson r
                r_num = _np.corrcoef(obs, pr)[0, 1]
                pearson_r = float(r_num) if not _np.isnan(r_num) else None
            except Exception:
                pearson_r = None

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": pearson_r,
                "n_obs": int(len(endog)),
            }
            # save metrics
            import json

            with open(os.path.join(args.out, "mixedlm_metrics.json"), "w") as fh:
                json.dump(metrics, fh, indent=2)
            # also save observed vs predicted for inspection
            try:
                import pandas as _pd

                df_pred = _pd.DataFrame(
                    {
                        "observed": _pd.to_numeric(endog, errors="coerce"),
                        "predicted": _pd.to_numeric(preds, errors="coerce"),
                    }
                )
                df_pred.to_csv(
                    os.path.join(args.out, "mixedlm_observed_predicted.csv"),
                    index=False,
                )
            except Exception:
                pass
            # print brief summary
            print("MixedLM equation:", equation)
            print("MixedLM in-sample metrics:", metrics)
        except Exception as e:
            print("Could not compute predictions/metrics:", e)
        return
    print("Results:")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
