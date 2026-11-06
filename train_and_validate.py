"""
Focal-Offset Regression: Random Forest vs Gradient Boosting vs Gaussian Process
=================================================================================
Predicts focal-offset distance (um) from AE waveform features extracted at
each printed position.

Two validation protocols are used, matching the manuscript:
  1. LOOCV on position-averaged data - each focal-offset position is held
     out once, the model is retrained on the rest, and the held-out
     position is predicted.
  2. Cross-round CV - train on 2 of the 3 replicate rounds, test on the
     entire 3rd round, repeated 3 times. This is the stricter check: it
     tests generalisation to a fully independent experimental session,
     not just an unseen position within an already-seen session.


Model hyperparameters are fixed in advance (not tuned via an additional
validation split, which would further shrink this already-small dataset),
and are deliberately conservative for the sample size (n=27-51):
  - RF:  300 trees, min_samples_leaf=2
  - GB:  shallow trees (depth 2), slow learning rate, subsampling
  - GPR: RBF + white-noise kernel, calibrated predictive uncertainty

"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Settings ─────────────────────────────────────────────────────────────
FEATURES_CSV = 'focal_features.csv'   # <-- set path
STEP_SIZES   = [50, 100]
TOP_N_FEATS  = 6                      # top features by Spearman |rho|, p<0.05

RF_PARAMS  = dict(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)
GB_PARAMS  = dict(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42)
GPR_KERNEL = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3)))
GPR_PARAMS = dict(kernel=GPR_KERNEL, n_restarts_optimizer=5, normalize_y=True, random_state=42)
ALL_MODELS = ['RF', 'GB', 'GPR']

# ── Model + pipeline factory ────────────────────────────────────────────
def make_model(name):
    if name == 'RF':  return RandomForestRegressor(**RF_PARAMS)
    if name == 'GB':  return GradientBoostingRegressor(**GB_PARAMS)
    if name == 'GPR': return GaussianProcessRegressor(**GPR_PARAMS)
    raise ValueError(name)

def make_pipeline(name):
    # Scaler + model bundled together so cross-validation refits the
    # scaler fresh on each fold's training data only. This is the fix.
    return Pipeline([('scaler', StandardScaler()), ('model', make_model(name))])

# ── Feature selection: top-N by Spearman |rho|, p<0.05 ──────────────────
def select_features(df, step_um, n_feats=TOP_N_FEATS):
    sub = df[df['step_um'] == step_um]
    meta = ('step_um', 'replicate', 'round', 'burst_index', 'offset_um', 'source_file')
    candidates = [c for c in df.columns if c not in meta]
    corr = {}
    for feat in candidates:
        s = sub.dropna(subset=[feat])
        s = s[np.isfinite(s[feat])]
        if len(s) < 5:
            continue
        rho, p = spearmanr(s['offset_um'], s[feat])
        if np.isnan(rho):
            continue
        corr[feat] = (abs(rho), p)
    ranked = sorted(corr.items(), key=lambda x: x[1][0], reverse=True)
    significant = [f for f, (rho, p) in ranked if p < 0.05]
    return significant[:n_feats] if significant else [f for f, _ in ranked[:n_feats]]

# LOOCV on position-averaged data ─────────────────────────
def loocv_eval(sub, feats, model_name):
    avg = sub.groupby('offset_um')[feats].mean().reset_index().dropna(subset=feats)
    X, y = avg[feats].values, avg['offset_um'].values   # raw, unscaled features
    y_pred = cross_val_predict(make_pipeline(model_name), X, y, cv=LeaveOneOut())
    return dict(n=len(y), y_true=y, y_pred=y_pred,
                r2=r2_score(y, y_pred),
                rmse=np.sqrt(mean_squared_error(y, y_pred)),
                mae=mean_absolute_error(y, y_pred))

# Cross-round CV (train on 2 rounds, test on the 3rd) ─────
def cross_round_eval(sub, feats, model_name):
    replicates = sorted(sub['replicate'].unique())
    all_true, all_pred, fold_r2 = [], [], []
    for test_rep in replicates:
        train_avg = (sub[sub['replicate'] != test_rep].dropna(subset=feats)
                        .groupby('offset_um')[feats].mean().reset_index())
        test_avg = (sub[sub['replicate'] == test_rep].dropna(subset=feats)
                        .groupby('offset_um')[feats].mean().reset_index())
        merged = train_avg.merge(test_avg, on='offset_um', suffixes=('_train', '_test'))
        if len(merged) < 4:
            continue

        X_train = merged[[f + '_train' for f in feats]].values
        X_test  = merged[[f + '_test'  for f in feats]].values
        y_train = merged['offset_um'].values
        y_test  = merged['offset_um'].values

        pipe = make_pipeline(model_name)   # scaler fit on X_train only
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        fold_r2.append(r2_score(y_test, y_pred))
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    if not all_true:
        return None
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    return dict(n=len(all_true), y_true=all_true, y_pred=all_pred, fold_r2=fold_r2,
                r2=r2_score(all_true, all_pred),
                rmse=np.sqrt(mean_squared_error(all_true, all_pred)),
                mae=mean_absolute_error(all_true, all_pred))

# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv(FEATURES_CSV)
    results = []

    for step_um in STEP_SIZES:
        sub = df[df['step_um'] == step_um]
        feats = select_features(df, step_um)
        print(f"\n{step_um} um step - selected features: {feats}")

        for model_name in ALL_MODELS:
            loo = loocv_eval(sub, feats, model_name)
            cr = cross_round_eval(sub, feats, model_name)

            print(f"  {model_name:<4} LOOCV        n={loo['n']:<3} "
                  f"R2={loo['r2']:.3f}  RMSE={loo['rmse']:.0f}um  MAE={loo['mae']:.0f}um")
            results.append(dict(step_um=step_um, model=model_name, protocol='LOOCV',
                                 n=loo['n'], r2=loo['r2'], rmse=loo['rmse'], mae=loo['mae']))

            if cr:
                print(f"  {model_name:<4} CrossRoundCV n={cr['n']:<3} "
                      f"R2={cr['r2']:.3f}  RMSE={cr['rmse']:.0f}um  MAE={cr['mae']:.0f}um  "
                      f"(per-fold R2: {[round(v, 3) for v in cr['fold_r2']]})")
                results.append(dict(step_um=step_um, model=model_name, protocol='CrossRoundCV',
                                     n=cr['n'], r2=cr['r2'], rmse=cr['rmse'], mae=cr['mae']))

    pd.DataFrame(results).to_csv('model_comparison_results.csv', index=False, float_format='%.4f')
    print("\nSaved -> model_comparison_results.csv")
