
import sys
sys.path.insert(0, '.')

import pandas as pd
from sklearn.linear_model import LinearRegression

from features import fit_clean_stats, prepare

train_full = pd.read_csv('data/train_test.csv')
val = pd.read_csv('data/validation.csv')
dec = pd.read_csv('data/december_chart_inputs.csv')
template = pd.read_csv('data/validation_predictions_template.csv')

stats = fit_clean_stats(train_full)
Xtr = prepare(train_full, stats)
ytr = train_full["posted_rate"].values

model = LinearRegression()
model.fit(Xtr, ytr)

# --- 1. validation_predictions.csv ---
Xval = prepare(val, stats)
val_preds = model.predict(Xval)

out = template.copy()
pred_map = dict(zip(val["load_id"], val_preds))
out["predicted_rate"] = out["load_id"].map(pred_map)
assert out["predicted_rate"].isna().sum() == 0, "missing predictions!"
n_nonpositive = (out["predicted_rate"] <= 0).sum()
if n_nonpositive:
    print(f"WARNING: clipping {n_nonpositive} non-positive predictions to a $1 floor")
    out["predicted_rate"] = out["predicted_rate"].clip(lower=1.0)
out.to_csv('validation_predictions.csv', index=False)
print("validation_predictions.csv:", out.shape, "min/max:", out['predicted_rate'].min(), out['predicted_rate'].max())

# --- 2. december_chart_inputs.csv (filled) ---
Xdec = prepare(dec, stats)
dec_preds = model.predict(Xdec)
dec_out = dec.copy()
dec_out["predicted_rate"] = dec_preds
dec_out.to_csv('data/december_chart_inputs.csv', index=False)
print()
print(dec_out[["date", "predicted_rate"]])