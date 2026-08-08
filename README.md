# Freight Rate Prediction Challenge
## Reproducing this solution

1. `pip install -r requirements.txt`
2. Open `Analysis.ipynb` and run all cells top to bottom. This will:
   - Explore and clean `data/train_test.csv`
   - Engineer features (`src/features.py`)
   - Time-split into Jan–Aug train / Sep–Oct holdout and compare
     Linear Regression vs. Gradient Boosting
   - Fit the final model on the full labeled set
   - Write `validation_predictions.csv` and fill `data/december_chart_inputs.csv`
3. Run the scorer to validate outputs and produce the chart:
   `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv`
