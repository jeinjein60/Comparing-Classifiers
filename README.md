# Comparing Classifiers: Bank Telemarketing 
 
Compares K-Nearest Neighbors, Logistic Regression, Decision Tree, and SVM on a Portuguese bank's telemarketing data to predict whether a client will subscribe to a term deposit.
 
## Business Problem
Problem: A Portuguese bank runs telemarketing campaigns to sell term deposits, but only 4640 out of 41188 said yes and succeeded (11%). Each call costs agent time and risks annoying customers.
Objective: To build a classifier that predicts, before the call, whether a client will subscribe (y = yes), so the bank can prioritize potential subscribers, cut wasted contacts, and keep the same amount of numbers of successes. Success is judged not just by accuracy but by how well the model finds the 'yes' clients (recall, precision, F1, AUC) since the classes are imbalanced.
 
## Data
- **Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (bank-additional-full.cs`)
- **Size:** 41,188 contacts, 20 features, target `y` (subscribed yes/no)
- **Class balance:** 88.7% "no" / 11.3% "yes"
- **Missing values:** coded as `'unknown'` (most in `default`, 8,597 rows); kept as their own category
- **Excluded:** `duration` (only known after the call, so it would cause data leakage)
- **Features used:** the 7 bank-client features: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- 
## Approach
1. Data understanding: target distribution, subscription rates by feature, age by outcome
2. Inferential statistics: Wilson confidence interval, chi-square tests with Cramér's V, Mann-Whitney U (age), logistic regression odds ratios
3. Preparation: one-hot encoding, scaled `age`, 70/30 stratified train/test split (`random_state=42`)
4. Baseline: majority-class predictor
5. Default-setting comparison of all four models (accuracy and train time)
6. Improvement: grid search (3-fold CV, scored on F1) with class balancing, evaluated with precision, recall, F1, and ROC AUC

   
## Key Results
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline (always "no") | 0.887 | n/a | 0.000 | n/a | n/a |
| Logistic Regression (tuned) | 0.593 | 0.161 | 0.623 | 0.256 | **0.655** |
| Decision Tree (tuned) | 0.609 | 0.157 | 0.568 | 0.247 | 0.647 |
| SVM (tuned) | 0.625 | 0.166 | 0.578 | 0.258 | 0.637 |
| KNN (tuned) | 0.865 | 0.261 | 0.107 | 0.152 | 0.575 |
 
- **Accuracy is misleading:** default models score about 88% but find almost no subscribers (recall below 9%).
- **Tuning and class balancing** raised recall from about 0 to about 0.6 at the cost of accuracy and precision.
- **Best model:** tuned Logistic Regression (highest AUC, simplest, trains in about 0.3 s). The tuned models are close to tied; the SVM took 71 s to train with default settings.
- **Segments:** students (~31%) and retirees (~25%) subscribe at 2-3x the average. Blue-collar, services, and entrepreneurs are lowest.
- **Statistics:** job has the strongest association with subscription (Cramér's V = 0.15). `housing` and `loan` show no significant association. Age differs significantly (p = 0.016) but the effect is small.
- **Limit:** bank-client features alone give modest predictive power (AUC about 0.65).
- 
## Recommendations
- Evaluate with recall, F1, AUC, and lift rather than accuracy.
- Add non-leaky campaign and economic features.
- Keep duration excluded
- Try Random Forest or Gradient Boosting, tune the decision threshold by call cost versus deposit value, and A/B test against the current process.
- Get better data
- Retrain regularly; the data is from 2008-2010.
  
## How to Run
1. Open `ComparingClassifiers_17.ipynb` in Google Colab.
2. Create a `data` folder and add `bank-additional-full.csv` inside of it
**Requirements:** pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn (all preinstalled in Colab). The SVM grid search is slow; it is tuned on a 10,000-row subsample.
 
## Reference
Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems.
