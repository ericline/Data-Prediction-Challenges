# Competitive Predictive Modeling

A collection of advanced predictive regression modeling challenges submitted on Kaggle.

## Prediction Challenge I — Movie Rating Classification

Built a decision tree classifier (rpart) to predict movie ratings ("Great" vs other) from the 2023 movies dataset. 
Engineered a weighted Delta feature (3Audience - 1.5Income) that improved class separation, computed Audience/Income ratios, and visualized feature distributions with boxplots and barplots across Genre and Content categories. 
Used 10-fold cross-validation at 80/20 split to evaluate model accuracy.

- Dataset: movies2023Tr.csv — Genre, Content, Audience, Income, RATING
- Approach: Feature engineering (Delta, Audience/Income ratio) + decision tree classification
- Validation: 10-fold cross-validation via CrossValidation package
- Stack: R, rpart, rpart.plot, CrossValidation

## Prediction Challenge II — Post-College Salary Prediction

Predicted post-college salaries using stratified linear regression, building 7 separate models across 6 major categories (Humanities, STEM, Vocational, Professional, Business, Other). 
Performed extensive EDA with scatter plots and boxplots per major, then engineered custom transformations for each stratum: GPA² and GPA²·log(Tuition) for Humanities, LinkedIn² for Other, GPA^(1/10) for Professional, GPA/Tuition and GPA·LinkedIn interactions for Vocational, and DOB parity-based sub-models for Business. 
Used automated bidirectional stepwise selection (AIC) to identify significant predictors, then combined all stratum predictions into a single submission.

- Dataset: incomeTrain2023.csv — Major, GPA, Tuition, LinkedIN, DOB, College_location, Salary
- Approach: Stratified regression (7 models across 6 majors) with per-stratum feature engineering
- Key transformations: GPA², GPA²·log(Tuition), log(GPA/Tuition), LinkedIn², GPA^(1/10), GPA·LinkedIn, GPA/Tuition
- Stack: R, rpart, Metrics (MSE), CrossValidation

## Prediction Challenge III — Loan Amount Prediction

Built an ensemble of 8 stratified linear regression models for loan amount prediction, segmenting borrowers by all combinations of home ownership (Yes/No), car ownership (Yes/No), and debt level (0 vs 100K). 
Applied Income^(1/2) power transformations for the no-home/no-car segments to capture nonlinear income-amount relationships, while using simple linear Income models for asset-owning segments. 
Validated with manual 80/20 train/test cross-validation, achieving per-segment MSE ranging from 85-112 and overall cross-validated MSE of ~99.89.

- Dataset: LoanTrain2023b.csv — Home, Car, Debt, Income, Amount
- Approach: 8 stratified regression models segmented by Home x Car x Debt combinations
- Key transformations: Income^(1/2) power transformation for no-asset segments
- Validation: Manual 80/20 cross-validation, per-segment MSE evaluation (85-112 range)
- Stack: R, rpart, Metrics (MSE), CrossValidation


