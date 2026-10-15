# Data Exploration and Leakage Audit

## 1. Objective

Before training a credit default prediction model, we need to understand the dataset, define the prediction target, and identify columns that could introduce data leakage.

This stage has four objectives:

1. Understand the distribution of loan statuses.
2. Examine how default rates vary by loan issue year.
3. Define a consistent binary target variable.
4. Audit individual features for suspicious predictive power and potential leakage.

The goal is not to select the final model or prove that every feature is safe. It is to understand the data and establish a defensible feature-selection policy before training.

## 2. Understanding the Prediction Problem

The project aims to predict whether a loan will default.

Each row represents a loan, and the dataset contains information about borrowers, loan terms, repayment activity, and loan outcomes.

The model learns patterns from historical loans whose outcomes are known. It then uses those patterns to estimate the default risk of loans it has not seen before.

However, a critical distinction must be established first:

**At what point in the loan lifecycle is the prediction supposed to be made?**

For this project, the initial objective is to estimate default risk using information available at or before loan origination. This approximates an application-time credit-risk assessment.

That decision determines which columns are eligible to become model features.

For example:

| Feature | Potentially available at origination? | Initial decision |
|---|---|---|
| Annual income (`annual_inc`) | Yes, if recorded at application time | Potentially allowed |
| Loan amount (`loan_amnt`) | Yes | Potentially allowed |
| Interest rate (`int_rate`) | Depends on when it is recorded | Verify timing |
| Loan grade (`grade`) | Depends on when it is assigned | Verify timing |
| Total payments received (`total_pymnt`) | No; accumulates during repayment | Exclude |
| Recoveries (`recoveries`) | Generally known only after delinquency or default-related events | Exclude |
| Latest credit score (`last_fico_range_high`) | May reflect later credit history | Exclude unless its timestamp proves otherwise |

These decisions must be validated against the dataset's documentation and feature timestamps. A column's name alone does not establish when its value became available.

## 3. Understanding Data Leakage

Data leakage occurs when information that would not legitimately be available at prediction time influences model training or evaluation.

A model affected by leakage can achieve excellent test metrics while performing poorly in a real deployment.

### Example: Total payments received

Suppose a loan has a total payment amount of ₹100,000 recorded after several years.

That amount reflects repayment activity that occurred after the loan originated. A loan that was repaid successfully may have accumulated substantial payments, while a defaulted loan may have accumulated fewer payments.

If the model uses `total_pymnt` to predict default at origination, it indirectly receives information about the future outcome.

The model may appear highly accurate because it is exploiting the loan's subsequent history rather than learning how to assess risk at application time.

### Types of leakage to investigate

- **Target leakage:** A feature directly or indirectly encodes the outcome being predicted.
- **Temporal leakage:** A feature contains information recorded after the prediction timestamp.
- **Train-test contamination:** Information from the evaluation data influences preprocessing, feature selection, or model development.
- **Duplicate or related-record leakage:** Closely related records appear across training and testing sets, allowing the model to exploit similarities that will not generalize to genuinely new cases.

A feature can be highly predictive without being leakage. The decisive question is whether that information would have been available at the intended prediction time.

## 4. Exploring Loan Statuses

The first step is to inspect the distribution of `loan_status`.

Grouping records by status helps establish:

- Which outcomes are present in the dataset.
- How many loans fall into each outcome category.
- Which statuses can be treated as successful repayment or default for this project.
- Which statuses are unresolved, ongoing, or otherwise unsuitable for binary classification.

The status distribution also reveals whether the dataset contains a substantial number of loans whose outcomes are not yet known.

### Defining good and bad outcomes

The project uses two explicitly defined groups:

- `GOOD_STATUSES`: statuses representing successful loan outcomes.
- `BAD_STATUSES`: statuses representing default or other agreed-upon adverse outcomes.

The exact membership of these groups must be documented and justified using the dataset's status definitions.

Statuses that belong to neither group should not automatically be classified as good. For example, an ongoing loan has not necessarily been repaid successfully.

These unresolved records should be excluded from the initial binary-target analysis unless a separate, defensible labeling strategy is established.

## 5. Examining Default Rates by Issue Year

The next step is to calculate the bad-loan rate for each loan issue year.

For a given year:

**Bad Rate** = Number of Eligible Bad Loans / Number of Eligible Good and Bad Loans

The result is multiplied by 100 when expressed as a percentage.

For example, suppose a year contains:
- 800 eligible good loans
- 200 eligible bad loans

Then:

**Bad Rate** = 200 / (800 + 200) = **20%**

This calculation uses only records with known outcomes included in the defined good and bad groups.

### Why analyze the rate by year?

Default rates can vary over time because of changes in:

- Lending standards and borrower composition.
- Economic conditions and unemployment.
- Interest rates and credit availability.
- Loan products and underwriting policies.
- The maturity of loans in the dataset.

A year-by-year analysis helps identify patterns that deserve further investigation. It can also reveal whether the dataset contains substantial temporal variation that should influence the validation strategy.

### Important limitation: Loan maturity

Recent loans may not have had enough time to complete their repayment period. Comparing their observed bad rate directly with that of older loans can be misleading.

For example, a recent loan that has not defaulted yet is not necessarily a successful loan.

Therefore, the analysis should be interpreted in light of how loan outcomes are recorded and how much observation time each issue-year cohort has had.

## 6. Creating the Binary Target

The target variable, `bad`, converts loan outcomes into a format suitable for binary classification.

The intended mapping is:

| Original loan status | `bad` |
|---|---:|
| Status belongs to `BAD_STATUSES` | 1 |
| Status belongs to `GOOD_STATUSES` | 0 |
| Status belongs to neither group | Excluded from this analysis |

Here, `1` represents an adverse loan outcome, and `0` represents a good loan outcome.

The target is created only after the good and bad status definitions have been established.

### Why is the target separate from the features?

The target is the outcome the model must learn to predict. Features are the information the model is allowed to use to make that prediction.

Keeping these roles separate prevents accidental inclusion of the outcome itself as an input.

The original `loan_status` column should therefore not be included as a model feature when the target is derived from it.

## 7. Auditing Individual Features Using AUC

After defining the target, the project evaluates how well individual numeric columns distinguish between good and bad loans.

The metric used for this initial audit is the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

### What does AUC measure?

AUC measures how well a score ranks positive examples above negative examples.

In this project:

- Positive class: `bad = 1`
- Negative class: `bad = 0`

An AUC of 0.50 indicates no ranking discrimination beyond random ordering. An AUC of 1.00 indicates perfect ranking discrimination for the evaluated sample.

An AUC below 0.50 indicates that the score tends to rank the classes in the opposite direction. Reversing a score's direction can change its AUC to approximately `1 - AUC`.

### Interpreting example values

| AUC | Interpretation |
|---:|---|
| 0.50 | No useful ranking discrimination |
| 0.60 | Weak discrimination |
| 0.70 | Moderate discrimination |
| 0.90 | Strong discrimination |
| 0.99 | Near-perfect discrimination; investigate carefully |

These are illustrative interpretations, not universal performance thresholds.

### Calculating feature strength

For convenient ranking, the audit uses:
**Feature Strength** = 2 × |AUC − 0.5|

This rescales the distance from random discrimination to a range from 0 to 1.

Examples:
- AUC = 0.50 → strength = 0.00
- AUC = 0.70 → strength = 0.40
- AUC = 0.98 → strength = 0.96

This is a heuristic for prioritizing columns for investigation. It is not a calibrated measure of feature importance, causal influence, or leakage probability.

### Handling missing values during the audit

AUC calculation requires usable feature values. For the initial audit, missing numeric values can be filled with the feature's median so that the calculation can be performed consistently.

The resulting audit should also report the percentage of missing values for each feature:

**Missing Percentage** = (Number of Missing Values / Total Number of Rows) × 100

Median imputation here is an exploratory convenience, not a final preprocessing decision. In the modeling pipeline, imputation parameters must be learned from the training data only and then applied to validation and test data.

## 8. Interpreting the Leakage Audit

The audit report combines each numeric feature's AUC, calculated strength, and missing-value percentage.

Features can then be sorted by strength to identify columns that deserve closer examination.

A suspiciously high AUC can indicate:

- A feature that directly encodes the target.
- Information recorded after the prediction timestamp.
- A variable that closely reflects the consequences of default.
- A genuinely powerful predictor available at the correct time.
- A data-quality or labeling issue.

**High AUC is a warning signal, not proof of leakage. Low AUC is not proof that a feature is safe.**

For every suspicious feature, investigate its business meaning, collection process, timestamp, and relationship to the target before making an exclusion decision.

The audit also focuses on numeric columns. It is not a complete leakage detector: categorical features, identifiers, text fields, duplicate records, and relationships between multiple features may require separate checks.

## 9. Defining an Initial Feature Policy

The project maintains two conceptual groups:

- `ALLOWED`: features that may be considered for modeling.
- `BANNED`: features that should not be used for the intended prediction task.

The lists provide an explicit feature policy and help make modeling decisions reproducible.

However, declaring a list does not automatically enforce it. The modeling pipeline must explicitly select the approved columns and verify that banned columns are absent.

The initial policy should exclude:

- The original target-defining status field.
- Direct target labels or outcome-derived fields.
- Repayment totals accumulated after origination.
- Recovery amounts and other post-default information.
- Credit-history measurements recorded after the intended prediction timestamp.
- Identifiers that can act as proxies for records, outcomes, or data collection processes.

Features such as loan grade, interest rate, and installment amount require particular care. They may be valid in a model built at a specific underwriting stage, but only if their values are available at that stage and do not encode information introduced later.

The feature policy should be revised when evidence from the dataset's documentation or audit contradicts an initial assumption.

## 10. Sampling and Reproducibility

For exploratory analysis, the project may use a random sample of eligible good and bad loans to reduce computation time.

Sampling is useful when the source table contains a large number of records. However, the sampling method affects the interpretation of the results.

The audit should record:

- The source table and query filters.
- The sample size.
- The random seed, if applicable.
- The target-label definitions.
- The feature exclusions.
- The date or version of the dataset.

A random sample of completed loans may not represent the full population of loans at application time. In particular, restricting the analysis to completed outcomes can introduce selection effects because recent or ongoing loans may be excluded.

For reproducibility, the project should use consistent eligibility rules and document any limitations introduced by the sample.

## 11. Limitations of This Audit

This stage is an initial data-quality and leakage investigation, not a complete model validation process.

Key limitations include:

1. **Single-feature analysis:** Individual AUC scores do not capture interactions or combined predictive effects between features.
2. **Numeric-only scope:** Categorical variables, identifiers, and text-based fields need additional investigation.
3. **Timing uncertainty:** Column names and predictive power cannot establish when a value became available.
4. **Outcome maturity:** Loans with incomplete observation periods may not have reliable final labels.
5. **Sampling effects:** A sample of loans with known outcomes may differ from the population that would be scored in production.
6. **Exploratory imputation:** Median imputation used for the audit must not leak information from validation or test sets into training.
7. **No causal conclusions:** AUC measures ranking discrimination, not whether a feature causes default.

These limitations should be addressed before the final feature set and evaluation strategy are considered reliable.

## 12. Next Steps

After completing the exploration and leakage audit:

1. Review the loan-status distribution and confirm the target definitions.
2. Examine bad rates by issue year and assess differences in loan maturity.
3. Review the highest-strength features and investigate suspicious columns individually.
4. Finalize the prediction timestamp and feature-availability rules.
5. Create an explicit feature list that excludes banned columns.
6. Split data into training, validation, and test sets using a strategy appropriate to the prediction setting. For a future-facing credit-risk model, a chronological split is often more realistic than a purely random split.
7. Fit preprocessing steps on training data only.
8. Evaluate the model on data that was not used for training, feature selection, or threshold tuning.

The purpose of this stage is to establish a trustworthy modeling dataset—not to maximize AUC at any cost.

## Core Principle

**A credit-risk model must learn from historical outcomes without using information that would only become available after the prediction is made.**

Every feature should have a clear definition, a defensible availability timestamp, and a documented reason for inclusion. Predictive performance matters only when it reflects information that can legitimately be used in the real-world decision.