"""
Explore historical loan outcomes and audit features for leakage.

Objectives:
1. Summarize loan statuses and bad rates by issue year.
2. Calculate single-feature AUC for numeric columns.
3. Flag suspiciously predictive features for investigation.
4. Export the audit report for review.

Core principle:
Every feature must be available at the intended prediction time.
High AUC is a warning signal, not proof of leakage.
See docs/data_exploration_and_leakage.md for the full theory.
"""