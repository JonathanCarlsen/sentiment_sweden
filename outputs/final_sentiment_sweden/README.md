# Final Sentiment Sweden Outputs

This folder is reserved for the thesis-facing outputs produced by:

`nordic_sentiment/notebooks/01_sweden/FINAL_SENTIMENT_SWEDEN.ipynb`

The notebook focuses on the selected final sentiment index:

`SENT_ORTH_DIV_PREMIUM`

It also runs the same predictive regressions with the raw non-orthogonalized index:

`SENT_DIV_PREMIUM`

This comparison is included only to quantify the empirical effect of macro-orthogonalizing the sentiment proxy set.

Expected output groups:

- final sentiment index tables and selected PCA diagnostics
- raw versus macro-orthogonalized sentiment comparison tables and figures
- dividend-premium proxy source and construction diagnostics
- lagged characteristic portfolio returns and directional spreads
- 1-, 3-, 6-, and 12-month predictive regression panels
- HAC/Newey-West predictive regression results
- moving-block bootstrap robustness columns
- compact thesis figures for the final index and predictive t-statistics

The broader exploratory audit outputs have been archived under `archive/outputs/`.
