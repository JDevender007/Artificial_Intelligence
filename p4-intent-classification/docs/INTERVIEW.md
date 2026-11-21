# P4 interview questions

**1. Why macro-F1 and not accuracy?**
Every intent counts equally in macro-F1. The test set is balanced (40 per intent), so the two numbers come out close here. On real traffic, though, rare intents such as `compromised_card` matter more than their frequency, and macro-F1 stops a weak rare class from hiding behind easy frequent ones.

**2. How did you make sure the TF-IDF baseline was not a straw man?**
I used word 1-2-grams plus char 2-5-grams, sublinear TF and a C grid chosen on val. When the first grid picked its top value (C=20), I widened the grid and reran it, so the selected C is an interior optimum. I counted the spelling variants in the data ("top up" / "top-up" / "topup") to justify the char n-grams.

**3. Your TF-IDF std is 0.000 over 3 seeds. Is that a bug?**
No. Logistic regression with lbfgs is a convex problem with a deterministic solver, and the split is fixed, so the seed changes nothing. The seeds matter for DistilBERT (initialisation of the classifier head, data order, dropout). For the TF-IDF vs DistilBERT question I use a paired bootstrap over test examples, which measures the uncertainty that comes from the test sample.

**4. How do you know the DistilBERT gain is real?**
[ ] Fill in from the paired bootstrap in reports/error_analysis.md: the gain and its 95% CI. Also check that the gain is larger than the seed std.

**5. What would you change before putting this in production?**
I would add an out-of-scope class trained with real "none of these" messages instead of only a confidence threshold. I would also calibrate the model (temperature scaling, checked with ECE), mask PII before logging, and monitor the predicted-intent distribution for drift. [ ] Add the latency trade-off once the CPU vs GPU numbers are in.
