# P4 interview questions

**1. Why macro-F1 and not accuracy?**
Every intent counts equally in macro-F1. The test set is balanced (40 per intent), so the two numbers come out close here. On real traffic, though, rare intents such as `compromised_card` matter more than their frequency, and macro-F1 stops a weak rare class from hiding behind easy frequent ones.

**2. How did you make sure the TF-IDF baseline was not a straw man?**
I used word 1-2-grams plus char 2-5-grams, sublinear TF and a C grid chosen on val. When the first grid picked its top value (C=20), I widened the grid and reran it, so the selected C is an interior optimum. I counted the spelling variants in the data ("top up" / "top-up" / "topup") to justify the char n-grams.

**3. Your TF-IDF std is 0.000 over 3 seeds. Is that a bug?**
No. Logistic regression with lbfgs is a convex problem with a deterministic solver, and the split is fixed, so the seed changes nothing. The seeds matter for DistilBERT (initialisation of the classifier head, data order, dropout). For the TF-IDF vs DistilBERT question I use a paired bootstrap over test examples, which measures the uncertainty that comes from the test sample.

**4. How do you know the DistilBERT gain is real?**
I don't claim it is. The mean gap over 3 seeds is +0.008 macro-F1, barely above DistilBERT's seed std of 0.006. A paired bootstrap on seed 0 (1,000 test resamples) gives +0.0019 with a 95% CI of [-0.0078, +0.0115], and 34% of resamples show no gain. Seed 0 is the weakest seed, so that is the pessimistic case, but the CI still includes zero. What *is* clearly better is calibration: ECE 0.023 vs 0.063. The agreement table also shows the models make different mistakes (101 vs 108 messages that only one model gets right), which points to an ensemble as a next experiment.

**5. What would you change before putting this in production?**
I would add an out-of-scope class trained with real "none of these" messages instead of only a confidence threshold. I would also calibrate the model (temperature scaling, checked with ECE), mask PII before logging, and monitor the predicted-intent distribution for drift. On latency: on a 4-core CPU, DistilBERT is 29 ms per request vs 9.4 ms for TF-IDF, and 30× slower in batches. On a T4 it takes 5.6 ms. Without a GPU I would ship TF-IDF and keep DistilBERT for the low-confidence cases only.
