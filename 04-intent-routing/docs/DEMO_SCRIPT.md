# P4 60-second demo script

1. **(0-10 s)** "A bank gets thousands of messages a day. This routes each one to one of 77 intents, so it reaches the right team."
2. **(10-25 s)** Show the results table: majority, TF-IDF + LR and DistilBERT, with macro-F1, accuracy and ECE as mean ± std over 3 seeds. DistilBERT 0.921 vs TF-IDF 0.913 macro-F1, but the paired-bootstrap 95% CI for the gain is [-0.008, +0.012], so it includes zero. The clear gain is calibration: ECE 0.023 vs 0.063.
3. **(25-40 s)** Show the latency table: on the same 4-core Kaggle CPU, TF-IDF takes 9.4 ms per request and DistilBERT 29 ms; on a T4, DistilBERT takes 5.6 ms. Then the question: is a gain within noise worth that cost? On CPU, no; with a GPU, or when you need trustworthy confidences, yes.
4. **(40-55 s)** Run `python scripts/predict.py --run outputs/banking77 --model tfidf_lr --threshold 0.5 "I'm starting to think my card is lost because it still hasn't arrived"`. It shows a low-confidence `card_arrival` / `lost_or_stolen_card` split, and the message is handed to a human.
5. **(55-60 s)** "The model card lists the weakest intents and the limits: a closed set, one domain, and PII handling."
