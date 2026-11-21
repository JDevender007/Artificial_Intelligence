# P4 60-second demo script

1. **(0-10 s)** "A bank gets thousands of messages a day. This routes each one to one of 77 intents, so it reaches the right team."
2. **(10-25 s)** Show the results table: majority, TF-IDF + LR and DistilBERT, with macro-F1, accuracy and ECE as mean ± std over 3 seeds. [ ] Read out the DistilBERT gain and its bootstrap CI.
3. **(25-40 s)** Show the latency table: TF-IDF runs at about 2.3 ms per message on one CPU request. [ ] Give the DistilBERT CPU and GPU numbers. Then the question: is the quality gain worth that cost?
4. **(40-55 s)** Run `python scripts/predict.py --run outputs/banking77 --model tfidf_lr --threshold 0.5 "I'm starting to think my card is lost because it still hasn't arrived"`. It shows a low-confidence `card_arrival` / `lost_or_stolen_card` split, and the message is handed to a human.
5. **(55-60 s)** "The model card lists the weakest intents and the limits: a closed set, one domain, and PII handling."
