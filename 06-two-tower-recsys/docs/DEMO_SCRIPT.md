# P6 60-second demo script

1. **(0-10 s)** "This recommends movies from MovieLens-1M. Every model is trained on the past and tested on the future, so nothing leaks."
2. **(10-25 s)** Show the results table. "BPR matrix factorisation wins at 0.097 recall@20. My two-tower model gets 0.088, and the bootstrap says that loss is real. Popularity gets 0.092."
3. **(25-40 s)** Show the ablation rows and the head/tail table. "Removing logQ correction halves recall. Side features add 0.012. The two-tower covers 6× more of the catalogue than popularity and finds 3× more long-tail hits than MF. The metric rewards popularity; the model trades some of it for discovery."
4. **(40-55 s)** Start `uvicorn movierec.service:app`. Call `/recommend/1?k=5`, then POST `/recommend/cold` with three liked Disney films. "A brand-new user gets personalised results from three movies and their demographics, with no retraining. MF can't do that."
5. **(55-60 s)** "Next experiment: a tuned popularity prior at serving time and a longer history for heavy users. That's where the tower loses most."
