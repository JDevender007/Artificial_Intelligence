# P6 career assets

**Repo folder:** `06-movie-recsys` (suggested standalone name: `movielens-two-tower-faiss`)

**Resume bullet**
- Built a MovieLens-1M recommender on a leak-free temporal split: popularity, BPR-MF and a feature two-tower model with logQ-corrected sampled softmax, served through a FAISS HNSW index (0.99 recall vs exact) and a FastAPI cold-start endpoint. Ablations and a paired bootstrap showed BPR-MF wins on recall@20 (0.097 vs 0.088), while the two-tower covers 6× more of the catalogue than popularity and logQ correction is worth +0.043 recall.

**LinkedIn post (draft)**
> My two-tower recommender lost to matrix factorisation, and I think that's the interesting part.
>
> On MovieLens-1M with a strict time split (train on the past, test on the future):
> - BPR-MF: 0.097 recall@20. Two-tower with side features: 0.088. Popularity: 0.092. A paired bootstrap says the MF win is real.
> - Removing logQ correction from the two-tower halves its recall and pushes 76% of its recommendations into the long tail.
> - The two-tower covers 34% of the catalogue at k=20, against 6% for popularity, and it can recommend to brand-new users from 3 liked movies.
>
> Offline recall rewards whatever was already popular. Knowing which metric to trust is the real job.
>
> Code, ablations and the FAISS/FastAPI serving: [repo link]
