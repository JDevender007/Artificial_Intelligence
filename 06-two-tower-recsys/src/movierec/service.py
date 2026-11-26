"""FastAPI recommendation service over the exported two-tower model and a FAISS (or NumPy) item index.

    uvicorn movierec.service:app --port 8000          # reads MOVIEREC_SERVE_DIR (default outputs/ml1m/serve)

GET  /recommend/{user_id}?k=10   known MovieLens user: precomputed embedding (history up to the test period)
POST /recommend/cold             new user: gender/age/occupation + liked movie ids -> user tower on the fly
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .data import AGES, N_OCC
from .index import ItemIndex, faiss_available
from .models import load_two_tower


class ColdUser(BaseModel):
    gender: Literal["F", "M"]
    age: int = Field(description=f"MovieLens age code, one of {AGES}")
    occupation: int = Field(ge=0, lt=N_OCC)
    liked_movie_ids: list[int] = Field(default_factory=list, max_length=200)
    k: int = Field(default=10, ge=1, le=100)


class Item(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class Recommendation(BaseModel):
    user: str
    index: str
    items: list[Item]


class Recommender:
    """Loads ``serve_dir`` once; the API and the demo script both use it."""

    def __init__(self, serve_dir: str | Path) -> None:
        d = Path(serve_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.item_raw = np.array(meta["item_raw"])
        self.titles, self.genres = meta["titles"], meta["genres"]
        self.item_of = {int(r): n for n, r in enumerate(self.item_raw)}
        self.user_of = {int(r): n for n, r in enumerate(meta["user_raw"])}
        self.seen = {int(u): set(v) for u, v in meta["seen"].items()}
        self.user_vecs = np.load(d / "user_vecs.npy")
        self.index = ItemIndex(np.load(d / "item_vecs.npy"), "flat" if faiss_available() else "numpy")
        self.model = load_two_tower(torch.load(d / "two_tower.pt", map_location="cpu", weights_only=True))
        self.history_len = self.model.cfg.history_len

    def _items(self, hits: list[tuple[int, float]]) -> list[Item]:
        return [Item(movie_id=int(self.item_raw[i]), title=self.titles[i], genres=self.genres[i], score=s)
                for i, s in hits]

    def for_user(self, user_id: int, k: int) -> list[Item]:
        if user_id not in self.user_of:
            raise KeyError(user_id)
        u = self.user_of[user_id]
        return self._items(self.index.recommend(self.user_vecs[u], k, self.seen.get(u, set())))

    @torch.no_grad()
    def for_cold_user(self, req: ColdUser) -> list[Item]:
        if req.age not in AGES:
            raise ValueError(f"age must be one of {AGES}")
        liked = [self.item_of[m] for m in req.liked_movie_ids if m in self.item_of][-self.history_len:]
        hist = torch.full((1, self.history_len), -1, dtype=torch.long)
        if liked:
            hist[0, -len(liked):] = torch.tensor(liked)
        feat = torch.tensor([[int(req.gender == "M"), AGES.index(req.age), req.occupation]])
        uid = torch.tensor([self.model.n_users])  # "unknown user" id
        q = self.model.user_vec(uid, feat, hist).numpy()[0]
        return self._items(self.index.recommend(q, req.k, set(liked)))


def create_app(serve_dir: str | Path | None = None) -> FastAPI:
    serve_dir = serve_dir or os.environ.get("MOVIEREC_SERVE_DIR", "outputs/ml1m/serve")
    app = FastAPI(title="movierec", version="0.1.0")
    state: dict[str, Any] = {}

    def rec() -> Recommender:
        if "rec" not in state:
            state["rec"] = Recommender(serve_dir)
        return state["rec"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        r = rec()
        return {"status": "ok", "items": len(r.item_raw), "users": len(r.user_of), "index": r.index.kind}

    @app.get("/recommend/{user_id}", response_model=Recommendation)
    def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> Recommendation:
        r = rec()
        try:
            items = r.for_user(user_id, k)
        except KeyError:
            raise HTTPException(404, f"unknown user {user_id}; use POST /recommend/cold") from None
        return Recommendation(user=str(user_id), index=r.index.kind, items=items)

    @app.post("/recommend/cold", response_model=Recommendation)
    def recommend_cold(req: ColdUser) -> Recommendation:
        r = rec()
        try:
            items = r.for_cold_user(req)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return Recommendation(user="cold", index=r.index.kind, items=items)

    return app


def __getattr__(name: str) -> Any:  # `uvicorn movierec.service:app` builds the app on first access
    if name == "app":
        return create_app()
    raise AttributeError(name)
