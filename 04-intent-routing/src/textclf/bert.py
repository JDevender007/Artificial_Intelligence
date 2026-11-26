"""DistilBERT fine-tuning with a hand-written loop (fp16 AMP, linear warmup/decay, early stopping on val
macro-F1). torch and transformers are imported lazily so the rest of the package runs without them."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from .config import BertConfig
from .metrics import classification_metrics
from .utils import get_logger, set_seed

log = get_logger(__name__)


def linear_warmup_lambda(total_steps: int, warmup_steps: int):
    """LR multiplier: 0 -> 1 linearly over ``warmup_steps``, then linearly down to 0 at ``total_steps``."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return f


class BertPredictor:
    def __init__(self, model, tokenizer, cfg: BertConfig, device: str) -> None:
        self.model, self.tokenizer, self.cfg, self.device = model, tokenizer, cfg, device

    def to(self, device: str) -> BertPredictor:
        self.model.to(device)
        self.device = device
        return self

    def predict_proba(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        import torch

        bs = batch_size or self.cfg.eval_batch_size
        self.model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                enc = self.tokenizer(list(texts[i:i + bs]), padding=True, truncation=True,
                                     max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self.cfg.fp16 and self.device.startswith("cuda")):
                    logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.num_labels))

    def save(self, path: str | Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def fit_bert(cfg: BertConfig, train_texts: list[str], train_y: np.ndarray, val_texts: list[str],
             val_y: np.ndarray, labels: list[str], seed: int) -> tuple[BertPredictor, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={n: i for i, n in enumerate(labels)}).to(device)
    pred = BertPredictor(model, tok, cfg, device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    steps_per_epoch = -(-len(train_texts) // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_warmup_lambda(total, int(cfg.warmup_ratio * total)))
    use_amp = cfg.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(seed)
    y_t = torch.as_tensor(train_y)

    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train_texts), generator=gen).tolist()
        loss_sum = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            enc = tok([train_texts[j] for j in idx], padding=True, truncation=True, max_length=cfg.max_length,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**enc, labels=y_t[idx].to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(idx)
        val_f1 = classification_metrics(val_y, pred.predict_proba(val_texts).argmax(1), len(labels))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(order), "val_macro_f1": val_f1})
        log.info("distilbert seed %d epoch %d loss %.4f val macro-F1 %.4f", seed, epoch, loss_sum / len(order), val_f1)
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return pred, {"best_epoch": best_epoch, "val_macro_f1": best_f1, "history": history,
                  "fit_s": time.perf_counter() - t0, "params_m": n_params / 1e6, "device": device}
