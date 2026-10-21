"""Does self-supervised pretraining help when labels are scarce? (5G-NIDD)

The first SSL evaluation (src/ssl) trained a linear classifier on all labelled
flows, where SSL is not expected to help, and it tied with raw features. SSL earns
its keep when many flows are unlabelled and only a few are labelled, so this
script measures exactly that:

  data      5G-NIDD, identifier columns (row index, Seq, Offset) dropped, exact
            duplicate flows removed before splitting, stratified 80/20 split
  pretrain  masked-feature modelling on the 80% pool with NO labels: 30% of the
            features are replaced by values drawn from other rows, and the encoder
            learns to rebuild the original values and to spot which were replaced
  evaluate  attack-type classification (6 classes, macro-F1) with 50 to all
            labelled flows, 5 random label draws each:
              raw features + logistic regression / MLP from scratch
              SSL embedding + logistic regression
              SSL encoder fine-tuned (same MLP shape, pretrained start)
              LightGBM on raw features (strong supervised reference)

Run: python src/unified/ssl_nidd.py   ->  reports/unified/ssl_nidd.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

DATA = Path(r"E:\FYP DATA\6G\data\raw\5g_nidd\Combined.csv")
OUT = Path(__file__).resolve().parents[2] / "reports" / "unified" / "ssl_nidd.json"
DROP = ["Unnamed: 0", "Seq", "Offset", "Label", "Attack Type", "Attack Tool"]
CATEGORICAL = ["Proto", "State", "Cause", "sDSb", "dDSb"]
LABEL_BUDGETS = [50, 200, 1000, 5000, 20000, None]   # None = every labelled flow in the pool
SEEDS = [0, 1, 2, 3, 4]
torch.set_num_threads(8)


def load():
    df = pd.read_csv(DATA, low_memory=False)
    y_text = df["Attack Type"].astype(str)
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in CATEGORICAL]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=np.float32)
    keep = ~X.assign(_y=y_text).duplicated()
    X, y_text = X[keep], y_text[keep]
    classes = sorted(y_text.unique())
    y = y_text.map({c: i for i, c in enumerate(classes)}).to_numpy()
    numeric = [c for c in num]
    Xv = X.to_numpy(np.float32)
    idx_num = [X.columns.get_loc(c) for c in numeric]
    Xv[:, idx_num] = np.sign(Xv[:, idx_num]) * np.log1p(np.abs(Xv[:, idx_num]))   # heavy-tailed counts
    return Xv, y, classes, int(len(df)), int(keep.sum())


class Encoder(nn.Module):
    def __init__(self, d_in, d_emb=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d_emb))

    def forward(self, x):
        return self.net(x)


def pretrain(Xp, epochs=12, mask_p=0.3, seed=0):
    torch.manual_seed(seed)
    d = Xp.shape[1]
    enc = Encoder(d)
    recon, maskhead = nn.Sequential(nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, d)), nn.Sequential(nn.ReLU(), nn.Linear(64, d))
    params = list(enc.parameters()) + list(recon.parameters()) + list(maskhead.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    X = torch.from_numpy(Xp)
    n, hist = len(X), []
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            xb = X[perm[i:i + 1024]]
            m = (torch.rand_like(xb) < mask_p).float()
            shuffled = xb[torch.randint(0, len(xb), (len(xb),))]   # VIME-style corruption: values from other rows
            xc = xb * (1 - m) + shuffled * m
            z = enc(xc)
            loss = nn.functional.mse_loss(recon(z), xb) + nn.functional.binary_cross_entropy_with_logits(maskhead(z), m)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        hist.append(round(tot / n, 5))
        print(f"  pretrain epoch {ep + 1}/{epochs}: loss {hist[-1]} ({time.time() - t0:.0f}s)", flush=True)
    return enc, hist


def pretrain_scarf(Xp, epochs=15, corrupt=0.6, temp=0.5, seed=0):
    """SCARF (Bahri et al., ICLR 2022): contrast each flow with a copy whose features are
    partly replaced by values from other flows (InfoNCE over the batch)."""
    torch.manual_seed(seed)
    d = Xp.shape[1]
    enc = Encoder(d)
    proj = nn.Sequential(nn.ReLU(), nn.Linear(64, 64))
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
    X = torch.from_numpy(Xp)
    n, hist = len(X), []
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        perm = torch.randperm(n)
        for i in range(0, n - 511, 512):
            xb = X[perm[i:i + 512]]
            m = (torch.rand_like(xb) < corrupt).float()
            other = X[torch.randint(0, n, (len(xb),))]
            xc = xb * (1 - m) + other * m
            za = nn.functional.normalize(proj(enc(xb)), dim=1)
            zb = nn.functional.normalize(proj(enc(xc)), dim=1)
            logits = za @ zb.T / temp
            target = torch.arange(len(xb))
            loss = (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        hist.append(round(tot / n, 5))
        print(f"  scarf epoch {ep + 1}/{epochs}: loss {hist[-1]} ({time.time() - t0:.0f}s)", flush=True)
    return enc, hist


def embed(enc, X):
    with torch.no_grad():
        return torch.cat([enc(torch.from_numpy(X[i:i + 65536])) for i in range(0, len(X), 65536)]).numpy()


def mlp_train(Xtr, ytr, n_classes, init: Encoder | None, seed, steps=600):
    torch.manual_seed(seed)
    enc = Encoder(Xtr.shape[1])
    if init is not None:
        enc.load_state_dict(init.state_dict())
    model = nn.Sequential(enc, nn.ReLU(), nn.Linear(64, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if init is None else 3e-4, weight_decay=1e-5)
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    w = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    lossf = nn.CrossEntropyLoss(weight=(w.sum() / w) / n_classes)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(X), (min(256, len(X)),), generator=g)
        opt.zero_grad(); lossf(model(X[b]), y[b]).backward(); opt.step()
    return model


def predict_mlp(model, X):
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(X[i:i + 65536])).argmax(1) for i in range(0, len(X), 65536)]).numpy()


def sample_labels(y, budget, seed):
    """Stratified-as-possible draw of `budget` labels, at least 2 per class."""
    rng = np.random.default_rng(seed)
    if budget is None:
        return np.arange(len(y))
    idx = []
    classes = np.unique(y)
    per = max(2, budget // (4 * len(classes)))
    for c in classes:
        pool = np.flatnonzero(y == c)
        idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False))
    rest = np.setdiff1d(np.arange(len(y)), idx)
    idx.extend(rng.choice(rest, size=max(0, budget - len(idx)), replace=False))
    return np.array(idx)


def main():
    t_start = time.time()
    X, y, classes, n_rows, n_unique = load()
    print(f"loaded {n_rows:,} flows, {n_unique:,} after removing duplicates, {X.shape[1]} features, classes {classes}", flush=True)
    Xp, Xt, yp, yt = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-6
    Xp, Xt = ((Xp - mu) / sd).astype(np.float32), ((Xt - mu) / sd).astype(np.float32)
    method = sys.argv[1] if len(sys.argv) > 1 else "masked"
    enc, hist = pretrain_scarf(Xp) if method == "scarf" else pretrain(Xp)
    Ep, Et = embed(enc, Xp), embed(enc, Xt)
    score = lambda pred: {"macro_f1": round(float(f1_score(yt, pred, average="macro")), 4), "accuracy": round(float(accuracy_score(yt, pred)), 4)}
    results = {}
    for budget in LABEL_BUDGETS:
        key = "all" if budget is None else str(budget)
        seeds = SEEDS if budget is not None and budget <= 5000 else SEEDS[:2]
        runs = {m: [] for m in ("raw_logreg", "ssl_logreg", "raw_plus_ssl_logreg", "raw_mlp", "ssl_finetune", "raw_lightgbm")}
        for s in seeds:
            i = sample_labels(yp, budget, s)
            lr = lambda A: LogisticRegression(max_iter=300, class_weight="balanced").fit(A[i], yp[i])
            runs["raw_logreg"].append(score(lr(Xp).predict(Xt)))
            runs["ssl_logreg"].append(score(lr(Ep).predict(Et)))
            runs["raw_plus_ssl_logreg"].append(score(lr(np.hstack([Xp, Ep])).predict(np.hstack([Xt, Et]))))
            steps = 600 if budget is not None and budget <= 5000 else 3000
            runs["raw_mlp"].append(score(predict_mlp(mlp_train(Xp[i], yp[i], len(classes), None, s, steps), Xt)))
            runs["ssl_finetune"].append(score(predict_mlp(mlp_train(Xp[i], yp[i], len(classes), enc, s, steps), Xt)))
            import lightgbm as lgb
            gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, class_weight="balanced", verbose=-1, random_state=s,
                                     min_child_samples=max(2, min(20, len(i) // 50)))
            runs["raw_lightgbm"].append(score(gbm.fit(Xp[i], yp[i]).predict(Xt)))
        results[key] = {m: {"macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in v])), 4),
                            "macro_f1_sd": round(float(np.std([r["macro_f1"] for r in v], ddof=1)), 4) if len(v) > 1 else 0.0,
                            "accuracy_mean": round(float(np.mean([r["accuracy"] for r in v])), 4), "runs": len(v)}
                        for m, v in runs.items()}
        print(f"labels={key}: " + ", ".join(f"{m} {v['macro_f1_mean']:.3f}" for m, v in results[key].items()), flush=True)
    out = {"method": method, "flows": n_rows, "unique_flows": n_unique, "features": int(X.shape[1]), "classes": classes,
           "pretrain_pool": int(len(Xp)), "test": int(len(Xt)), "pretrain_loss": hist, "label_budgets": results,
           "seconds": round(time.time() - t_start)}
    (OUT if method == "masked" else OUT.with_name(f"ssl_nidd_{method}.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
