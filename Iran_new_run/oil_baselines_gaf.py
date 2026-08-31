# -*- coding: utf-8 -*-
"""
oil_baselines_gaf.py

Run the eight baseline detectors on GAF images of the anomaly-score windows.

Run

    python oil_baselines_gaf.py
"""
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fanogan_four_gaf_compare import (
    load_prices, label_window, ensure_dir, GAF_METHODS, IMG_SIZE,
    WINDOW_SIZE, FORECAST_DAYS, LATENT_DIM, BATCH_SIZE, LR_GAN, LR_ENC,
    N_EPOCHS_GAN, N_EPOCHS_ENC, RANDOM_SEED, THRESHOLD_Q, device,
)

EXCEL_DIR = "excel"
DATA_FILE = os.path.join(EXCEL_DIR, "USOIL_daily_final2.xlsx")
STOCK_NAME = "oil"
_H = IMG_SIZE // 4   # matches the 2-stride-of-2 Conv2d stacks below (32 -> 16 -> 8)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Shared GAF data prep (same windowing/labels/split as the main pipeline)
# ============================================================
def build_gaf_windows(prices: np.ndarray, gaf_fn):
    gafs, labels, starts = [], [], []
    n = len(prices)
    for start in range(0, n - WINDOW_SIZE + 1):
        end = start + WINDOW_SIZE
        gafs.append(gaf_fn(prices[start:end], IMG_SIZE))
        labels.append(label_window(start, end))
        starts.append(start)
    return np.array(gafs, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(starts, dtype=np.int64)


def split_train_test(prices, gafs, labels, starts):
    n = len(prices)
    test_start_idx = max(0, n - FORECAST_DAYS - WINDOW_SIZE + 1)
    train_mask = starts < test_start_idx
    test_mask = starts >= test_start_idx
    train_normal_mask = train_mask & (labels == 0)
    return (gafs[train_normal_mask], gafs[test_mask], labels[test_mask], starts[test_mask])


def _loader(x, batch_size=BATCH_SIZE):
    return DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size, shuffle=True, drop_last=True)


# ============================================================
# Shared evaluation / saving -> .\oil\<method>\<mapping>\...
# ============================================================
def evaluate_and_save(method_name, mapping_name, x_test, y_test, starts_test, train_scores, test_scores):
    out_dir = os.path.join(STOCK_NAME, method_name, mapping_name)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir); ensure_dir(plots_dir)

    threshold = float(np.quantile(train_scores, THRESHOLD_Q))
    y_pred = (test_scores > threshold).astype(np.int64)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, test_scores))
    except Exception:
        auc = float("nan")

    pd.DataFrame({
        "window_start": starts_test, "label_0normal_1crash": y_test,
        "anomaly_score": test_scores, "predicted_label": y_pred,
    }).to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)

    result = {
        "stock_name": STOCK_NAME, "method": method_name, "mapping": mapping_name,
        "n_train_normal": int(len(train_scores)), "n_test_total": int(len(x_test)),
        "n_test_normal": int((y_test == 0).sum()), "n_test_crash": int((y_test == 1).sum()),
        "threshold": threshold, "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc,
        "mean_train_score": float(np.mean(train_scores)), "mean_test_score": float(np.mean(test_scores)),
    }
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred normal(0)", "Pred crash(1)"]); ax.set_yticklabels(["True normal(0)", "True crash(1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax); ax.set_title(f"{method_name}/{mapping_name}: Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "confusion_matrix.jpg"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    except Exception:
        ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(f"{method_name}/{mapping_name}: ROC"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "roc_curve.jpg"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_scores[y_test == 0], bins=25, alpha=0.7, label="Normal label=0")
    ax.hist(test_scores[y_test == 1], bins=25, alpha=0.7, label="Crash label=1")
    ax.set_title(f"{method_name}/{mapping_name}: Anomaly Score Distribution"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "score_histogram.jpg"), dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"    [RESULT] {method_name}/{mapping_name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
          f"f1={f1:.4f} auc={auc:.4f}", flush=True)
    return result


# ============================================================
# (1)/(2) Isolation Forest / One-Class SVM on flattened GAF image
# ============================================================
def run_isolation_forest(prices, mapping_name, gaf_fn):
    print(f"\n[RUN] oil | method=isolation_forest | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)
    xf_train = x_train.reshape(len(x_train), -1)
    xf_test = x_test.reshape(len(x_test), -1)

    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
    clf.fit(xf_train)
    train_scores = -clf.score_samples(xf_train)
    test_scores = -clf.score_samples(xf_test)
    return evaluate_and_save("isolation_forest", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


def run_one_class_svm(prices, mapping_name, gaf_fn):
    print(f"\n[RUN] oil | method=one_class_svm | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)
    xf_train = x_train.reshape(len(x_train), -1)
    xf_test = x_test.reshape(len(x_test), -1)

    clf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    clf.fit(xf_train)
    train_scores = -clf.decision_function(xf_train)
    test_scores = -clf.decision_function(xf_test)
    return evaluate_and_save("one_class_svm", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (3) Deep SVDD -- Conv2d encoder, hypersphere distance
# ============================================================
class SVDDNet2D(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(64 * _H * _H, latent_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def run_deep_svdd(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN):
    print(f"\n[RUN] oil | method=deep_svdd | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    net = SVDDNet2D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR_ENC, weight_decay=1e-6)
    loader = _loader(np.expand_dims(x_train, 1))

    net.eval()
    with torch.no_grad():
        all_z = torch.cat([net(xb.to(device)) for (xb,) in loader], dim=0)
        c = all_z.mean(dim=0)
        c[(c.abs() < 1e-2)] = 1e-2

    net.train()
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            z = net(xb)
            loss = ((z - c) ** 2).sum(dim=1).mean()
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [DeepSVDD/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        def scores(x):
            xb = torch.tensor(np.expand_dims(x, 1)).to(device)
            z = net(xb)
            return ((z - c) ** 2).sum(dim=1).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("deep_svdd", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (4) DAGMM -- Conv2d autoencoder + GMM estimation network
# ============================================================
class DAGMMNet2D(nn.Module):
    def __init__(self, latent_dim=4, n_gmm=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(), nn.Linear(32 * _H * _H, latent_dim),
        )
        self.dec_fc = nn.Linear(latent_dim, 32 * _H * _H)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Tanh(),
        )
        self.est = nn.Sequential(nn.Linear(latent_dim + 2, 16), nn.Tanh(), nn.Dropout(0.3),
                                  nn.Linear(16, n_gmm), nn.Softmax(dim=1))
        self.latent_dim = latent_dim

    def forward(self, x):  # x: (B,1,H,W)
        z_c = self.enc(x)
        x_hat = self.dec(self.dec_fc(z_c).view(-1, 32, _H, _H))
        xf, xhf = x.flatten(1), x_hat.flatten(1)
        rec_euc = torch.norm(xf - xhf, p=2, dim=1, keepdim=True) / (torch.norm(xf, p=2, dim=1, keepdim=True) + 1e-8)
        rec_cos = nn.functional.cosine_similarity(xf, xhf, dim=1, eps=1e-8).unsqueeze(1)
        z = torch.cat([z_c, rec_euc, rec_cos], dim=1)
        gamma = self.est(z)
        return z, x_hat, gamma


def _dagmm_gmm_params(z, gamma):
    phi = gamma.sum(dim=0) / gamma.size(0)
    mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / gamma.sum(dim=0).unsqueeze(1)
    z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
    cov = (gamma.unsqueeze(2).unsqueeze(3) * (z_mu.unsqueeze(3) * z_mu.unsqueeze(2))).sum(dim=0) \
          / gamma.sum(dim=0).unsqueeze(1).unsqueeze(2)
    return phi, mu, cov


def _dagmm_energy(z, phi, mu, cov, eps=1e-6):
    K, D, _ = cov.shape
    cov = cov + torch.eye(D, device=cov.device).unsqueeze(0) * eps
    z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
    energies = []
    for k in range(K):
        L = torch.linalg.cholesky(cov[k])
        inv_cov_k = torch.cholesky_inverse(L)
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        zk = z_mu[:, k, :]
        maha = (zk @ inv_cov_k * zk).sum(dim=1)
        log_prob = -0.5 * (maha + logdet + D * math.log(2 * math.pi))
        energies.append(torch.log(phi[k] + eps) + log_prob)
    return -torch.logsumexp(torch.stack(energies, dim=1), dim=1)


def run_dagmm(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN, lambda_energy=0.1, lambda_cov=0.005):
    print(f"\n[RUN] oil | method=dagmm | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    net = DAGMMNet2D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR_ENC)
    loader = _loader(np.expand_dims(x_train, 1))

    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            z, x_hat, gamma = net(xb)
            rec_loss = nn.functional.mse_loss(x_hat, xb)
            phi, mu, cov = _dagmm_gmm_params(z, gamma)
            energy = _dagmm_energy(z, phi, mu, cov).mean()
            cov_penalty = sum((1.0 / torch.diagonal(cov[k])).sum() for k in range(cov.size(0)))
            loss = rec_loss + lambda_energy * energy + lambda_cov * cov_penalty
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [DAGMM/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        z_all, _, gamma_all = net(torch.tensor(np.expand_dims(x_train, 1)).to(device))
        phi, mu, cov = _dagmm_gmm_params(z_all, gamma_all)

        def scores(x):
            z, _, _ = net(torch.tensor(np.expand_dims(x, 1)).to(device))
            return _dagmm_energy(z, phi, mu, cov).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("dagmm", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (5) OmniAnomaly -- GRU-VAE over GAF rows (each row = one sequence token)
# ============================================================
class GRUVAERows(nn.Module):
    def __init__(self, n_rows=IMG_SIZE, row_dim=IMG_SIZE, hidden=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc_gru = nn.GRU(row_dim, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc_dec_in = nn.Linear(latent_dim, hidden)
        self.dec_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc_out = nn.Linear(hidden, row_dim)
        self.n_rows = n_rows

    def forward(self, x):  # x: (B, n_rows, row_dim)
        _, h = self.enc_gru(x)
        h = h[-1]
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        dec_in = self.fc_dec_in(z).unsqueeze(1).repeat(1, self.n_rows, 1)
        out, _ = self.dec_gru(dec_in)
        return self.fc_out(out), mu, logvar


def run_omnianomaly(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN, beta=0.01):
    print(f"\n[RUN] oil | method=omnianomaly | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    net = GRUVAERows().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR_ENC)
    loader = _loader(x_train)  # (B, 32, 32) rows-as-sequence directly

    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = net(xb)
            rec_loss = nn.functional.mse_loss(x_hat, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + beta * kl
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [OmniAnomaly/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        def scores(x):
            xb = torch.tensor(x).to(device)
            x_hat, mu, logvar = net(xb)
            return ((x_hat - xb) ** 2).mean(dim=[1, 2]).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("omnianomaly", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (6) USAD -- Conv2d dual-autoencoder, two-phase adversarial
# ============================================================
class USADEncoder2D(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(), nn.Linear(64 * _H * _H, latent_dim), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class USADDecoder2D(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * _H * _H)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, _H, _H))


def run_usad(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN):
    print(f"\n[RUN] oil | method=usad | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    E, D1, D2 = USADEncoder2D().to(device), USADDecoder2D().to(device), USADDecoder2D().to(device)
    opt1 = torch.optim.Adam(list(E.parameters()) + list(D1.parameters()), lr=LR_ENC)
    opt2 = torch.optim.Adam(list(E.parameters()) + list(D2.parameters()), lr=LR_ENC)
    loader = _loader(np.expand_dims(x_train, 1))

    for ep in range(1, epochs + 1):
        for (xb,) in loader:
            xb = xb.to(device)
            z = E(xb); ae1 = D1(z); ae2 = D2(z)
            loss1 = (1.0 / ep) * nn.functional.mse_loss(ae1, xb) + \
                    (1 - 1.0 / ep) * nn.functional.mse_loss(D1(E(ae2)), xb)
            opt1.zero_grad(); loss1.backward(retain_graph=True); opt1.step()

            z = E(xb); ae1 = D1(z).detach(); ae2 = D2(z)
            loss2 = (1.0 / ep) * nn.functional.mse_loss(ae2, xb) - \
                    (1 - 1.0 / ep) * nn.functional.mse_loss(D2(E(ae1)), xb)
            opt2.zero_grad(); loss2.backward(); opt2.step()
        print(f"    [USAD/{mapping_name}] epoch {ep}/{epochs} loss1={loss1.item():.5f} loss2={loss2.item():.5f}", flush=True)

    E.eval(); D1.eval(); D2.eval()
    alpha = beta = 0.5
    with torch.no_grad():
        def scores(x):
            xb = torch.tensor(np.expand_dims(x, 1)).to(device)
            z = E(xb); ae1 = D1(z)
            rec2_of_1 = D2(E(ae1))
            return (alpha * ((xb - ae1) ** 2).mean(dim=[1, 2, 3]) +
                    beta * ((xb - rec2_of_1) ** 2).mean(dim=[1, 2, 3])).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("usad", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (7) TranAD -- Transformer over GAF rows (each row = one token), dual-decoder
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TranADNet2D(nn.Module):
    def __init__(self, n_rows=IMG_SIZE, row_dim=IMG_SIZE, d_model=64, nhead=4):
        super().__init__()
        self.in_proj = nn.Linear(row_dim * 2, d_model)  # [row, focus-row] like TranAD self-conditioning
        self.pos = PositionalEncoding(d_model, max_len=n_rows + 1)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.dec1 = nn.Linear(d_model, row_dim)
        self.dec2 = nn.Linear(d_model, row_dim)

    def forward(self, x, focus):  # x, focus: (B, n_rows, row_dim)
        inp = torch.cat([x, focus], dim=-1)
        h = self.pos(self.in_proj(inp))
        h = self.encoder(h)
        return self.dec1(h), self.dec2(h)


def run_tranad(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN):
    print(f"\n[RUN] oil | method=tranad | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    net = TranADNet2D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR_ENC)
    loader = _loader(x_train)

    for ep in range(1, epochs + 1):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            zero_focus = torch.zeros_like(xb)
            o1, _ = net(xb, zero_focus)
            focus = (o1 - xb) ** 2
            _, o2 = net(xb, focus.detach())
            loss = (1.0 / ep) * nn.functional.mse_loss(o1, xb) + (1 - 1.0 / ep) * nn.functional.mse_loss(o2, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [TranAD/{mapping_name}] epoch {ep}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        def scores(x):
            xb = torch.tensor(x).to(device)
            zero_focus = torch.zeros_like(xb)
            o1, _ = net(xb, zero_focus)
            focus = (o1 - xb) ** 2
            _, o2 = net(xb, focus)
            return ((o2 - xb) ** 2).mean(dim=[1, 2]).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("tranad", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


# ============================================================
# (8) Anomaly Transformer -- anomaly-attention over GAF rows, association discrepancy
# ============================================================
class AnomalyAttentionRows(nn.Module):
    def __init__(self, n_rows=IMG_SIZE, d_model=64):
        super().__init__()
        self.n_rows = n_rows
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.sigma_fc = nn.Linear(d_model, 1)
        idx = torch.arange(n_rows).float()
        self.register_buffer("dist", (idx.unsqueeze(0) - idx.unsqueeze(1)).abs())

    def forward(self, h):
        B, T, D = h.shape
        q, k, v = self.q(h), self.k(h), self.v(h)
        series = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(D), dim=-1)
        sigma = (torch.sigmoid(self.sigma_fc(h)).squeeze(-1) * 5 + 0.1).unsqueeze(-1)
        prior = torch.exp(-(self.dist.unsqueeze(0) ** 2) / (2 * sigma ** 2))
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        return series @ v, series, prior


class AnomalyTransformerNet2D(nn.Module):
    def __init__(self, n_rows=IMG_SIZE, row_dim=IMG_SIZE, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(row_dim, d_model)
        self.attn = AnomalyAttentionRows(n_rows, d_model)
        self.out_proj = nn.Linear(d_model, row_dim)

    def forward(self, x):  # x: (B, n_rows, row_dim)
        h = self.in_proj(x)
        out, series, prior = self.attn(h)
        return self.out_proj(out), series, prior


def _kl(p, q, eps=1e-8):
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def run_anomaly_transformer(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN, lam=3.0):
    print(f"\n[RUN] oil | method=anomaly_transformer | mapping={mapping_name}", flush=True)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)

    net = AnomalyTransformerNet2D().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR_ENC)
    loader = _loader(x_train)

    def assoc_discrepancy(series, prior):
        return 0.5 * (_kl(series, prior.detach()) + _kl(prior, series.detach())).mean(dim=1)

    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, series, prior = net(xb)
            rec_loss = nn.functional.mse_loss(x_hat, xb)
            disc = assoc_discrepancy(series, prior).mean()
            loss = rec_loss - lam * disc
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [AnomalyTransformer/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    net.eval()
    with torch.no_grad():
        def scores(x):
            xb = torch.tensor(x).to(device)
            x_hat, series, prior = net(xb)
            rec_err = ((x_hat - xb) ** 2).mean(dim=[1, 2])
            disc = assoc_discrepancy(series, prior)
            ass_weight = torch.softmax(-disc, dim=0)
            return (ass_weight * rec_err).cpu().numpy()
        train_scores = scores(x_train); test_scores = scores(x_test)
    return evaluate_and_save("anomaly_transformer", mapping_name, x_test, y_test, starts_test, train_scores, test_scores)


ALL_GAF_METHODS = {
    "isolation_forest": run_isolation_forest,
    "one_class_svm": run_one_class_svm,
    "deep_svdd": run_deep_svdd,
    "dagmm": run_dagmm,
    "omnianomaly": run_omnianomaly,
    "usad": run_usad,
    "tranad": run_tranad,
    "anomaly_transformer": run_anomaly_transformer,
}


# ============================================================
# Same protocol as the other baselines: train on normal train GAFs,
# anomaly score = reconstruction error (+ KL for the VAE), Q95 threshold.
# ============================================================
class _DenseAE(nn.Module):
    def __init__(self, d_in, d_hid=256, d_lat=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def run_autoencoder(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN):
    print(f"\n[RUN] oil | method=autoencoder | mapping={mapping_name}", flush=True)
    torch.manual_seed(RANDOM_SEED)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)
    d = IMG_SIZE * IMG_SIZE
    xf_tr = torch.tensor(x_train.reshape(len(x_train), -1), dtype=torch.float32)
    xf_te = torch.tensor(x_test.reshape(len(x_test), -1), dtype=torch.float32)
    model = _DenseAE(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader(xf_tr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    [AE/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        tr = ((model(xf_tr.to(device)) - xf_tr.to(device)) ** 2).mean(dim=1).cpu().numpy()
        te = ((model(xf_te.to(device)) - xf_te.to(device)) ** 2).mean(dim=1).cpu().numpy()
    return evaluate_and_save("autoencoder", mapping_name, x_test, y_test, starts_test, tr, te)


class _ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8x8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())  # 4x4
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):
        return self.dec(self.enc(x))


def run_cnn_autoencoder(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN):
    print(f"\n[RUN] oil | method=cnn_autoencoder | mapping={mapping_name}", flush=True)
    torch.manual_seed(RANDOM_SEED)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)
    xt_tr = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    xt_te = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    model = _ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader(xt_tr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    [CNN-AE/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        tr = ((model(xt_tr.to(device)) - xt_tr.to(device)) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        te = ((model(xt_te.to(device)) - xt_te.to(device)) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return evaluate_and_save("cnn_autoencoder", mapping_name, x_test, y_test, starts_test, tr, te)


class _ConvVAE(nn.Module):
    def __init__(self, d_lat=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        self.fc_mu = nn.Linear(32 * 8 * 8, d_lat)
        self.fc_lv = nn.Linear(32 * 8 * 8, d_lat)
        self.fc_up = nn.Linear(d_lat, 32 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        out = self.dec(self.fc_up(z).view(-1, 32, 8, 8))
        return out, mu, lv


def run_vae(prices, mapping_name, gaf_fn, epochs=N_EPOCHS_GAN, beta=1.0):
    print(f"\n[RUN] oil | method=vae | mapping={mapping_name}", flush=True)
    torch.manual_seed(RANDOM_SEED)
    gafs, labels, starts = build_gaf_windows(prices, gaf_fn)
    x_train, x_test, y_test, starts_test = split_train_test(prices, gafs, labels, starts)
    xt_tr = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    xt_te = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    model = _ConvVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader(xt_tr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            out, mu, lv = model(xb)
            rec = nn.functional.mse_loss(out, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            loss = rec + beta * kl
            loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    [VAE/{mapping_name}] epoch {ep+1}/{epochs} loss={tot/max(1,len(loader)):.5f}", flush=True)

    def _scores(xt):
        model.eval()
        with torch.no_grad():
            out, mu, lv = model(xt.to(device))
            rec = ((out - xt.to(device)) ** 2).mean(dim=(1, 2, 3))
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp(), dim=1)
            return (rec + beta * kl).cpu().numpy()

    return evaluate_and_save("vae", mapping_name, x_test, y_test, starts_test,
                             _scores(xt_tr), _scores(xt_te))


ALL_GAF_METHODS.update({
    "autoencoder": run_autoencoder,
    "cnn_autoencoder": run_cnn_autoencoder,
    "vae": run_vae,
})
