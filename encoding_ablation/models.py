# -*- coding: utf-8 -*-
"""
models.py
=========

Network definitions and the training loop shared by the ablation programs.

Run

    python models.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

IMG = 32
LATENT = 32
N_CRITIC = 5
LAMBDA_GP = 10.0
KAPPA = 1.0

# published settings, kept for the paper variant
PAPER_LR = 1e-4
PAPER_BATCH = 16
# the compact variant is trained with a slightly larger step because it is
# smaller; both are stated here rather than buried in the runner
COMPACT_LR = 2e-4
COMPACT_BATCH = 64

CLS_LR = 1e-3
CLS_WEIGHT_DECAY = 1e-4
CLS_BATCH = 64
CLS_DROPOUT = 0.3

_H = IMG // 16


# ------------------------------------------------------------------ classifier
def make_classifier(kernel=3, ch1=8, ch2=16, dropout=CLS_DROPOUT):
    """The supervised network of the window sweep, layer for layer."""
    p = kernel // 2
    return nn.Sequential(
        nn.Conv2d(1, ch1, kernel, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(ch1, ch2, kernel, padding=p), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(ch2, 1), nn.Flatten(0))


def train_classifier(model, x_tr, y_tr, x_te, epochs, seed, lr=CLS_LR):
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=CLS_WEIGHT_DECAY)
    pos = float(np.sum(y_tr))
    w = torch.tensor(max(len(y_tr) - pos, 1.0) / max(pos, 1.0),
                     dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_tr, float), dtype=torch.float32)
    n = len(xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, CLS_BATCH):
            idx = perm[i:i + CLS_BATCH]
            opt.zero_grad()
            lossf(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_te, dtype=torch.float32)).numpy()


# --------------------------------------------------------- the published f-AnoGAN
class PaperGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 512 * _H * _H)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 512, _H, _H))


class PaperCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(512 * _H * _H, 1))

    def forward(self, x):
        return self.head(self.feat(x))

    def features(self, x):
        return self.feat(x)


class PaperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(512 * _H * _H, LATENT))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------ the compact variant
class CompactGenerator(nn.Module):
    def __init__(self, ch=(32, 16)):
        super().__init__()
        c1, c2 = ch
        self.c1, self.s = c1, IMG // 4
        self.fc = nn.Linear(LATENT, c1 * self.s * self.s)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(c2, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.c1, self.s, self.s))


class CompactCritic(nn.Module):
    def __init__(self, ch=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = ch
        self.feat = nn.Sequential(
            nn.Conv2d(1, c1, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c1, c2, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(
            nn.Conv2d(c2, c3, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(c3 * (IMG // 8) * (IMG // 8), 1))

    def forward(self, x):
        return self.head(self.feat(x))

    def features(self, x):
        return self.feat(x)


class CompactEncoder(nn.Module):
    def __init__(self, ch=(16, 32)):
        super().__init__()
        c1, c2 = ch
        s = IMG // 4
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c1, c2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(c2 * s * s, LATENT))

    def forward(self, x):
        return self.net(x)


VARIANTS = {
    "paper": (PaperGenerator, PaperCritic, PaperEncoder, PAPER_LR, PAPER_BATCH),
    "compact": (CompactGenerator, CompactCritic, CompactEncoder,
                COMPACT_LR, COMPACT_BATCH),
}


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


# The published counts, so that a change to the paper variant cannot pass
# unnoticed. If these ever fail, the reproduction is no longer the published
# network and the failure should be read rather than silenced.
PUBLISHED_SIZES = {"generator": 2822465, "critic": 2756545,
                   "encoder": 2821856}


def check_paper_sizes():
    got = {"generator": count_parameters(PaperGenerator()),
           "critic": count_parameters(PaperCritic()),
           "encoder": count_parameters(PaperEncoder())}
    bad = {k: (v, PUBLISHED_SIZES[k]) for k, v in got.items()
           if v != PUBLISHED_SIZES[k]}
    if bad:
        raise AssertionError(
            "the reproduced f-AnoGAN no longer matches the published sizes: %s"
            % bad)
    return got


def gradient_penalty(D, real, fake, device):
    b = real.size(0)
    a = torch.rand(b, 1, 1, 1, device=device)
    interp = (a * real + (1.0 - a) * fake).requires_grad_(True)
    out = D(interp)
    g = torch.autograd.grad(out, interp, grad_outputs=torch.ones_like(out),
                            create_graph=True, retain_graph=True)[0]
    return ((g.norm(2, dim=[1, 2, 3]) - 1.0) ** 2).mean()


def fit_fanogan(x_tr, x_te, variant, epochs, seed, device, verbose=False,
                lambda_gp=LAMBDA_GP):
    """Fit the pair and the encoder on normal windows only, then score.

    `lambda_gp` is exposed because the gradient penalty constrains the norm of
    the critic's input gradient to one, which is the quantity that differs most
    between the four angular mappings. A sweep over it tests whether that
    constraint is what removes the difference.
    """
    from torch.utils.data import DataLoader, TensorDataset
    Gc, Dc, Ec, lr, batch = VARIANTS[variant]
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32)),
        batch_size=batch, shuffle=True, drop_last=len(x_tr) >= 2 * batch)
    G, D, E = Gc().to(device), Dc().to(device), Ec().to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))
    G.train()
    D.train()
    d_last = float("nan")
    for ep in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            b = real.size(0)
            for _ in range(N_CRITIC):
                z = torch.randn(b, LATENT, device=device)
                fake = G(z).detach()
                d_loss = (-D(real).mean() + D(fake).mean()
                          + lambda_gp * gradient_penalty(D, real, fake,
                                                         device))
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()
            z = torch.randn(b, LATENT, device=device)
            g_loss = -D(G(z)).mean()
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            d_last = float(d_loss.item())
        if verbose:
            print("      gan epoch %2d  D %9.3f" % (ep + 1, d_last), flush=True)

    opt_E = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    E.train()
    G.eval()
    D.eval()
    for ep in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            recon = G(E(real))
            with torch.no_grad():
                f_real = D.features(real)
            loss = (nn.functional.mse_loss(recon, real)
                    + KAPPA * nn.functional.mse_loss(D.features(recon), f_real))
            opt_E.zero_grad()
            loss.backward()
            opt_E.step()

    E.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            xb = torch.tensor(x_te[i:i + 256], dtype=torch.float32).to(device)
            recon = G(E(xb))
            rec = ((xb - recon) ** 2).mean(dim=[1, 2, 3])
            feat = ((D.features(xb) - D.features(recon)) ** 2).mean(
                dim=[1, 2, 3])
            out.extend((rec + KAPPA * feat).cpu().numpy().tolist())
    return np.array(out, dtype=np.float64), d_last


def fit_fanogan_scores_on(x_tr, variant, epochs, seed, device):
    """The same fit, but scoring the training windows, which is what the
    pseudo-label arm needs before it can set a threshold."""
    return fit_fanogan(x_tr, x_tr, variant, epochs, seed, device)
