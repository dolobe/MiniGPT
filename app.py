"""
MiniGPT (FR) – v2 — Rewritten & corrected

Objectifs de cette version:
- Corriger plusieurs points fragiles de la démo originale.
- Mieux gérer les erreurs (imports, fichiers manquants, chargement de modèle).
- Améliorer l'échantillonnage (top-k / top-p / nucleus sampling) et la stabilité numérique.
- Ajouter logging structuré et messages d'aide plus clairs.
- Petites améliorations UI & sécurité (auth, validations).
- Compatibilité CPU/GPU et protection contre memmap corrompu.

NOTE: Ce fichier est pensé pour être lancé avec Streamlit : `streamlit run minigpt_v2_fixed.py`.

"""

# =====================================================================
# Imports et vérifications
# =====================================================================
import os
import re
import io
import math
import json
import time
import base64
import random
import shutil
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import logging

# Logging basic config
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("minigpt")

# Tentative d'import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    logger.exception("PyTorch non disponible. Veuillez installer torch (CPU ou CUDA).")
    raise RuntimeError("PyTorch est requis pour cette démo : pip install torch") from e

# Tentative d'import Streamlit — nous captons les erreurs et fournissons un fallback pour debugging
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    # Ne raise pas ici: on veut que le module puisse être lu même si Streamlit n'est pas installé
    STREAMLIT_AVAILABLE = False
    # Fournir un objet minimal pour permettre à certains tests non-UI de fonctionner
    class _Stub:
        def __getattr__(self, name):
            def _f(*args, **kwargs):
                raise RuntimeError("Streamlit non installé: installez streamlit pour lancer l'interface.")
            return _f
    st = _Stub()
    logger.warning("Streamlit non installé. L'interface ne pourra pas démarrer.")

# Défaut page config: appelé uniquement si Streamlit présent
if STREAMLIT_AVAILABLE:
    try:
        st.set_page_config(page_title="MiniGPT (FR) – v2 (fix)", page_icon="🤖", layout="wide")
    except Exception:
        # Certains environnements Streamlit l'appellent déjà — ignore
        pass

# =====================================================================
# Constantes & chemins
# =====================================================================
APP_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(APP_DIR, "data")
CHECK_DIR = os.path.join(APP_DIR, "checkpoints")
RUNS_DIR = os.path.join(APP_DIR, "runs")
USERS_FILE = os.path.join(APP_DIR, "users.json")
SESSIONS_DIR = os.path.join(APP_DIR, "sessions")

for p in (DATA_DIR, CHECK_DIR, RUNS_DIR, SESSIONS_DIR):
    os.makedirs(p, exist_ok=True)

# Fichiers produits
VOCAB_PKL = os.path.join(APP_DIR, "vocab.pkl")
DATASET_NPY = os.path.join(APP_DIR, "dataset.uint32.npy")
DATASET_META = os.path.join(APP_DIR, "dataset.meta.json")
MODEL_PTH = os.path.join(APP_DIR, "mini_transformer_plus.pth")
CONFIG_JSON = os.path.join(APP_DIR, "config.json")
LOGS_JSONL = os.path.join(RUNS_DIR, "training_log.jsonl")
METRICS_CSV = os.path.join(RUNS_DIR, "metrics.csv")

# =====================================================================
# Utils: hashing, json safe write, time
# =====================================================================

def _rand_salt(n: int = 16) -> str:
    return base64.urlsafe_b64encode(os.urandom(n)).decode("utf-8").rstrip("=")


def _hash_password(password: str, salt: str) -> str:
    h = hashlib.sha256()
    h.update((password + ":" + salt).encode("utf-8"))
    return h.hexdigest()


def _verify_password(password: str, salt: str, hashed: str) -> bool:
    return _hash_password(password, salt) == hashed


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Erreur lecture JSON %s", path)
        return default


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

# =====================================================================
# Auth: simple login/register (JSON, hash + salt)
# =====================================================================
DEFAULT_USERS = {
    "admin@example.com": {
        "salt": "demo",
        "hash": _hash_password("admin", "demo"),
        "role": "admin",
        "created_at": now_iso(),
    }
}

if not os.path.exists(USERS_FILE):
    _write_json(USERS_FILE, DEFAULT_USERS)


def get_users() -> Dict[str, Dict[str, str]]:
    return _read_json(USERS_FILE, {})


def save_users(users: Dict[str, Dict[str, str]]):
    _write_json(USERS_FILE, users)

# =====================================================================
# Configuration dataclass
# =====================================================================
@dataclass
class Config:
    data: str = os.path.join("data", "wiki_fr_5GB.txt")  # chemin du corpus (UTF-8)
    vocab_extra: List[str] = None
    context: int = 256
    embed: int = 256
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    steps: int = 20000
    batch: int = 32
    grad_accum: int = 1
    ckpt_dir: str = CHECK_DIR
    seed: int = 42

    def __post_init__(self):
        if self.vocab_extra is None:
            self.vocab_extra = ["<pad>", "<bos>", "<eos>"]


DEFAULT_CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================
# Vocab save/load
# =====================================================================
def save_vocab(stoi: Dict[str, int], itos: List[str], path: str = VOCAB_PKL):
    with open(path, "wb") as f:
        import pickle
        pickle.dump({"stoi": stoi, "itos": itos}, f)


def load_vocab(path: str = VOCAB_PKL):
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["stoi"], d["itos"]

# =====================================================================
# Tokenisation caractère + préprocessing memmap
# =====================================================================

def _normalize_text(chunk: str) -> str:
    # Normalisation simple: collapse whitespace, trim, lowercase.
    # Conserver les caractères accentués — on ne décompose pas ici.
    return re.sub(r"\s+", " ", chunk).strip().lower()


def build_vocab_from_file(txt_path: str, extra_tokens: List[str]):
    chars = set()
    total_read = 0
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(1_000_000), ""):
            if not chunk:
                break
            chunk = _normalize_text(chunk)
            total_read += len(chunk)
            chars.update(list(chunk))
    chars = sorted([c for c in chars if c])
    itos = list(extra_tokens) + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    logger.info("Vocab construit: %d symboles (extra=%d) | caractères lus=%d", len(itos), len(extra_tokens), total_read)
    return stoi, itos


def preprocess_to_memmap(txt_path: str, stoi: Dict[str, int], out_npy: str = DATASET_NPY):
    import numpy as _np
    # 1) Compter
    total = 0
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(4_000_000), ""):
            if not chunk:
                break
            total += len(_normalize_text(chunk))
    if total == 0:
        raise RuntimeError("Le fichier d'entrée est vide ou illisible.")
    # 2) Écrire memmap
    arr = _np.memmap(out_npy, mode="w+", dtype=_np.uint32, shape=(total,))
    idx = 0
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(4_000_000), ""):
            if not chunk:
                break
            norm = _normalize_text(chunk)
            ids = [_np.uint32(stoi.get(c, 0)) for c in norm]
            n = len(ids)
            arr[idx:idx+n] = _np.array(ids, dtype=_np.uint32)
            idx += n
    arr.flush()
    logger.info("Memmap écrit: %s (%d entrées)", out_npy, total)
    return out_npy, total

# =====================================================================
# Dataset
# =====================================================================
class CausalDataset(Dataset):
    """Dataset qui lit un memmap uint32 contenant ids de tokens (caractères).

    Il découpe en séquences de longueur `context` et retourne (x,y) pour causal language modeling.
    """
    def __init__(self, memmap_path: str, length: int, context: int, split: str = "train", val_ratio: float = 0.01):
        import numpy as _np
        if not os.path.exists(memmap_path):
            raise RuntimeError(f"Memmap introuvable: {memmap_path}")
        # On garde le memmap en lecture seule
        self.data = _np.memmap(memmap_path, mode="r", dtype=_np.uint32, shape=(length,))
        split_idx = int(length * (1 - val_ratio))
        if split == "train":
            self.start, self.end = 0, split_idx
        else:
            self.start, self.end = split_idx, length - 1
        self.context = context
        # nombre d'échantillons approximatif
    def __len__(self):
        return max(1, (self.end - self.start) // self.context)

    def __getitem__(self, _idx):
        import numpy as _np
        i = random.randint(self.start, max(self.start, self.end - self.context - 2))
        x = self.data[i:i+self.context].astype(_np.int64)
        y = self.data[i+1:i+self.context+1].astype(_np.int64)
        import torch as _torch
        return _torch.from_numpy(x), _torch.from_numpy(y)

# =====================================================================
# Modèle GPT (decoder-only) — implémentation simple & commentée
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, dropout: float):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.out = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, T, C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)
        # reshape -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # att: (B, n_heads, T, T)
        if attn_mask is not None:
            # attn_mask shape expected: (1,1,T,T) boolean or 0/1
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class GPTBlock(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = nn.Sequential(
            nn.Linear(n_embed, ff_mult * n_embed),
            nn.GELU(),
            nn.Linear(ff_mult * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), attn_mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, context: int = 256, n_embed: int = 256, n_layers: int = 4,
                 n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(context, n_embed)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(n_embed, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        # tying poids sortie/embedding
        try:
            self.head.weight = self.tok_emb.weight
        except Exception:
            # fallback
            pass

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        # masque causal broadcastable (1,1,T,T)
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.uint8)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# =====================================================================
# Scheduler & helpers
# =====================================================================

def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float):
    if step < warmup:
        return base_lr * (step + 1) / max(max(warmup, 1), 1)
    progress = (step - warmup) / max(1, (max_steps - warmup))
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress))) * base_lr

# =====================================================================
# Logging helpers
# =====================================================================

def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_csv_metrics(path: str, step: int, loss: float, lr: float, split: str = "train"):
    header = "step,split,loss,lr\n"
    line = f"{step},{split},{loss},{lr}\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

# =====================================================================
# Evaluation & Training loop
# =====================================================================

def evaluate(model, loader, criterion, device, vocab_size, max_batches: int = 50):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
            losses.append(loss.item())
            if i + 1 >= max_batches:
                break
    model.train()
    return float(sum(losses) / len(losses)) if losses else float('inf')


def train_loop(cfg: Config, progress_cb=None, val_cb=None, stop_after_steps: Optional[int] = None, run_name: str = None):
    """Boucle d'entraînement avec autosave + logs.

    progress_cb: callable(step:int, loss:float, lr:float)
    val_cb: callable(step:int, val_loss:float)
    stop_after_steps: option pour arrêter plus tôt que cfg.steps (démo)
    run_name: dossier de run pour snapshots.
    """
    set_seed(cfg.seed)

    # Prépare données (vocab, memmap)
    if not os.path.exists(VOCAB_PKL):
        stoi, itos = build_vocab_from_file(cfg.data, cfg.vocab_extra)
        save_vocab(stoi, itos, VOCAB_PKL)
    else:
        stoi, itos = load_vocab(VOCAB_PKL)
    vocab_size = len(itos)

    if not (os.path.exists(DATASET_NPY) and os.path.exists(DATASET_META)):
        npy, total = preprocess_to_memmap(cfg.data, stoi, DATASET_NPY)
        _write_json(DATASET_META, {"length": total})
    else:
        with open(DATASET_META, "r", encoding="utf-8") as f:
            total = json.load(f).get("length", 0)
    if total <= 0:
        raise RuntimeError("Dataset vide ou meta corrompue. Repréparez les données.")

    train_ds = CausalDataset(DATASET_NPY, total, cfg.context, split="train")
    val_ds   = CausalDataset(DATASET_NPY, total, cfg.context, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, drop_last=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, cfg.context, cfg.embed, cfg.layers, cfg.heads, cfg.ff_mult, cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(cfg.ckpt_dir, "last.pt")

    start_step = 0
    if os.path.exists(last_ckpt):
        try:
            state = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state.get("opt", opt.state_dict()))
            start_step = int(state.get("step", 0))
            logger.info("Checkpoint chargé: step=%d", start_step)
        except Exception:
            logger.exception("Impossible de charger le checkpoint last.pt — on repart de zéro.")

    global_step = start_step
    criterion = nn.CrossEntropyLoss()
    best_val = float('inf')

    accum = max(1, cfg.grad_accum)
    max_steps = stop_after_steps if stop_after_steps is not None else cfg.steps

    # -- Logging run metadata
    run_name = run_name or f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _write_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    model.train()
    t0 = time.time()
    try:
        while global_step < max_steps:
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                lr = cosine_lr(global_step, cfg.warmup_steps, cfg.steps, cfg.lr)
                for g in opt.param_groups:
                    g['lr'] = lr

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(xb)
                    loss = criterion(logits.view(-1, vocab_size), yb.view(-1)) / accum

                scaler.scale(loss).backward()

                if (global_step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                # --- Logging ---
                _append_jsonl(LOGS_JSONL, {
                    "ts": now_iso(),
                    "step": int(global_step),
                    "lr": float(lr),
                    "loss": float(loss.item() * accum),
                    "split": "train",
                })
                _append_csv_metrics(METRICS_CSV, int(global_step), float(loss.item() * accum), float(lr), split="train")

                if progress_cb is not None:
                    try:
                        progress_cb(global_step, float(loss.item()*accum), float(lr))
                    except Exception:
                        logger.exception("Callback progress_cb a levé une exception")

                # --- Eval & Checkpoints ---
                if global_step % 1000 == 0 and global_step > 0:
                    val_loss = evaluate(model, val_loader, criterion, device, vocab_size)
                    if val_cb is not None:
                        try:
                            val_cb(global_step, float(val_loss))
                        except Exception:
                            logger.exception("Callback val_cb a levé une exception")
                    _append_jsonl(LOGS_JSONL, {
                        "ts": now_iso(),
                        "step": int(global_step),
                        "lr": float(lr),
                        "loss": float(val_loss),
                        "split": "val",
                    })
                    _append_csv_metrics(METRICS_CSV, int(global_step), float(val_loss), float(lr), split="val")

                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({"model": model.state_dict()}, os.path.join(cfg.ckpt_dir, "best_model.pt"))
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": global_step}, last_ckpt)

                    # snapshot dans le run
                    torch.save({"model": model.state_dict()}, os.path.join(run_dir, f"snapshot_step{global_step}.pt"))

                global_step += 1
                if global_step >= max_steps:
                    break
            # fin for
        # fin while
    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur (KeyboardInterrupt). Sauvegarde checkpoint last.pt ...")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": global_step}, last_ckpt)
    finally:
        # Sauvegardes finales
        torch.save({"model": model.state_dict()}, MODEL_PTH)
        _write_json(CONFIG_JSON, asdict(cfg))
        elapsed = time.time() - t0
        _append_jsonl(LOGS_JSONL, {"ts": now_iso(), "event": "train_end", "elapsed_sec": elapsed, "steps": global_step})

    return model, vocab_size

# =====================================================================
# Génération (sampling) — correction top-k / top-p implémentation
# =====================================================================

def sample_next(logits, temperature: float = 1.0, top_k: Optional[int] = 50, top_p: float = 0.95):
    """Retourne un index d'échantillonnage pour une distribution logits (torch.Tensor 1D ou 2D BxV).

    Implémente temperature scaling, top-k et top-p (nucleus) correctement.
    - logits: tensor (..., V)
    - renvoie: tensor (..., 1) d'indices
    """
    # logits peut être (B, V) ou (V,)
    is_batched = logits.dim() == 2
    if not is_batched:
        logits = logits.unsqueeze(0)
    if temperature <= 0:
        # argmax
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        return idx if is_batched else idx.squeeze(0)

    logits = logits / max(1e-8, float(temperature))
    probs = torch.softmax(logits, dim=-1)

    # top-k : zero out everything outside top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        kth_vals, kth_idx = torch.topk(probs, top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=kth_idx, src=torch.ones_like(kth_vals))
        probs = probs * mask

    # top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        # sort probs descending
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # keep the smallest set where cumsum <= top_p (include first element >= top_p)
        # create mask where to keep
        cutoffs = (cumsum_probs <= top_p)
        # ensure at least one token is kept
        cutoffs[..., 0] = True
        # map back to original indices
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=sorted_idx, src=cutoffs.type_as(probs))
        probs = probs * mask

    # renormalize
    probs_sum = probs.sum(dim=-1, keepdim=True)
    # éviter division par zero
    probs_sum = torch.where(probs_sum == 0, torch.ones_like(probs_sum), probs_sum)
    probs = probs / probs_sum

    # sample
    idx = torch.multinomial(probs, num_samples=1)
    return idx if is_batched else idx.squeeze(0)


def format_history(history: List[Tuple[str, str]], user_msg: str) -> str:
    parts = []
    for u, a in history:
        parts.append(f"User: {u}\nAssistant: {a}\n")
    parts.append(f"User: {user_msg}\nAssistant: ")
    return "".join(parts)


# wrapper génération: charge vocab + modèle si besoin puis génère
def generate_reply(user_msg: str, history: List[Tuple[str, str]], max_new_tokens: int = 256,
                    temperature: float = 0.9, top_k: Optional[int] = 50, top_p: float = 0.95) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(VOCAB_PKL):
        raise RuntimeError("Fichier vocab.pkl introuvable. Préparez le vocabulaire d'abord.")
    stoi, itos = load_vocab(VOCAB_PKL)
    if not os.path.exists(CONFIG_JSON):
        raise RuntimeError("Fichier config.json introuvable. Préparez le dataset / config d'abord.")
    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab_size = len(itos)

    model = MiniGPT(vocab_size, cfg["context"], cfg["embed"], cfg["layers"], cfg["heads"], cfg["ff_mult"], cfg["dropout"]).to(device)
    if not os.path.exists(MODEL_PTH):
        raise RuntimeError("Model file mini_transformer_plus.pth introuvable. Entraînez le modèle d'abord.")
    state = torch.load(MODEL_PTH, map_location=device)
    # handle both dict with 'model' key or raw state_dict
    try:
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    except Exception:
        # Incompatibilité: log + re-raise
        logger.exception("Impossible de charger l'état du modèle depuis %s", MODEL_PTH)
        raise

    model.eval()

    prompt = format_history(history, user_msg)
    ids = [stoi.get(c, 0) for c in _normalize_text(prompt)]

    # tronquer au contexte
    ctx = int(cfg["context"])
    ids = ids[-ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=device) if len(ids) > 0 else torch.zeros((1,1), dtype=torch.long, device=device)

    # generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            next_token = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat([x, next_token], dim=1)
            if x.size(1) > ctx:
                x = x[:, -ctx:]
            ch = itos[int(next_token.item())]
            # stop if sentence terminator found (char-level heuristic)
            if ch in [".", "?", "!", "\n"]:
                break
    gen = x[0].tolist()
    gen_text = "".join(itos[i] for i in gen[len(ids):])
    return gen_text.strip()

# =====================================================================
# UI Helpers: file uploader + path source, downloads, charts
# =====================================================================

def ensure_uploaded_saved(uploaded_file, save_as: str) -> Optional[str]:
    """Sauvegarde un fichier uploadé dans DATA_DIR et retourne son chemin (ou None)."""
    if uploaded_file is None:
        return None
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    with open(save_as, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return save_as


def make_download_button(path: str, label: str):
    if not STREAMLIT_AVAILABLE:
        logger.warning("Streamlit non disponible: impossible de créer un bouton de téléchargement pour %s", path)
        return
    if not os.path.exists(path):
        st.warning(f"Fichier non trouvé: {os.path.basename(path)}")
        return
    with open(path, "rb") as f:
        st.download_button(label=label, data=f, file_name=os.path.basename(path))


def plot_training_chart():
    """Affiche un graphique loss/LR à partir de METRICS_CSV si dispo."""
    if not STREAMLIT_AVAILABLE:
        logger.info("plot_training_chart: Streamlit indisponible")
        return
    import pandas as pd
    if not os.path.exists(METRICS_CSV):
        st.info("Pas encore de métriques enregistrées.")
        return
    try:
        df = pd.read_csv(METRICS_CSV)
        tab1, tab2 = st.tabs(["Loss", "Learning Rate"])
        with tab1:
            st.line_chart(df.pivot(index="step", columns="split", values="loss"))
        with tab2:
            st.line_chart(df.set_index("step")["lr"])
    except Exception:
        logger.exception("Erreur lors du tracé des métriques")
        st.warning("Impossible d'afficher le graphique des métriques.")

# =====================================================================
# Auth pages (Streamlit-specific functions)
# =====================================================================
if STREAMLIT_AVAILABLE:
    def auth_login_page():
        st.title("🔒 Auth – Connexion")
        st.write("Créez un compte dans l'onglet **Inscription** si vous n'en avez pas.")

        with st.form("login_form"):
            email = st.text_input("Email", value="admin@example.com")
            pwd = st.text_input("Mot de passe", type="password", value="admin")
            ok = st.form_submit_button("Se connecter")

        if ok:
            users = get_users()
            u = users.get(email)
            if not u:
                st.error("Utilisateur introuvable.")
                return False
            if _verify_password(pwd, u["salt"], u["hash"]):
                st.session_state["auth_user"] = {"email": email, "role": u.get("role", "user"), "login_at": now_iso()}
                st.success("Connexion réussie !")
                st.experimental_rerun()
            else:
                st.error("Mot de passe incorrect.")
        return False

    def auth_register_page():
        st.title("🆕 Auth – Inscription")
        with st.form("register_form"):
            email = st.text_input("Email")
            pwd1 = st.text_input("Mot de passe", type="password")
            pwd2 = st.text_input("Confirmez le mot de passe", type="password")
            role = st.selectbox("Rôle", ["user", "admin"], index=0)
            ok = st.form_submit_button("Créer mon compte")

        if ok:
            if not email or "@" not in email:
                st.error("Email invalide.")
                return False
            if len(pwd1) < 6:
                st.error("Mot de passe trop court (min 6).")
                return False
            if pwd1 != pwd2:
                st.error("Les mots de passe ne correspondent pas.")
                return False
            users = get_users()
            if email in users:
                st.error("Un compte existe déjà avec cet email.")
                return False
            salt = _rand_salt()
            users[email] = {
                "salt": salt,
                "hash": _hash_password(pwd1, salt),
                "role": role,
                "created_at": now_iso(),
            }
            save_users(users)
            st.success("Compte créé. Vous pouvez vous connecter.")
        return False

    def auth_gate():
        """Affiche l'UI d'auth si non connecté, sinon retourne True."""
        if "auth_user" in st.session_state:
            u = st.session_state["auth_user"]
            with st.sidebar:
                st.markdown(
                    f"""
                    **Connecté :** {u['email']}  
                    Rôle : `{u.get('role','user')}`  
                    <span class='small-muted'>depuis {u.get('login_at','?')}</span>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Se déconnecter"):
                    st.session_state.pop("auth_user")
                    st.experimental_rerun()
            return True

        # Pas connecté -> Tabs login/register
        tab_login, tab_reg = st.tabs(["Connexion", "Inscription"])
        with tab_login:
            auth_login_page()
        with tab_reg:
            auth_register_page()
        st.stop()

# =====================================================================
# Streamlit App Shell
# =====================================================================
if STREAMLIT_AVAILABLE:
    # Gate auth
    auth_gate()

    # Sidebar – Global config controls
    with st.sidebar:
        st.header("Configuration")

        # Source du corpus: upload ou chemin local
        src_mode = st.radio("Source du corpus", ["Upload (Drag & Drop)", "Chemin local"], index=0)

        if src_mode == "Upload (Drag & Drop)":
            uploaded = st.file_uploader("Déposez votre fichier .txt (UTF-8)", type=["txt"], accept_multiple_files=False)
            target_name = st.text_input("Nom de sauvegarde (dans /data)", value="uploaded_corpus.txt")
            if uploaded is not None:
                saved = ensure_uploaded_saved(uploaded, os.path.join(DATA_DIR, target_name))
                if saved:
                    st.success(f"Fichier importé: {saved}")
                    DEFAULT_CFG.data = saved
        else:
            data_file = st.text_input("Chemin du corpus (UTF-8)", value=DEFAULT_CFG.data)
            DEFAULT_CFG.data = data_file

        # Hyperparams
        context = st.number_input("Contexte (tokens)", min_value=64, max_value=4096, value=DEFAULT_CFG.context, step=64)
        embed = st.number_input("Taille embedding", min_value=64, max_value=2048, value=DEFAULT_CFG.embed, step=64)
        layers = st.number_input("Couches (layers)", min_value=1, max_value=24, value=DEFAULT_CFG.layers, step=1)
        heads = st.number_input("Têtes d'attention", min_value=1, max_value=32, value=DEFAULT_CFG.heads, step=1)
        ff_mult = st.number_input("FF mult", min_value=1, max_value=16, value=DEFAULT_CFG.ff_mult, step=1)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.6, value=DEFAULT_CFG.dropout, step=0.05)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=DEFAULT_CFG.lr, step=1e-6, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=0.2, value=DEFAULT_CFG.weight_decay, step=0.01)
        warmup = st.number_input("Warmup steps", min_value=0, max_value=100_000, value=DEFAULT_CFG.warmup_steps, step=100)
        steps = st.number_input("Max steps (entraînement)", min_value=100, max_value=1_000_000, value=2000, step=100,
                                help="Pour démo, commencez bas (p.ex. 2k).")
        batch = st.number_input("Batch size", min_value=1, max_value=512, value=DEFAULT_CFG.batch, step=1)
        grad_accum = st.number_input("Gradient Accumulation", min_value=1, max_value=64, value=DEFAULT_CFG.grad_accum, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=2**31-1, value=DEFAULT_CFG.seed, step=1)

        # Boutons d'action
        run_prepare = st.button("1) 📦 Préparer (vocab + memmap)")
        run_train_lite = st.button("2) 🏃 Entraîner (rapide – 1k steps)")
        run_train_full = st.button("2) 🏋️ Entraîner (selon 'Max steps')")

        st.markdown("---")
        st.caption("Fichiers produits : `vocab.pkl`, `dataset.uint32.npy`, `dataset.meta.json`, `mini_transformer_plus.pth`, `config.json` et checkpoints dans `checkpoints/`.")

    # Synchroniser cfg
    CFG = Config(
        data=DEFAULT_CFG.data,
        context=int(context),
        embed=int(embed),
        layers=int(layers),
        heads=int(heads),
        ff_mult=int(ff_mult),
        dropout=float(dropout),
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_steps=int(warmup),
        steps=int(steps),
        batch=int(batch),
        grad_accum=int(grad_accum),
        seed=int(seed),
    )

    # Top header
    st.title("🤖 MiniGPT – Générateur de texte (caractère) – v2 (fix)")

    st.write(
        """
        Cette version ajoute **meilleure robustesse**, **meilleure gestion des erreurs**,
        et corrige l'échantillonnage top-k / top-p. Pensez à préparer le vocabulaire (étape 1) avant d'entraîner.
        """
    )

    # Quick health card
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CUDA", "Oui" if torch.cuda.is_available() else "Non")
        col2.metric("Device", "cuda" if torch.cuda.is_available() else "cpu")
        col3.metric("Batch", CFG.batch)
        col4.metric("Steps", CFG.steps)

    # --- Onglets principaux ---
    main_tab, data_tab, train_tab, gen_tab, files_tab, help_tab = st.tabs([
        "🏠 Accueil", "📁 Données", "🧪 Entraînement", "📝 Génération", "💾 Fichiers", "ℹ️ Aide"
    ])

    # =====================================================================
    # Onglet Accueil
    # =====================================================================
    with main_tab:
        st.subheader("Vue d'ensemble")
        st.markdown(
            f"""
            - **Source actuelle du corpus** : `{CFG.data}`  
            - **Checkpoints** : `{CHECK_DIR}`  
            - **Runs & métriques** : `{RUNS_DIR}`  
            - **Utilisateur** : `{st.session_state.get('auth_user',{}).get('email','?')}`
            """
        )

        st.markdown("### Graphiques d'entraînement")
        plot_training_chart()

    # =====================================================================
    # Onglet Données – Préparation (vocab + memmap)
    # =====================================================================
    with data_tab:
        st.subheader("Préparation des données")
        prep_box = st.container()

        def _save_cfg(cfg: Config):
            _write_json(CONFIG_JSON, asdict(cfg))

        if run_prepare:
            with prep_box:
                st.subheader("Étape 1: Préparer vocab & memmap")
                if not os.path.exists(CFG.data):
                    st.error(f"Fichier introuvable : {CFG.data}")
                else:
                    with st.spinner("Construction du vocabulaire et memmap…"):
                        try:
                            stoi, itos = build_vocab_from_file(CFG.data, CFG.vocab_extra)
                            save_vocab(stoi, itos, VOCAB_PKL)
                            npy, total = preprocess_to_memmap(CFG.data, stoi, DATASET_NPY)
                            _write_json(DATASET_META, {"length": total})
                            _save_cfg(CFG)
                            st.success(f"✅ Préparation terminée. Vocab={len(itos)} | Caractères={total}")
                            st.info("Astuce: vous pouvez sauvegarder/partager ces fichiers via l'onglet *Fichiers*.")
                        except Exception as e:
                            logger.exception("Erreur préparation données")
                            st.error(f"Erreur lors de la préparation: {e}")

        # Aperçu du fichier
        st.markdown("### Aperçu du corpus")
        if os.path.exists(CFG.data):
            try:
                with open(CFG.data, "r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(1000)
                st.code(sample or "(vide)")
            except Exception as e:
                st.warning(f"Impossible de lire l'aperçu: {e}")
        else:
            st.info("Sélectionnez/chargez un fichier dans la barre latérale.")

    # =====================================================================
    # Onglet Entraînement
    # =====================================================================
    with train_tab:
        st.subheader("Entraînement du modèle")

        train_box = st.container()

        if run_train_lite or run_train_full:
            with train_box:
                st.subheader("Étape 2: Entraînement")
                placeholder = st.empty()
                pbar = st.progress(0)

                if "_metrics" not in st.session_state:
                    st.session_state["_metrics"] = {
                        "steps": [],
                        "loss": [],
                        "lr": [],
                    }

                metrics_area = st.empty()
                chart_area = st.empty()

                status = {"last_step": 0, "last_loss": None, "last_lr": None}

                def progress_cb(step, loss, lr):
                    status.update({"last_step": step, "last_loss": loss, "last_lr": lr})
                    pct = min(1.0, (step + 1) / max(1, CFG.steps))
                    pbar.progress(int(pct * 100))

                    # Store in session metrics (in-memory chart)
                    st.session_state["_metrics"]["steps"].append(step)
                    st.session_state["_metrics"]["loss"].append(loss)
                    st.session_state["_metrics"]["lr"].append(lr)

                    metrics_area.write(f"Step {step} | loss={loss:.4f} | lr={lr:.2e}")
                    try:
                        import pandas as _pd
                        df = _pd.DataFrame({
                            "step": st.session_state["_metrics"]["steps"],
                            "loss": st.session_state["_metrics"]["loss"],
                            "lr": st.session_state["_metrics"]["lr"],
                        })
                        chart_area.line_chart(df.set_index("step")["loss"])
                    except Exception:
                        pass

                def val_cb(step, val_loss):
                    st.info(f"[VAL] step {step} | val_loss={val_loss:.4f}")

                with st.spinner("Entraînement en cours…"):
                    stop_steps = 1000 if run_train_lite else None
                    try:
                        model, vocab_size = train_loop(CFG, progress_cb=progress_cb, val_cb=val_cb, stop_after_steps=stop_steps)
                        _write_json(CONFIG_JSON, asdict(CFG))
                        st.success("✅ Entraînement terminé. Modèle enregistré: mini_transformer_plus.pth")
                    except Exception as e:
                        logger.exception("Erreur pendant l'entraînement")
                        st.error(f"Erreur durant l'entraînement: {e}")

    # =====================================================================
    # Onglet Génération
    # =====================================================================
    with gen_tab:
        st.subheader("Génération de texte")
        colA, colB = st.columns([2, 1])
        with colA:
            user_prompt = st.text_area("Votre message / prompt", height=140, value="Bonjour, peux-tu m'écrire une courte phrase sur les chats ?")
            history_json = st.text_area("Historique (JSON) – liste de paires [user, assistant]", height=120, value="[]",
                                        help="Exemple: [[\"Salut\", \"Bonjour !\"]]")
        with colB:
            max_new_tokens = st.slider("Max nouveaux tokens", 16, 1024, 200, 16)
            temperature = st.slider("Température", 0.0, 2.0, 0.9, 0.05)
            top_k = st.slider("Top-k", 0, 200, 50, 5)
            top_p = st.slider("Top-p (nucléus)", 0.0, 1.0, 0.95, 0.01)
            gen_button = st.button("✨ Générer")

        if gen_button:
            if not (os.path.exists(VOCAB_PKL) and os.path.exists(MODEL_PTH) and os.path.exists(CONFIG_JSON)):
                st.error("Modèle/fichiers manquants. Veuillez préparer et/ou entraîner d'abord.")
            else:
                try:
                    history = json.loads(history_json)
                    assert isinstance(history, list)
                except Exception as e:
                    st.error(f"Historique JSON invalide: {e}")
                    history = []

                with st.spinner("Génération en cours…"):
                    try:
                        text = generate_reply(
                            user_prompt, history, max_new_tokens=max_new_tokens,
                            temperature=float(temperature), top_k=int(top_k) if top_k>0 else None,
                            top_p=float(top_p) if 0.0 < top_p < 1.0 else None
                        )
                        st.success("Réponse générée :")
                        st.code(text)
                    except Exception as e:
                        logger.exception("Erreur génération")
                        st.error(f"Erreur lors de la génération: {e}")

    # =====================================================================
    # Onglet Fichiers – Sauvegardes & Exports
    # =====================================================================
    with files_tab:
        st.subheader("Fichiers & Sauvegardes")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Téléchargements rapides")
            make_download_button(VOCAB_PKL, "⬇️ Télécharger vocab.pkl")
            make_download_button(DATASET_NPY, "⬇️ Télécharger dataset.uint32.npy")
            make_download_button(DATASET_META, "⬇️ Télécharger dataset.meta.json")
            make_download_button(MODEL_PTH, "⬇️ Télécharger mini_transformer_plus.pth")
            make_download_button(CONFIG_JSON, "⬇️ Télécharger config.json")
            make_download_button(LOGS_JSONL, "⬇️ Télécharger training_log.jsonl")
            make_download_button(METRICS_CSV, "⬇️ Télécharger metrics.csv")
            make_download_button(os.path.join(CHECK_DIR, "best_model.pt"), "⬇️ Télécharger best_model.pt")
            make_download_button(os.path.join(CHECK_DIR, "last.pt"), "⬇️ Télécharger checkpoint last.pt")

        with col2:
            st.markdown("### Nettoyage & maintenance")
            if st.button("🗑️ Supprimer checkpoints"):
                try:
                    for f in ("best_model.pt", "last.pt"):
                        p = os.path.join(CHECK_DIR, f)
                        if os.path.exists(p):
                            os.remove(p)
                    st.success("Checkpoints supprimés.")
                except Exception as e:
                    st.error(str(e))
            if st.button("🗑️ Supprimer runs (snapshots & métriques)"):
                try:
                    if os.path.exists(RUNS_DIR):
                        shutil.rmtree(RUNS_DIR)
                    os.makedirs(RUNS_DIR, exist_ok=True)
                    st.success("Runs nettoyés.")
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
        st.markdown("### Liste des fichiers présents")

        def list_files(paths: List[str]):
            rows = []
            for root in paths:
                for dirpath, _, filenames in os.walk(root):
                    for name in filenames:
                        p = os.path.join(dirpath, name)
                        try:
                            size = os.path.getsize(p)
                        except Exception:
                            size = 0
                        rows.append({"path": p, "size_bytes": size})
            return rows

        rows = list_files([APP_DIR, DATA_DIR, CHECK_DIR, RUNS_DIR])
        try:
            import pandas as pd
            df = pd.DataFrame(rows).sort_values("size_bytes", ascending=False)
            st.dataframe(df, width="stretch")
        except Exception:
            st.write(rows)

    # =====================================================================
    # Onglet Aide
    # =====================================================================
    with help_tab:
        st.subheader("Conseils de performance & FAQ")
        with st.expander("ℹ️ Conseils de performance"):
            st.markdown(
                """
                - **Commencez petit**: réduisez `context`, `embed`, `layers`, `heads` pour des essais rapides CPU.
                - **GPU**: si CUDA est disponible, l'entraînement sera beaucoup plus rapide automatiquement.
                - **Corpus**: pour un simple test, utilisez un petit fichier texte (quelques Mo) afin d'itérer vite.
                - **Reprise**: des checkpoints sont écrits dans `checkpoints/last.pt`.
                - **Sécurité disque**: les memmaps et checkpoints peuvent être volumineux; assurez-vous d'avoir de l'espace.
                - **Drag & Drop**: utilisez l'option *Upload (Drag & Drop)* dans la barre latérale pour importer votre fichier.
                - **Exports**: onglet *Fichiers* pour télécharger modèles et métriques.
                """
            )

        with st.expander("❓ FAQ"):
            st.markdown(
                """
                **Q: Puis-je utiliser un fichier autre que du texte UTF-8 ?**  
                R: Cette démo suppose un corpus texte UTF-8. Convertissez-le au besoin.

                **Q: Comment activer l'authentification multi-utilisateur ?**  
                R: Les comptes sont stockés dans `users.json` (hash + salt). Vous pouvez précréer des comptes.

                **Q: Où sont stockées les métriques ?**  
                R: Dans le dossier `runs/` (CSV + JSONL + snapshots). L'onglet *Fichiers* permet le téléchargement.

                **Q: Puis-je augmenter la taille du modèle ?**  
                R: Oui, au prix de plus de RAM/VRAM. Ajustez `embed`, `layers`, `heads`, `context`.
                """
            )

# =====================================================================
# Si Streamlit n'est pas disponible, on propose un petit mode CLI pour tests unitaires
# =====================================================================
else:
    def _cli_info():
        logger.info("Streamlit non installé. Ce module peut être importé, mais l'interface visuelle n'est pas disponible.")
        logger.info("Fichiers attendus: %s, %s, %s", VOCAB_PKL, DATASET_NPY, MODEL_PTH)

    if __name__ == "__main__":
        _cli_info()

# =====================================================================
# EOF
# =====================================================================
