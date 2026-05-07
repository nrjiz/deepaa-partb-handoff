#!/usr/bin/env python3
# Deep Archetypal Analysis (DAA) for HAM10000 (TF2/Keras)
# - Encoder learns convex weights A(x) over archetypes (softmax)
# - Latent mean mu(x) = A(x) @ Z_fixed  (simplex vertices)
# - Posterior t ~ N(mu, sigma^2 I) via reparameterization
# - Decoder reconstructs image + predicts label (side info)
# - Archetype loss enforces Z_pred ≈ Z_fixed (batch-wise B weights)
#
# Works with HAM10000_images_part_1 / _part_2 or "hamdatasetpart1/2" in Downloads.

import os
import math
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# -----------------------------
# GPU: safer memory behavior
# -----------------------------
def set_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# -----------------------------
# Simplex (Z_fixed) like your old create_z_fix()
# -----------------------------
def create_z_fixed(latent_dim: int) -> np.ndarray:
    """
    Create coordinates of a regular simplex with k = latent_dim + 1 vertices in R^latent_dim.
    Matches the idea in the paper: place archetypes at simplex vertices (unit sphere, centered). :contentReference[oaicite:2]{index=2}
    Returns shape: [k, latent_dim]
    """
    k = latent_dim + 1
    z_fixed_t = np.zeros((latent_dim, k), dtype=np.float32)

    for r in range(latent_dim):
        s = 0.0
        for i in range(r):
            s += z_fixed_t[i, r] ** 2
        z_fixed_t[r, r] = np.sqrt(max(1.0 - s, 1e-8))

        for c in range(r + 1, k):
            s = 0.0
            for i in range(r):
                s += z_fixed_t[i, r] * z_fixed_t[i, c]
            z_fixed_t[r, c] = (-1.0 / float(latent_dim) - s) / max(z_fixed_t[r, r], 1e-8)

    return z_fixed_t.T  # [k, latent_dim]


def barycentric_weights_grid(n_per_axis: int = 15) -> np.ndarray:
    """
    For latent_dim=2 => k=3 archetypes, create barycentric weights over triangle.
    Returns shape [N, 3], sums to 1.
    """
    if n_per_axis < 2:
        n_per_axis = 2
    weights = []
    for i in range(n_per_axis):
        a = i / (n_per_axis - 1)
        for j in range(n_per_axis - i):
            b = j / (n_per_axis - 1)
            c = 1.0 - a - b
            weights.append([a, b, c])
    return np.array(weights, dtype=np.float32)


def triangle_grid_plot(images: np.ndarray, n_per_axis: int, out_path: Path, title: str = ""):
    """
    Plot triangle-like grid by placing rows with decreasing counts.
    images: [N, H, W, C] in [0,1]
    """
    N, H, W, C = images.shape
    # compute expected count
    expected = int(n_per_axis * (n_per_axis + 1) / 2)
    if N != expected:
        # fallback: square grid
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        for idx in range(N):
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.axis("off")
            if C == 1:
                ax.imshow(images[idx, :, :, 0], cmap="gray")
            else:
                ax.imshow(images[idx])
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    # triangle layout: row r has (n_per_axis - r) images
    fig = plt.figure(figsize=(10, 10))
    idx = 0
    for r in range(n_per_axis):
        count = n_per_axis - r
        for c in range(count):
            ax = fig.add_subplot(n_per_axis, n_per_axis, r * n_per_axis + c + 1)
            ax.axis("off")
            img = images[idx]
            if C == 1:
                ax.imshow(img[:, :, 0], cmap="gray")
            else:
                ax.imshow(img)
            idx += 1
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# HAM10000 data discovery + loading
# -----------------------------
def find_ham_dirs(downloads_dir: Path):
    keys = ["hamdatasetpart", "hamdatestpart", "ham10000_images_part", "ham10000"]
    candidates = []
    if downloads_dir.exists():
        for p in downloads_dir.iterdir():
            if p.is_dir():
                name = p.name.lower().replace(" ", "")
                if any(k in name for k in keys):
                    candidates.append(p)

    # Prefer exactly two parts if found
    # We'll collect dirs that contain many jpg files
    img_dirs = []
    for d in candidates:
        jpgs = list(d.glob("*.jpg"))
        if len(jpgs) > 100:  # heuristic
            img_dirs.append(d)

    # If nothing found, also check common exact names
    for nm in ["HAM10000_images_part_1", "HAM10000_images_part_2",
               "hamdatasetpart1", "hamdatasetpart2", "hamdatestpart1", "hamdatestpart2"]:
        p = downloads_dir / nm
        if p.exists() and p.is_dir():
            img_dirs.append(p)

    # Deduplicate
    img_dirs = list(dict.fromkeys(img_dirs))
    return img_dirs


def find_labels_csv(downloads_dir: Path, near_dirs: list[Path]) -> Path:
    """
    Pick a CSV that likely contains HAM labels.
    Must include 'image_id' and either 'dx' (diagnosis) or 'label' style columns.
    """
    candidates = []

    # first: Downloads root
    candidates += list(downloads_dir.glob("*.csv"))

    # then: in nearby dirs
    for d in near_dirs:
        candidates += list(d.glob("*.csv"))

    # also: one-level subfolders inside near dirs (common for Kaggle extracts)
    for d in near_dirs:
        for sub in d.iterdir():
            if sub.is_dir():
                candidates += list(sub.glob("*.csv"))

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))

    def score_csv(p: Path) -> int:
        name = p.name.lower()
        score = 0
        if "ham" in name:
            score += 2
        if "meta" in name or "label" in name:
            score += 2
        if "metadata" in name:
            score += 3
        return score

    # Sort by filename score first
    candidates.sort(key=score_csv, reverse=True)

    for csv_path in candidates:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        if "image_id" in cols and ("dx" in cols or "label" in cols or "diagnosis" in cols):
            return csv_path

    raise FileNotFoundError(
        "Could not find a labels CSV in Downloads that contains 'image_id' and a label column (like 'dx'). "
        "Put HAM10000_metadata.csv in Downloads, or pass --labels_csv explicitly."
    )


def build_image_index(img_dirs: list[Path]) -> dict[str, str]:
    """
    Map image_id -> full path. HAM10000 uses file name = image_id.jpg
    """
    index = {}
    for d in img_dirs:
        for p in d.glob("*.jpg"):
            index[p.stem] = str(p)
    return index


def load_ham_dataframe(labels_csv: Path, img_dirs: list[Path]) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # Normalize columns
    df_cols = {c.lower(): c for c in df.columns}
    if "image_id" not in df_cols:
        raise ValueError("labels CSV must have column 'image_id'.")

    label_col = None
    for cand in ["dx", "label", "diagnosis"]:
        if cand in df_cols:
            label_col = df_cols[cand]
            break
    if label_col is None:
        raise ValueError("labels CSV must have a label column like 'dx'.")

    img_index = build_image_index(img_dirs)

    img_ids = df[df_cols["image_id"]].astype(str).values
    paths = []
    keep = []
    for img_id in img_ids:
        p = img_index.get(img_id)
        if p is not None:
            paths.append(p)
            keep.append(True)
        else:
            keep.append(False)

    df = df.loc[np.array(keep)].copy()
    df["filepath"] = paths
    df["label_str"] = df[label_col].astype(str).values

    # label mapping stable
    classes = sorted(df["label_str"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["label_str"].map(class_to_idx).astype(int)

    return df, classes


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for lab in sorted(df["label"].unique()):
        idx = df.index[df["label"] == lab].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_part = idx[:n_test]
        val_part = idx[n_test:n_test + n_val]
        train_part = idx[n_test + n_val:]
        test_idx.extend(test_part.tolist())
        val_idx.extend(val_part.tolist())
        train_idx.extend(train_part.tolist())

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def make_tf_dataset(df: pd.DataFrame, img_size: int, batch_size: int, shuffle: bool):
    paths = df["filepath"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 5000), reshuffle_each_iteration=True)

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Deep AA model (Keras)
# -----------------------------
@dataclass
class DAAConfig:
    img_size: int = 128
    latent_dim: int = 2  # d
    num_classes: int = 7
    recon_w: float = 1.0
    class_w: float = 1.0
    kl_w: float = 1e-2
    at_w: float = 1.0


class DeepAA(tf.keras.Model):
    def __init__(self, cfg: DAAConfig):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.latent_dim + 1

        z_fixed_np = create_z_fixed(cfg.latent_dim)  # [k, d]
        self.z_fixed = tf.constant(z_fixed_np, dtype=tf.float32)

        # Encoder (conv -> dense)
        self.enc = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(cfg.img_size, cfg.img_size, 3)),
            tf.keras.layers.Conv2D(64, 4, strides=2, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(64, 4, strides=2, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
        ])

        # A-weights: convex weights over archetypes (softmax) => [B, k]
        self.a_logits = tf.keras.layers.Dense(self.k)

        # sigma / logvar for posterior
        self.logvar = tf.keras.layers.Dense(cfg.latent_dim)

        # B-weights (batch-wise) to predict archetypes Z_pred = B * mu
        # We'll compute B via logits per sample then softmax across batch dimension.
        self.b_logits = tf.keras.layers.Dense(self.k)

        # Decoder: latent -> image
        self.dec_img = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(cfg.latent_dim,)),
            tf.keras.layers.Dense(8 * 8 * 128, activation="relu"),
            tf.keras.layers.Reshape((8, 8, 128)),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
        ])

        # Side-info head (classification): latent -> logits
        self.dec_cls = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(cfg.latent_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(cfg.num_classes)  # logits
        ])

    def encode(self, x, training: bool):
        h = self.enc(x, training=training)

        # A weights
        a = tf.nn.softmax(self.a_logits(h), axis=-1)  # [B, k]

        # mu = A @ Z_fixed
        mu = tf.matmul(a, self.z_fixed)  # [B, d]

        # posterior variance
        lv = self.logvar(h)  # [B, d]
        lv = tf.clip_by_value(lv, -10.0, 10.0)
        sigma = tf.exp(0.5 * lv)

        # B weights across batch dimension
        # logits per sample -> transpose -> softmax over batch axis => [k, B]
        b_t = self.b_logits(h)            # [B, k]
        b = tf.nn.softmax(tf.transpose(b_t), axis=1)  # [k, B]

        # predicted archetypes from batch convex hull: Z_pred = B @ mu   => [k, d]
        z_pred = tf.matmul(b, mu)

        return a, b, mu, sigma, lv, z_pred

    def reparam(self, mu, sigma, training: bool):
        if not training:
            return mu
        eps = tf.random.normal(tf.shape(mu))
        return mu + sigma * eps

    def decode(self, t, training: bool):
        x_rec = self.dec_img(t, training=training)               # [B, H, W, 3]
        y_logits = self.dec_cls(t, training=training)            # [B, C]
        return x_rec, y_logits

    def call(self, x, training: bool = False):
        a, b, mu, sigma, lv, z_pred = self.encode(x, training=training)
        t = self.reparam(mu, sigma, training=training)
        x_rec, y_logits = self.decode(t, training=training)
        return {
            "a": a, "b": b, "mu": mu, "sigma": sigma, "logvar": lv,
            "z_pred": z_pred, "t": t,
            "x_rec": x_rec, "y_logits": y_logits
        }


# -----------------------------
# Losses (TF2)
# -----------------------------
def kl_standard_normal(mu, logvar):
    """
    KL( N(mu, diag(sigma^2)) || N(0, I) ) in closed form.
    """
    return 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mu) - 1.0 - logvar, axis=1)


def train_step(model: DeepAA, optimizer, x, y, cfg: DAAConfig):
    with tf.GradientTape() as tape:
        out = model(x, training=True)

        # Reconstruction: MSE per sample
        rec = tf.reduce_mean(tf.reduce_sum(tf.square(x - out["x_rec"]), axis=[1, 2, 3]))

        # Side-info classification: sparse CE
        cls = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out["y_logits"])
        )

        # KL
        kl = tf.reduce_mean(kl_standard_normal(out["mu"], out["logvar"]))

        # Archetype loss: ||Z_fixed - Z_pred||^2 (Eq.14 idea) :contentReference[oaicite:3]{index=3}
        at = tf.reduce_mean(tf.reduce_sum(tf.square(model.z_fixed - out["z_pred"]), axis=1))

        total = cfg.recon_w * rec + cfg.class_w * cls + cfg.kl_w * kl + cfg.at_w * at

    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total, rec, cls, kl, at


# Removed @tf.function because DAAConfig is a Python dataclass and TensorFlow 2.20+ may fail to trace it.
def eval_step(model: DeepAA, x, y, cfg: DAAConfig):
    out = model(x, training=False)
    rec = tf.reduce_mean(tf.reduce_sum(tf.square(x - out["x_rec"]), axis=[1, 2, 3]))
    cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out["y_logits"]))
    kl = tf.reduce_mean(kl_standard_normal(out["mu"], out["logvar"]))
    at = tf.reduce_mean(tf.reduce_sum(tf.square(model.z_fixed - out["z_pred"]), axis=1))
    total = cfg.recon_w * rec + cfg.class_w * cls + cfg.kl_w * kl + cfg.at_w * at
    return total, rec, cls, kl, at


# -----------------------------
# Plot helpers
# -----------------------------
def save_image_grid(images, out_path: Path, title: str = "", cols: int = 4):
    images = np.clip(images, 0, 1)
    N = images.shape[0]
    rows = int(math.ceil(N / cols))
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(N):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.axis("off")
        ax.imshow(images[i])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_latents(model: DeepAA, ds):
    mus = []
    labs = []
    for x, y in ds:
        out = model(x, training=False)
        mus.append(out["mu"].numpy())
        labs.append(y.numpy())
    return np.vstack(mus), np.concatenate(labs)


def load_one_image_for_display(path: str, img_size: int):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def plot_latent_scatter(mus, labs, classes, out_path: Path):
    if mus.shape[1] != 2:
        return
    fig = plt.figure(figsize=(7, 6))
    for c in range(len(classes)):
        idx = labs == c
        plt.scatter(mus[idx, 0], mus[idx, 1], s=5, alpha=0.5, label=classes[c])
    plt.legend(fontsize=7, loc="best")
    plt.title("Latent means μ(x)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)




# -----------------------------
# HAM10000 interpretability + table generation
# -----------------------------
def entropy_rows(a: np.ndarray) -> np.ndarray:
    """Normalized entropy of archetype weights A(x). 0=single archetype, 1=uniform."""
    eps = 1e-8
    k = a.shape[1]
    return -(a * np.log(a + eps)).sum(axis=1) / np.log(k)


def gini_like_sparsity(a: np.ndarray) -> float:
    """Simple, readable sparsity score: 1 - normalized entropy averaged over samples."""
    return float(1.0 - entropy_rows(a).mean())


def active_archetype_count(a: np.ndarray, threshold: float = 0.10) -> float:
    """Average number of archetypes with at least threshold contribution per image."""
    return float((a >= threshold).sum(axis=1).mean())


def collect_full_outputs(model: DeepAA, ds, df: pd.DataFrame, cfg: DAAConfig, classes: list[str]):
    """
    Runs the trained model on a dataset and returns per-image outputs needed to prove
    how each HAM10000 lesion type is interpreted by the archetypes.
    """
    rows = []
    cursor = 0
    for x, y in ds:
        out = model(x, training=False)
        x_np = x.numpy()
        rec_np = out["x_rec"].numpy()
        a_np = out["a"].numpy()
        mu_np = out["mu"].numpy()
        logits_np = out["y_logits"].numpy()
        pred_np = logits_np.argmax(axis=1)
        rec_mse_np = np.mean((x_np - rec_np) ** 2, axis=(1, 2, 3))
        batch_n = x_np.shape[0]
        batch_df = df.iloc[cursor:cursor + batch_n].reset_index(drop=True)
        cursor += batch_n
        for i in range(batch_n):
            row = {
                "image_id": Path(batch_df.loc[i, "filepath"]).stem,
                "filepath": batch_df.loc[i, "filepath"],
                "true_label_idx": int(y.numpy()[i]),
                "true_label": classes[int(y.numpy()[i])],
                "pred_label_idx": int(pred_np[i]),
                "pred_label": classes[int(pred_np[i])],
                "recon_mse_per_pixel": float(rec_mse_np[i]),
                "dominant_archetype": f"A{int(np.argmax(a_np[i])) + 1}",
                "dominant_weight": float(np.max(a_np[i])),
                "entropy": float(entropy_rows(a_np[i:i+1])[0]),
            }
            for j in range(a_np.shape[1]):
                row[f"A{j+1}_weight"] = float(a_np[i, j])
            for j in range(mu_np.shape[1]):
                row[f"latent_{j+1}"] = float(mu_np[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_archetype_distances(z_fixed: np.ndarray):
    vals = []
    for i in range(len(z_fixed)):
        for j in range(i + 1, len(z_fixed)):
            vals.append(float(np.linalg.norm(z_fixed[i] - z_fixed[j])))
    return vals


def save_examples_by_class_and_archetype(out_df: pd.DataFrame, run_dir: Path, img_size: int, top_n: int = 3):
    """
    Saves visual proofs: for every lesion type and every archetype, show real HAM10000
    examples with the highest contribution of that archetype.
    """
    examples_dir = run_dir / "interpretability_examples"
    examples_dir.mkdir(exist_ok=True)

    for lesion in sorted(out_df["true_label"].unique()):
        lesion_df = out_df[out_df["true_label"] == lesion]
        for col in [c for c in out_df.columns if c.endswith("_weight") and c.startswith("A")]:
            a_name = col.replace("_weight", "")
            top = lesion_df.sort_values(col, ascending=False).head(top_n)
            if len(top) == 0:
                continue
            imgs = [load_one_image_for_display(fp, img_size) for fp in top["filepath"].tolist()]
            title = f"{lesion}: top {a_name} examples"
            safe_lesion = lesion.replace("/", "_").replace(" ", "_")
            save_image_grid(np.array(imgs), examples_dir / f"{safe_lesion}_{a_name}_top_examples.png", title=title, cols=top_n)


def generate_ham10000_tables(model: DeepAA, ds, df: pd.DataFrame, classes: list[str], cfg: DAAConfig,
                             run_dir: Path, model_name: str = "DeepAA"):
    """
    Produces the tables your instructor asked for: not only a single lesion type, but
    proof of how every HAM10000 diagnosis is represented by the archetypes.
    """
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    out_df = collect_full_outputs(model, ds, df.reset_index(drop=True), cfg, classes)
    out_df.to_csv(tables_dir / "per_image_archetype_interpretation.csv", index=False)

    k = model.k
    z_fixed = model.z_fixed.numpy()
    a_cols = [f"A{i+1}_weight" for i in range(k)]
    A = out_df[a_cols].values

    # Table 1: metrics for this run
    test_recon_mse = float(out_df["recon_mse_per_pixel"].mean())
    kl_vals = []
    simplex_penalties = []
    for x, y in ds:
        out = model(x, training=False)
        kl_vals.append(float(tf.reduce_mean(kl_standard_normal(out["mu"], out["logvar"])).numpy()))
        simplex_penalties.append(float(tf.reduce_mean(tf.reduce_sum(tf.square(model.z_fixed - out["z_pred"]), axis=1)).numpy()))
    table1 = pd.DataFrame([{
        "Dataset": "HAM10000",
        "Model": model_name,
        "K": k,
        "Recon_MSE_per_pixel_down": test_recon_mse,
        "Recon_term": "MSE",
        "Reg_term": "KL + simplex penalty" if model_name.lower() == "deepaa" else "simplex penalty",
        "KL_down": float(np.mean(kl_vals)),
        "Simplex_Penalty_down": float(np.mean(simplex_penalties)),
        "Stability_metric": "Var(A) per archetype/class",
        "Mean_Var_A_down": float(np.mean(np.var(A, axis=0))),
        "Accuracy_optional": float((out_df["true_label_idx"] == out_df["pred_label_idx"]).mean()),
    }])
    table1.to_csv(tables_dir / "table1_reconstruction_regularization_metrics.csv", index=False)

    # Instructor-focused replacement for emotion table: lesion-type archetype profiles
    table2 = out_df.groupby("true_label")[a_cols + ["entropy", "recon_mse_per_pixel"]].mean().reset_index()
    table2.rename(columns={"true_label": "Skin_lesion_type", "entropy": "Mean_entropy_down",
                           "recon_mse_per_pixel": "Mean_recon_MSE_down"}, inplace=True)
    table2["Dominant_archetype"] = table2[a_cols].idxmax(axis=1).str.replace("_weight", "", regex=False)
    table2.to_csv(tables_dir / "table2_skin_lesion_archetype_profiles.csv", index=False)

    # Table 3: class distribution per archetype, one row per archetype, columns are HAM10000 dx labels
    table3 = pd.crosstab(out_df["dominant_archetype"], out_df["true_label"], normalize="index") * 100
    table3 = table3.reindex([f"A{i+1}" for i in range(k)]).fillna(0).reset_index().rename(columns={"dominant_archetype": "Archetype"})
    table3.to_csv(tables_dir / "table3_class_distribution_per_archetype_percent.csv", index=False)

    # Table 4: effect of K. This run writes one row. Run K=3,5,7,9 and concatenate rows.
    table4 = pd.DataFrame([{
        "K": k,
        "Recon_MSE_down": test_recon_mse,
        "Archetype_Diversity_up": float(np.mean(pairwise_archetype_distances(z_fixed))),
        "Entropy_down": float(out_df["entropy"].mean()),
        "Interpretability_score_up": float(1.0 - out_df["entropy"].mean()),
        "Note": "For final Table 4, run --latent_dim 2,4,6,8 to obtain K 3,5,7,9."
    }])
    table4.to_csv(tables_dir / "table4_effect_of_K_single_run.csv", index=False)

    # Table 5: pairwise archetype distances
    dists = pairwise_archetype_distances(z_fixed)
    table5 = pd.DataFrame([{
        "Model": model_name,
        "K": k,
        "Mean_Dist_up": float(np.mean(dists)),
        "Min_Dist_up": float(np.min(dists)),
        "Max_Dist_up": float(np.max(dists)),
    }])
    table5.to_csv(tables_dir / "table5_pairwise_archetype_distances.csv", index=False)

    # Table 6: coefficient sparsity/simplex usage
    table6 = pd.DataFrame([{
        "Model": model_name,
        "K": k,
        "Avg_Active_Archetypes_down": active_archetype_count(A),
        "Mean_Entropy_down": float(out_df["entropy"].mean()),
        "Sparsity_up": gini_like_sparsity(A),
        "Usage_Per_Archetype": "; ".join([f"A{i+1}={float((out_df['dominant_archetype'] == f'A{i+1}').mean()*100):.1f}%" for i in range(k)])
    }])
    table6.to_csv(tables_dir / "table6_coefficient_sparsity_simplex_usage.csv", index=False)

    # Table 7: comparison row for DeepAA/AANet. AANet needs a separate baseline run.
    table7 = pd.DataFrame([{
        "Model": model_name,
        "Recon_down": test_recon_mse,
        "Stability_up": float(1.0 / (1e-8 + np.mean(np.var(A, axis=0)))),
        "Interpretability_up": float(1.0 - out_df["entropy"].mean()),
        "Complexity_down": int(np.sum([np.prod(v.shape) for v in model.trainable_variables])),
        "Note": "Run the baseline with AANet/deterministic settings if a true comparison is required."
    }])
    table7.to_csv(tables_dir / "table7_model_comparison_single_run.csv", index=False)

    # Table 8: training configuration
    table8 = pd.DataFrame([{
        "Model": model_name,
        "LR": "set by --lr",
        "Batch": "set by --batch_size",
        "Epochs": "set by --epochs",
        "Noise": "posterior reparameterization during training",
        "Image_size": cfg.img_size,
        "Loss": f"{cfg.recon_w}*MSE + {cfg.class_w}*CE + {cfg.kl_w}*KL + {cfg.at_w}*simplex"
    }])
    table8.to_csv(tables_dir / "table8_training_configuration_template.csv", index=False)

    # Proof examples by lesion type and archetype
    save_examples_by_class_and_archetype(out_df, run_dir, cfg.img_size, top_n=3)

    # A README to tell the instructor what each proof file means
    with open(tables_dir / "README_for_instructor.txt", "w", encoding="utf-8") as f:
        f.write(
            "These outputs answer the requirement: DeepAA interpretation for EVERY HAM10000 skin lesion type.\n\n"
            "Use table2_skin_lesion_archetype_profiles.csv to show the average archetype contribution for each lesion type.\n"
            "Use table3_class_distribution_per_archetype_percent.csv to show which lesion types each archetype mainly represents.\n"
            "Use per_image_archetype_interpretation.csv as per-image proof: image id, true label, predicted label, reconstruction error, and all A weights.\n"
            "Use interpretability_examples/*.png as visual proof: top real images per lesion type and archetype.\n"
            "For K comparison, run four experiments: --latent_dim 2, --latent_dim 4, --latent_dim 6, --latent_dim 8. These correspond to K=3,5,7,9.\n"
        )

    return {
        "per_image": out_df,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "table5": table5,
        "table6": table6,
        "table7": table7,
        "table8": table8,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    set_gpu_memory_growth()

    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads_dir", type=str, default=str(Path.home() / "Downloads"))
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="Optional: explicit path to HAM labels CSV (HAM10000_metadata.csv).")
    parser.add_argument("--img_dirs", type=str, nargs="*", default=None,
                        help="Optional: explicit image directories (part1 part2).")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)

    # Loss weights (you can tune)
    parser.add_argument("--recon_w", type=float, default=1.0)
    parser.add_argument("--class_w", type=float, default=1.0)
    parser.add_argument("--kl_w", type=float, default=1e-2)
    parser.add_argument("--at_w", type=float, default=1.0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    downloads_dir = Path(args.downloads_dir).expanduser().resolve()

    if args.img_dirs is None or len(args.img_dirs) == 0:
        img_dirs = find_ham_dirs(downloads_dir)
        if len(img_dirs) == 0:
            raise FileNotFoundError(
                f"Could not auto-find image dirs in {downloads_dir}. "
                f"Pass --img_dirs <part1> <part2> explicitly."
            )
    else:
        img_dirs = [Path(d).expanduser().resolve() for d in args.img_dirs]

    if args.labels_csv is None:
        labels_csv = find_labels_csv(downloads_dir, img_dirs)
    else:
        labels_csv = Path(args.labels_csv).expanduser().resolve()

    df, classes = load_ham_dataframe(labels_csv, img_dirs)
    num_classes = len(classes)

    train_df, val_df, test_df = stratified_split(df, args.val_frac, args.test_frac, args.seed)

    train_ds = make_tf_dataset(train_df, args.img_size, args.batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_df, args.img_size, args.batch_size, shuffle=False)
    test_ds = make_tf_dataset(test_df, args.img_size, args.batch_size, shuffle=False)

    cfg = DAAConfig(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        recon_w=args.recon_w,
        class_w=args.class_w,
        kl_w=args.kl_w,
        at_w=args.at_w
    )

    model = DeepAA(cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Results dir
    run_dir = Path("Results_HAM_DAA") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    print("===== DATA =====")
    print("Downloads:", downloads_dir)
    print("Labels CSV:", labels_csv)
    print("Image dirs:", [str(d) for d in img_dirs])
    print("Classes:", classes)
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print("Run dir:", run_dir)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train
        tr = []
        for x, y in train_ds:
            total, rec, cls, kl, at = train_step(model, optimizer, x, y, cfg)
            tr.append([total.numpy(), rec.numpy(), cls.numpy(), kl.numpy(), at.numpy()])
        tr = np.mean(tr, axis=0)

        # Val
        va = []
        for x, y in val_ds:
            total, rec, cls, kl, at = eval_step(model, x, y, cfg)
            va.append([total.numpy(), rec.numpy(), cls.numpy(), kl.numpy(), at.numpy()])
        va = np.mean(va, axis=0)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train: total={tr[0]:.4f} rec={tr[1]:.4f} cls={tr[2]:.4f} kl={tr[3]:.4f} at={tr[4]:.4f} || "
                f"val: total={va[0]:.4f} rec={va[1]:.4f} cls={va[2]:.4f} kl={va[3]:.4f} at={va[4]:.4f}"
            )

    # Test
    te = []
    for x, y in test_ds:
        total, rec, cls, kl, at = eval_step(model, x, y, cfg)
        te.append([total.numpy(), rec.numpy(), cls.numpy(), kl.numpy(), at.numpy()])
    te = np.mean(te, axis=0)
    print(
        f"TEST | total={te[0]:.4f} rec={te[1]:.4f} cls={te[2]:.4f} kl={te[3]:.4f} at={te[4]:.4f}"
    )

    # Save weights
    model.save_weights(str(run_dir / "daa_ham_weights.weights.h5"))

    # ---- Visualizations ----
    # 1) Decode Z_fixed (generated archetypes)
    z_fixed = model.z_fixed.numpy()  # [k,d]
    x_gen, _ = model.decode(tf.constant(z_fixed, tf.float32), training=False)
    x_gen = x_gen.numpy()
    save_image_grid(x_gen, run_dir / "plots" / "archetypes_generated.png",
                    title="Generated Archetypes (decode(Z_fixed))", cols=model.k)

    # 2) Closest real images to each archetype
    # Compute latent means on a subset (val+test is enough & faster)
    mus, labs = compute_latents(model, val_ds)
    closest_imgs = []
    closest_titles = []
    for i in range(model.k):
        dists = np.linalg.norm(mus - z_fixed[i], axis=1)
        idx = int(np.argmin(dists))
        # retrieve original filepath from val_df using idx
        fp = val_df.iloc[idx]["filepath"]
        closest_imgs.append(load_one_image_for_display(fp, args.img_size))
        closest_titles.append(f"AT{i+1}: closest real")
    save_image_grid(np.array(closest_imgs), run_dir / "plots" / "archetypes_closest_real.png",
                    title="Closest Real Images to Z_fixed", cols=model.k)

    # 3) Latent scatter + CSV
    # (use test set for a nice plot)
    mus_t, labs_t = compute_latents(model, test_ds)
    plot_latent_scatter(mus_t, labs_t, classes, run_dir / "plots" / "latent_scatter.png")

    lat_cols = [f"ldim{i}" for i in range(cfg.latent_dim)]
    out_df = pd.DataFrame(mus_t, columns=lat_cols)
    out_df["label_idx"] = labs_t
    out_df["label_str"] = [classes[i] for i in labs_t]
    out_df.to_csv(run_dir / "latent_codes_test.csv", index=False)

    # 4) Instructor-required HAM10000 interpretability tables and visual proof examples
    generate_ham10000_tables(model, test_ds, test_df, classes, cfg, run_dir, model_name="DeepAA")

    # 5) Latent traversal (only meaningful + pretty for latent_dim=2 => k=3)
    if cfg.latent_dim == 2:
        w = barycentric_weights_grid(n_per_axis=15)  # [N,3]
        z = w @ z_fixed[:3]  # [N,2]
        imgs, _ = model.decode(tf.constant(z, tf.float32), training=False)
        imgs = imgs.numpy()
        triangle_grid_plot(imgs, n_per_axis=15, out_path=run_dir / "plots" / "latent_traversal.png",
                           title="Latent Traversal over Simplex (k=3)")

    print("\nDONE. Results saved in:")
    print(run_dir)


if __name__ == "__main__":
    main()