# Deep Archetypal Analysis (DAA) on HAM10000 — TensorFlow 2 (Windows)

This project runs **Deep Archetypal Analysis (DAA)** on the **HAM10000 skin lesion dataset** using **TensorFlow 2 / Keras**.
It trains an encoder–decoder model that learns **archetypes** (extremal representations) and produces:

- **Generated archetypes** (decode of simplex vertices `Z_fixed`)
- **Closest real images** to each archetype
- **Latent codes CSV**
- For **latent_dim = 2** (3 archetypes): a **simplex traversal plot**

---

## 1) Dataset location (Windows)

Your dataset is in:

**`C:\Users\User\Downloads`**

Make sure these exist:

- **`HAM10000_images_part_1\`** (images)
- **`HAM10000_images_part_2\`** (images)
- **`HAM10000_metadata.csv`** (labels/metadata)

> The script matches images using the `image_id` column in the CSV and filenames like `ISIC_XXXXXXX.jpg`.

---

## 2) Create & activate a virtual environment (recommended)

Open **Command Prompt** and run:

```bat
cd C:\Users\User\Downloads
py -m venv ham_env
ham_env\Scripts\activate
```

You should see:

`(ham_env) C:\Users\User\Downloads>`

---

## 3) Install requirements

Upgrade pip and install packages:

```bat
python -m pip install --upgrade pip
pip install numpy pandas matplotlib tensorflow
```

### If TensorFlow fails to install
This usually means your Python version is too new for the available TensorFlow wheels on Windows.

Recommended Python versions for easiest install:
- **Python 3.10** or **Python 3.11**

Then recreate the venv and reinstall.

---

## 4) Run training (recommended commands)

### ✅ Latent dimension 2 (3 archetypes) — recommended
Good for a clear simplex structure + traversal plot.

```bat
python daa_ham10000_tf2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 40 --batch_size 32 --latent_dim 2 --kl_w 0.001 --at_w 5.0 --a_temp 0.6 --a_entropy_w 0.02
```

### Example: latent dimension 8 (9 archetypes)
(Your previous run style)

```bat
python daa_ham10000_tf2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 10 --batch_size 64 --latent_dim 8 --kl_w 0.001 --at_w 5.0 --a_temp 0.7 --a_entropy_w 0.05
```

---

## 5) Where results are saved

After training, the script prints:

`DONE. Results saved in: Results_HAM_DAA\YYYYMMDD_HHMMSS`

Inside that folder, check:

- `plots/archetypes_generated.png`
- `plots/archetypes_closest_real.png`
- `plots/latent_scatter.png`
- `plots/latent_traversal.png` *(only when latent_dim=2)*
- `latent_codes_test.csv`
- `daa_ham_weights.weights.h5` *(trained weights)*

---

## 6) Understanding the outputs (quick)

### `archetypes_generated.png`
What the decoder generates when decoding simplex vertices `Z_fixed`.
- Early epochs may look **blurry / similar** → train longer.

### `archetypes_closest_real.png`
Real images whose latent mean `mu(x)` is closest to each archetype vertex.
- Usually the **most interpretable** archetype visualization.

### `latent_scatter.png`
2D latent scatter plot **only meaningful when latent_dim = 2**.

### `latent_codes_test.csv`
Latent mean vectors + class label for the test set.

---

## 7) Common issues & fixes

### A) `ModuleNotFoundError: No module named 'numpy' / 'pandas' / ...`
Install missing libraries inside the venv:

```bat
pip install numpy pandas matplotlib tensorflow
```

### B) “oneDNN custom operations are on…”
This is **not an error**. If you want to turn it off:

```bat
set TF_ENABLE_ONEDNN_OPTS=0
```

### C) Out of memory / very slow
Lower batch size:

- Try: `--batch_size 16`
- If still heavy: `--batch_size 8`

### D) Images look all the same
Most common reasons:
- Too few epochs (train longer)
- KL too strong early → try smaller `--kl_w` (you used `0.001`, good)
- Archetype loss too weak → try bigger `--at_w`

---

## 8) Tips for better archetypes
- Start with **latent_dim=2** for a stable and interpretable simplex.
- Use **40–100 epochs** for clearer archetype images.
- If archetypes collapse (too similar), increase `--at_w` a bit or reduce `--kl_w` slightly.

---

## License
Add your preferred license (MIT / Apache-2.0 / etc).
