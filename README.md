# Deep Archetypal Analysis on HAM10000 — Interpretability Version

This project applies **Deep Archetypal Analysis (DeepAA)** to the **HAM10000 skin lesion dataset** using **TensorFlow 2 / Keras**.

The goal is not only to train a reconstruction model, but also to explain how DeepAA interprets **every skin lesion type** in the dataset using archetypes, visual examples, and table-ready metrics.

This version is based on the updated script:

```text
 daa_ham10000_interpretability_fixed_v2.py
```

---

## 1. Project Purpose

The model is designed to answer the following research question:

> How can Deep Archetypal Analysis help interpret different HAM10000 skin lesion types using learned archetypes?

Instead of producing results for only one lesion type, this version generates interpretation outputs for all HAM10000 classes:

```text
akiec, bcc, bkl, df, mel, nv, vasc
```

The model exports tables and visual proof examples showing how each lesion type is represented by the learned archetypes.

---

## 2. Main Model Idea

The model uses a **variational encoder–decoder structure** combined with **Deep Archetypal Analysis**.

Each image is encoded into a latent representation. This latent representation is not free-floating; instead, it is expressed as a **convex mixture of archetypes**.

The model uses:

- An encoder for extracting image features.
- Softmax-constrained mixture weights `alpha`.
- Fixed simplex vertices as archetype targets.
- A variational latent bottleneck with KL divergence.
- A decoder for image reconstruction.
- A classification head using HAM10000 lesion labels as side information.

---

## 3. Core Requirements Implemented

This implementation satisfies the following requirements:

- Uses a **variational autoencoder / variational information bottleneck structure**.
- Learns a latent representation where each data point is a **convex mixture of archetypes**.
- Uses `K` archetypes placed at the vertices of a fixed simplex.
- Sets:

```text
latent_dim = K - 1
```

- Uses softmax-constrained mixture coefficients so that:

```text
alpha_i >= 0
sum(alpha_i) = 1
```

- Includes reconstruction loss.
- Includes KL divergence loss.
- Includes archetype loss that forces predicted archetypes toward fixed simplex vertices.
- Includes side-information classification loss using HAM10000 diagnosis labels.
- Supports configurable archetype numbers such as:

```text
K = 3, 5, 7, 9
```

- Exports generated archetype images.
- Exports closest real samples to each archetype.
- Exports reconstruction grids.
- Exports latent/simplex plots.
- Exports interpolation and mixture visualizations when supported.
- Exports table-ready metrics for the report.

---

## 4. Dataset Location on Windows

Place the HAM10000 dataset inside:

```bat
C:\Users\User\Downloads
```

The following files/folders must exist:

```text
C:\Users\User\Downloads\HAM10000_images_part_1
C:\Users\User\Downloads\HAM10000_images_part_2
C:\Users\User\Downloads\HAM10000_metadata.csv
C:\Users\User\Downloads\daa_ham10000_interpretability_fixed_v2.py
```

The metadata file should contain the `image_id` column and the diagnosis label column, usually called `dx`.

---

## 5. Create and Activate the Environment

Open **Command Prompt** and run:

```bat
cd C:\Users\User\Downloads
py -3.11 -m venv ham_env
ham_env\Scripts\activate
```

After activation, the terminal should look like this:

```bat
(ham_env) C:\Users\User\Downloads>
```

---

## 6. Install Requirements

Run these commands one by one:

```bat
python -m pip install --upgrade pip
```

Then:

```bat
pip install numpy pandas matplotlib tensorflow
```

Recommended Python version:

```text
Python 3.11.x
```

TensorFlow may fail on Python versions that are too new, such as Python 3.14.

---

## 7. Run the Model

### Recommended first run: K = 5 archetypes

Since:

```text
K = latent_dim + 1
```

using `latent_dim = 4` gives:

```text
K = 5
```

Run:

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 32 --latent_dim 4
```

---

## 8. Run Different K Values

To fill comparison tables for different archetype numbers, run the model multiple times.

### K = 3

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 32 --latent_dim 2
```

### K = 5

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 32 --latent_dim 4
```

### K = 7

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 32 --latent_dim 6
```

### K = 9

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 32 --latent_dim 8
```

---

## 9. Training Output

During training, the terminal prints lines like:

```text
Epoch 001 | train: total=... rec=... cls=... kl=... at=... || val: total=... rec=... cls=... kl=... at=...
```

Meaning:

| Term | Meaning |
|---|---|
| `total` | Full training objective |
| `rec` | Reconstruction loss |
| `cls` | Side-information classification loss |
| `kl` | KL divergence loss |
| `at` | Archetype loss |
| `train` | Training set performance |
| `val` | Validation set performance |

A good sign is when validation `total`, `rec`, `cls`, and `at` generally decrease.

KL may increase slightly or stabilize because the variational latent space starts encoding more information.

---

## 10. Results Folder

After training finishes, the script prints something like:

```text
DONE. Results saved in:
Results_HAM_DAA\YYYYMMDD_HHMMSS
```

Open the newest folder inside:

```text
C:\Users\User\Downloads\Results_HAM_DAA
```

The results folder contains:

```text
plots/
tables/
interpretability_examples/
latent_codes_test.csv
daa_ham_weights.weights.h5
```

---

## 11. Important Output Folders

### `plots/`

Contains visual outputs such as:

```text
archetypes_generated.png
archetypes_closest_real.png
latent_scatter.png
latent_traversal.png
reconstruction_grid.png
```

These images help show how the model reconstructs lesions and visualizes archetypes.

### `tables/`

Contains CSV files ready to use in the report or paper.

Important files include:

```text
table1_metrics.csv
table2_skin_lesion_archetype_profiles.csv
table3_class_distribution_per_archetype_percent.csv
table5_pairwise_archetype_distances.csv
table6_sparsity_simplex.csv
table7_model_comparison.csv
table8_training_configuration.csv
per_image_archetype_interpretation.csv
```

### `interpretability_examples/`

Contains real HAM10000 image examples grouped by lesion type and archetype.

This folder is useful as visual proof that DeepAA can interpret every lesion type, not only one class.

---

## 12. Most Important Tables for the Report

### Table 2: Skin Lesion Archetype Profiles

File:

```text
tables/table2_skin_lesion_archetype_profiles.csv
```

This table shows how each lesion type is represented by the learned archetypes.

It directly answers:

> How does DeepAA help interpret every skin lesion type in the dataset?

### Table 3: Class Distribution per Archetype

File:

```text
tables/table3_class_distribution_per_archetype_percent.csv
```

This table shows which lesion classes dominate each archetype.

### Per-Image Interpretation Table

File:

```text
tables/per_image_archetype_interpretation.csv
```

This table gives interpretation for each image, including:

```text
image_id
true lesion type
predicted lesion type
dominant archetype
archetype mixture weights
reconstruction error
```

---

## 13. Common Warnings

### oneDNN message

You may see:

```text
oneDNN custom operations are on
```

This is not an error. The model can continue running.

### GPU warning on Windows

You may see:

```text
TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11
```

This means the model is running on CPU. It is not an error.

Training may be slower, but it can still work.

---

## 14. If Training Is Too Slow

Use a smaller batch size.

Try:

```bat
--batch_size 16
```

If it is still slow, try:

```bat
--batch_size 8
```

Example:

```bat
python daa_ham10000_interpretability_fixed_v2.py --labels_csv "C:\Users\User\Downloads\HAM10000_metadata.csv" --img_dirs "C:\Users\User\Downloads\HAM10000_images_part_1" "C:\Users\User\Downloads\HAM10000_images_part_2" --epochs 50 --batch_size 16 --latent_dim 4
```

---

## 15. Suggested Explanation for the Report

The updated DeepAA model explains HAM10000 skin lesion images by representing each image as a convex mixture of archetypes located at fixed simplex vertices. The learned mixture coefficients show how strongly each image belongs to each archetype. By grouping these coefficients by lesion type, the model provides class-level interpretability for all HAM10000 categories. The generated archetype images, closest real examples, reconstruction grids, and class distribution tables provide both visual and quantitative evidence that DeepAA can support interpretation of different skin lesion types.

---

## 16. License

Add your preferred license, such as:

```text
MIT License
```
