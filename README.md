# Potato Blight Classification System 🥔🌱

An automated deep learning computer vision system designed to help rapidly diagnose potato leaf diseases from smartphone photographs. Built by **Group 1**.

---

## 📌 Outline
- [Business Understanding](#-business-understanding)
- [Deliverables](#-deliverables)
- [Data](#-data)
- [Model Architecture & Training Strategy](#-model-architecture--training-strategy)
- [Performance](#-performance)
- [Explainability (Grad-CAM)](#-explainability-grad-cam)
- [Practical Constraints & Field Guidelines](#-practical-constraints--field-guidelines)

---

## Business Understanding

### The Problem
Potato farming is vital for both subsistence and income in Sub-Saharan Africa, South Asia, and Latin America. However, crops are highly susceptible to fungal and water-mould infections:
* **Early Blight** (*Alternaria solani*)
* **Late Blight** (*Phytophthora infestans*) — historically responsible for the Great Irish Famine.

Left undetected, these diseases can devastate **30% to 70% of a crop yield**, leading to severe food insecurity and financial ruin. Traditional disease identification relies heavily on manual, in-person field inspections by scarce agricultural experts. By the time symptoms are widely visible and a professional arrives, the window for effective treatment has often closed.

### The Solution
An automated mobile-ready diagnostic tool capable of classifying leaf conditions in **under 2 seconds**. This provides low-cost plant disease self-diagnosis right at the farm gate, optimizing intervention timing and protecting livelihoods.

### Core Stakeholders
1. **Smallholder Farmers:** For early, rapid self-diagnosis to protect crop yields and household income.
2. **Agricultural Extension Officers:** A scalable digital tool to streamline field advice and support services.
3. **NGOs & Food Security Bodies:** Data-driven mitigation of crop loss in structurally vulnerable agricultural communities.

---

## Deliverables
1. GitHub repo
2. Jupyter Notebook
3. [Streamlit deployment](https://potato-blight-classification-2o9j8q7bjtur3yfp8nztav.streamlit.app/)
4. [Non-technical Presentation](https://canva.link/3xzlmepr150avjv)
5. [Tableau](https://public.tableau.com/app/profile/yvonnie.wanyoike/viz/Potato_Blight_Classification_Analysis/Story1)

---

## Data

The project utilizes a specific **2,152-image subset** of the benchmark open-source **PlantVillage** dataset, divided into three target classes:

| Leaf Condition Class | Image Count | Dataset Share |
| :--- | :---: | :---: |
| **Potato Early Blight** (*Alternaria solani*) | 1,000 | ~46.5% |
| **Potato Late Blight** (*Phytophthora infestans*) | 1,000 | ~46.5% |
| **Potato Healthy** | 152 | ~7.1% |
| **Total Dataset** | **2,152** | **100%** |

### Data Preprocessing & Augmentation
* **Normalization:** All input photographs are resized to standard resolutions ($224 \times 224 \times 3$) and pixel values scaled from btn 0 and 255 to the range of [-1, 1].
* **Class Imbalance Mitigation:** Because the `Healthy` class represents only ~7% of the input space, Scikit-Learn’s `compute_class_weight` utility was integrated during loss minimization steps to prevent the network from ignoring or misclassifying underrepresented healthy samples.
* **Augmentation:** Applied explicitly to the training split to combat over-fitting and improve generalizability.

---

## Model Architecture & Training Strategy

The system leverages **Transfer Learning** using a pretrained **MobileNetV2** backbone network, chosen specifically because its highly serialized, lightweight footprint is optimized for low-compute mobile and edge deployment.

### Dual-Phase Training Orchestration
1. **Phase 1: Head Training (15 Epochs)**
   * The pretrained MobileNetV2 base feature extractor is completely frozen.
   * A custom dense classification head is attached and trained at a baseline learning rate of $\eta = 1\times10^{-3}$.
2. **Phase 2: Cooperative Fine-Tuning (20 Epochs)**
   * Deep network layers from index layer 130 onward are unfrozen.
   * The un-frozen top partition trains alongside the custom head using a highly restrictive, smaller fine-tuning step-size learning rate ($\eta = 5\times10^{-5}$) to carefully adapt high-level leaf features without obliterating pretrained weights.

### Optimization & Regularization Callbacks
* Explicit seed configurations (`SEED = 42`) ensure pipeline reproducibility.
* `ModelCheckpoint` preserves weights to `'best_potato_model.h5'`.
* `EarlyStopping` prevents overfitting, coupled with validation adjustments handled dynamically via `ReduceLROnPlateau`.

---

## Performance

To account for the severe dataset imbalance, the system relies strictly on the **F1-Score** (harmonic mean of Precision and Recall) rather than raw accuracy.

### Test Set Performance
The model achieves an outstanding **Overall Macro F1-Score of 0.939 (>0.9)**. 

* **Early Blight F1-Score:** `0.987`
* **Late Blight F1-Score:** `0.974`
* **Healthy Leaf F1-Score:** `0.857` 

---

## Explainability (Grad-CAM)

To establish clinical trust, the pipeline incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization layers. Grad-CAM generates heatmaps overlaid directly onto input samples. This allows verification that the neural network’s focus corresponds directly to leaf lesions and structural abnormalities, rather than noise or uniform background clutter.

---

## Practical Constraints & Field Guidelines

Because the underlying model was trained exclusively on clean, controlled *PlantVillage* source images (characterized by single-leaf photographs captured against uniform backgrounds under steady, consistent lighting), users **should follow certain guidelines** for best results:

* **Single Subject:** Photograph exactly one single leaf close-up, ensuring it fills most of the camera frame.
* **Natural, Even Light:** Capture imagery under neutral lighting—avoid harsh direct sunlight, dark shadows, or heavy flash glare.
* **Plain Backgrounds:** Keep the background as uniform and plain as possible (e.g., holding the leaf up against the sky or a clear surface).
* **Grad-CAM Verification:** Always check the Grad-CAM overlay. If the heatmap highlights arbitrary background zones instead of leaf tissue, the output prediction should be treated with skepticism.
* **Strict Scope Boundaries:** The tool is specialized exclusively for Early Blight, Late Blight, and Healthy states. It possesses *no out-of-distribution (OOD) layer* and is **not** a general-purpose "crop doctor." If provided with non-potato leaves or unrelated backgrounds, it will unpredictably force a label onto the image.

---
