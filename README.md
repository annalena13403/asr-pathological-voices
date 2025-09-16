# Data Augmentation for ASR of Pathological Voices

This repository contains the code, notebooks, and configuration files accompanying the bachelor thesis  
**“Improving Automatic Speech Recognition for Pathological Voices Using Data Augmentation.”**  
Bachelor thesis, Free University of Bozen-Bolzano.

The project investigates how data augmentation and synthesized voices can improve the performance of ASR systems on **pathological speech**, where training data is extremely limited.

---

## Repository Structure

```
Feature_Extraction_and_Filtering/
│   extract_features.ipynb      # Extract acoustic features of acoustic samples (Praat, Librosa, NumPy)
│   filter_synth_samples.ipynb  # Random Forest feature importance + filtering pipeline
│
Fine-tuning/
│   analyse_experiment.ipynb    # Evaluate & compare ASR fine-tuning runs (WER, plots)
│   config.yaml                 # NeMo trainer config for fine-tuning (dataset paths, options, augmentation toggles)
│   trainer.py                  # Fine-tuning pipeline with weighting & augmentation
│
.gitignore
README.md
requirements.txt
```

---

## Workflow Overview

1. **ASR Fine-Tuning**
   - Fine-tune NVIDIA **NeMo Conformer-CTC** models (e.g., `stt_it_conformer_ctc_large`) on:
     - **Collected pathological recordings**
     - **Synthesized pathological recordings** 
     - **Combined datasets**
   - Configurable options:
     - Oversampling factors for collected data (↑ weight of real samples).
     - Undersampling/filtering synthesized (See 3. **Acoustic Feature Extraction** & 4. **Feature Importance & Filtering**). 
     - Toggle for dynamic perturbations.
   - Training outputs: `metrics.csv` (loss, WER, etc.), checkpoints in `output_dir`.

2. **Experiment Analysis**
   - Compute **baseline WER** (pretrained model on pathological data).
   - Compare augmentation strategies across runs:
     - Collected-only, Synth-only, Combined, Oversampling, Filtering.
   - Plot WER curves with **95% confidence intervals**.
   - Report best epoch results (absolute improvement + relative error reduction).
   - Generate visual summaries for thesis figures.

3. **Acoustic Feature Extraction**
   - Extract features from audio recordings:
     - **Perturbation measures**: jitter, shimmer.
     - **Noise measures**: HNR, NHR, ZCR.
     - **Pitch & resonance**: F0, formants (F1–F3).
     - **Spectral features**: centroid, slope.
     - **Cepstral features**: MFCCs.
     - **Intensity measures**: mean, std, dynamic range.
   - Outputs: `audio_features_out.csv` containing all extracted features.

4. **Feature Importance & Filtering**
   - Train a **Random Forest classifier** on collected data (healthy vs. pathological).
   - Rank features by **Gini importance** (select top-K).
   - Use these features to:
     - Compare distributions of real vs. synthesized speech (Wasserstein distance).
     - Filter synthesized datasets via classifiers:
       - **All predicted pathological** samples.
       - **High-confidence pathological** samples (≥ 80%).
   - Outputs:
     - `all_features_gini_ranked.csv`
     - Filtered datasets (e.g., `pathological_filtered.csv`).
---

## Reproducing Experiments from the Thesis

All main experiments described in the thesis can be reproduced using the training script `Fine-tuning/trainer.py`. Experiment settings are adjusted directly in the code (see the `ModelTrainer._load_model` section).

### Steps

1. **Run fine-tuning**
   - Open `trainer.py`, go to the `ModelTrainer._load_model` section and uncomment the dataset configuration block corresponding to your experiment:
     - **Collected-only** → use only `collected_train.json` / eval splits.
     - **Synth-only** → use only `synth_train.json` / eval splits (+ collected eval for reference).
     - **Combined** → use both `collected_train.json` and `synth_train.json`.
   - Toggle `use_augmentation=True` in the config to enable perturbation-based augmentation.
    - Adjust `weight_factor` to oversample real data (e.g., `2` doubles its weight).

2. **For  Experiments with the filtered synthesized data**
      - Follow the steps from the notebooks in the Feature_Extraction_and_Filtering folder.
      - Extract acoustic features from your synthesized dataset. [Extract Features](Feature_Extraction_and_Filtering/extract_features.ipynb)
      - Extract acoustic features from your collected dataset. [Extract Features](Feature_Extraction_and_Filtering/extract_features.ipynb)
      - Train a Random Forest classifier on the collected data and rank features by importance. [Feature Importance & Filtering](Feature_Extraction_and_Filtering/filter_synth_samples.ipynb)
      - Filter the synthesized dataset using the trained classifier. [Filter Synth Samples](Feature_Extraction_and_Filtering/filter_synth_samples.ipynb)

      - Adjust in the [config.yml](Fine-tuning/config.yaml) the `synthesized_dataset:` config to point to the filtered synthesized    dataset you want to use.

---
## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/annalena13403/asr-pathological-voices
cd asr-pathological-voices
pip install -r requirements.txt
```
---
