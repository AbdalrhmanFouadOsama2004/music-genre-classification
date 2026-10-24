# 🎵 Music Genre Classification

A machine learning and deep learning project for classifying music into **10 genres** using handcrafted audio features, Convolutional Neural Networks (CNNs), and transfer learning.

The project uses the **GTZAN music genre dataset** and compares three approaches under the same track-level train/validation/test split.

## 📌 Project Overview

Music genre classification is an audio classification task: given a music clip, predict the genre it most likely belongs to.

This project explores three approaches:

| Approach | Input representation | Models |
|---|---|---|
| **A — Traditional ML** | 60 handcrafted audio features | Logistic Regression, KNN, SVM, Random Forest, HistGradientBoosting, MLP |
| **B — Deep Learning** | 128 × 128 Mel-spectrograms | CNN trained from scratch |
| **C — Transfer Learning** | 128 × 128 Mel-spectrograms | ImageNet-pretrained MobileNetV2, with optional EfficientNetB0 |

The goal is to compare their performance at both the **3-second segment level** and the **30-second track level**.

## ✨ Key Features

- Extracts audio features with `librosa`.
- Converts audio segments into Mel-spectrogram representations.
- Compares traditional machine learning, a custom CNN, and transfer learning.
- Uses a track-level split to prevent segments from the same song appearing in multiple data splits.
- Evaluates predictions at both segment and track level.
- Uses SpecAugment-style time and frequency masking when training the CNN.
- Applies early stopping and learning-rate scheduling.
- Visualizes training history, model comparisons, feature importance, and confusion matrices.
- Caches extracted features to reduce repeated preprocessing.
- Includes a helper function to predict the top genres for an audio file.
- Saves trained models and a CSV file containing model comparison results.

## 🎼 Dataset

This project uses the [GTZAN Music Genre Classification dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

The notebook expects a directory named `genres_original` containing one subdirectory per genre.

The ten target genres are:

`blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`

Each original track is processed as ten 3-second segments, producing multiple training examples from each track.

> **Important:** the split is performed by track, not by segment. All segments from a song remain in the same split, reducing data leakage and making evaluation more reliable.

## 🧰 Tech Stack

- **Python**
- **NumPy** and **Pandas** — data manipulation
- **Librosa** — audio loading and feature extraction
- **Scikit-learn** — traditional ML models, data splitting, and evaluation
- **TensorFlow / Keras** — CNN and transfer learning
- **Matplotlib** and **Seaborn** — visualizations
- **Joblib** — parallel feature extraction
- **KaggleHub** — dataset download when needed
- **Jupyter Notebook** — experimentation and analysis

## 🏗️ Methodology

### 1. Audio preprocessing

- Audio is loaded at a sample rate of **22,050 Hz**.
- Each 30-second track is divided into **ten 3-second segments**.
- Short segments are padded when necessary to maintain a consistent length.

### 2. Handcrafted feature extraction

For each segment, the notebook extracts a **60-dimensional feature vector** containing:

- Chroma statistics
- RMS energy statistics
- Spectral centroid statistics
- Spectral bandwidth statistics
- Spectral rolloff statistics
- Zero-crossing-rate statistics
- Spectral contrast means
- MFCC means and variances (20 coefficients)
- Track tempo

These features are used to train and compare traditional classifiers.

### 3. Mel-spectrogram generation

Each segment is also represented as a **128 × 128 Mel-spectrogram**. The spectrogram is converted to a decibel scale and stored as an 8-bit image-like array to reduce memory usage.

### 4. Data splitting

The data is split by original track using stratification by genre:

- **70%** training
- **15%** validation
- **15%** testing

The split uses a fixed random seed (`42`). The validation set supports model selection, while the test set is reserved for final evaluation.

### 5. Model training

**Approach A — Traditional machine learning**

The notebook compares six classifiers on the handcrafted features. Models that require feature scaling are placed in pipelines with `StandardScaler`.

**Approach B — CNN from scratch**

A four-block CNN learns patterns from Mel-spectrograms. Training includes time/frequency masking, dropout, early stopping, and learning-rate reduction on a plateau.

**Approach C — Transfer learning**

A pretrained image backbone is adapted to spectrogram classification. The notebook initially freezes the backbone and trains a classification head, then fine-tunes part of the backbone using a smaller learning rate. `MobileNetV2` is the default; `EfficientNetB0` is also supported by the model-building function.

### 6. Evaluation

Models are evaluated using:

- Segment-level accuracy
- Track-level accuracy
- Classification reports
- Confusion matrices
- Training and validation curves
- A comparison table and accuracy chart

For track-level predictions, the model probabilities from a track's segments are averaged before choosing the predicted genre.

## 📂 Project Structure

```text
.
├── music_genre_classification.ipynb   # End-to-end workflow
├── README.md                          # Project documentation
├── gtzan_cache.npz                    # Generated preprocessing cache
├── cnn_scratch.keras                  # Saved CNN model
├── transfer_model.keras               # Saved transfer-learning model
└── results.csv                        # Model comparison results
```

The cache, model files, and results CSV are generated by running the notebook; they do not need to be present before the first run. Dataset files are not included in this repository by default.

## 🚀 Getting Started

### Requirements

- Python 3.10 or a compatible Python version supported by your installed TensorFlow release
- Jupyter Notebook or JupyterLab
- Internet access for automatic dataset download, unless the dataset is already available locally
- A GPU is recommended for the CNN and transfer-learning experiments; the traditional ML approach can run on a CPU

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your Git repository URL and folder name.

### 2. Install dependencies

The notebook installs some dependencies in its setup cell. You can also install the main packages manually:

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow joblib kagglehub jupyter
```

TensorFlow installation requirements vary by operating system and Python version. Follow the [official TensorFlow installation guide](https://www.tensorflow.org/install) if installation fails.

### 3. Set up the dataset

The notebook first searches for `genres_original` under `/kaggle/input`. If it cannot find the folder, it attempts to download the dataset using KaggleHub.

For automatic download, make sure your Kaggle account has access to the dataset and configure Kaggle credentials as required by KaggleHub. **Never commit API tokens or `kaggle.json` to GitHub.**

Alternatively, set `AUDIO_ROOT` in the notebook to the path of your local `genres_original` directory.

Expected layout:

```text
genres_original/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

### 4. Run the notebook

```bash
jupyter notebook music_genre_classification.ipynb
```

Run the cells from top to bottom. The first run extracts features and creates `gtzan_cache.npz`; later runs can reuse the cache when it is available.

> **Note:** preprocessing and neural-network training can take time. A GPU is recommended for the CNN and transfer-learning sections.

## 🔍 Predicting the Genre of an Audio File

After running the notebook and training the transfer-learning model, use the included `predict_genre` helper:

```python
predictions = predict_genre("path/to/audio.wav", model=tl, top_k=3)

for genre, probability in predictions:
    print(f"{genre}: {probability:.2%}")
```

The function returns the top `k` genre predictions and their averaged model scores.

**Important:** the function uses the trained model and genre mapping from the notebook session. Run the notebook setup and training cells first. The returned values are model probabilities/scores, not a guarantee that a prediction is correct.

## 📊 Results

The notebook generates a comparison table sorted by test track-level accuracy, a chart comparing segment-level and track-level accuracy, and confusion matrices for selected models.

**Add your measured results here after running the notebook.** Do not report example or expected accuracies as experimental results.

The notebook writes the model comparison table to `results.csv`.

## ⚠️ Limitations

- GTZAN is a relatively small dataset for training deep neural networks from scratch.
- The dataset is known to have data-quality concerns, including possible duplicates and incorrect labels.
- Music genres can overlap in style, making some examples inherently ambiguous.
- Track-level evaluation is more representative of full-clip classification, but the held-out test set is still limited in size.
- Transfer learning from image models is an experiment: image features do not necessarily transfer optimally to audio spectrograms.
- Results may vary with library versions, hardware, model initialization, and training settings.

For stronger conclusions, repeat experiments with multiple seeds or use stratified group cross-validation by track.

## 📦 Model Artifacts and Large Files

The trained model files can be large, especially transfer_model.keras. To keep this repository lightweight and easy to clone, trained models and generated files are not included in the GitHub repository.

Files Not Included

transfer_model.keras — Trained transfer learning model.

🚀 How to Get Started

Clone this repository.

Install the required dependencies.

Download and configure the GTZAN dataset.

Open and run music_genre_classification.ipynb to preprocess the data and train the models.

The required model files will be generated during notebook execution.

Using a Pre-trained Model

If you want to make predictions without training the model again, you must download the corresponding pre-trained .keras file separately.

Note: Model files are excluded to reduce repository size. You may need to retrain the models or download the pre-trained weights before running inference.

## 🔮 Future Improvements

- Compare `MobileNetV2` and `EfficientNetB0` systematically.
- Evaluate audio-specific pretrained models such as YAMNet, PANNs, or Audio Spectrogram Transformers.
- Combine model predictions using probability averaging or an ensemble.
- Use stratified group cross-validation for more robust evaluation.
- Add a simple web interface for uploading audio and displaying predictions.
- Test different spectrogram resolutions and segment lengths.
- Track per-class precision, recall, and F1-score across all approaches.

## 💾 Generated Artifacts

After the relevant notebook cells run, the project saves:

- `cnn_scratch.keras` — CNN trained from scratch
- `transfer_model.keras` — trained transfer-learning model
- `results.csv` — model comparison metrics
- `gtzan_cache.npz` — cached features and spectrograms

## 📚 References

- [GTZAN Music Genre Classification Dataset — Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [Librosa documentation](https://librosa.org/doc/latest/index.html)
- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow documentation](https://www.tensorflow.org/learn)
## 👤 Author

**Abdalrhman Attia**

- GitHub: https://github.com/AbdalrhmanFouadOsama2004
- LinkedIn: https://www.linkedin.com/in/abdalrhman-fouad-osama-attia-06147025a/

Replace the placeholders above with your profile details.

---

If you find this project useful, consider giving the repository a ⭐.
