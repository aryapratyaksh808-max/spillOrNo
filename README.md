# 🛢️ SAR Oil Spill Detection Using Deep Learning

A deep learning project that predicts whether a given **Sentinel-1 Synthetic Aperture Radar (SAR) image contains an oil spill or not**.

The project uses Python, computer vision, and deep learning to classify SAR imagery into oil-spill and non-oil-spill categories.

## 📊 Dataset

The model uses the **Sentinel-1 SAR Oil Spill Detection Dataset** available on Kaggle.

🔗 **Dataset:** [https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset)

You must download the dataset separately. It is not included in this repository because of its size.

## ⚙️ Prerequisites

Before running the project, install:

- Python (compatible with the project's dependencies)
- Git
- pip
- The dataset linked above

## 🚀 Getting Started

### 1. Clone the repository

Open your terminal and run:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset).
2. Sign in to Kaggle.
3. Download the dataset.
4. Extract the downloaded archive.
5. Place the extracted files in the directory expected by the project's code.

**Important:** Check the dataset paths in the notebook or Python scripts before running the project. Update them if your dataset is stored elsewhere.

### 5. Run the project

**If the project uses Jupyter Notebook:**

```bash
pip install notebook
jupyter notebook
```

Open the project's main notebook and execute its cells in order, starting with dataset loading and preprocessing.

**If the project uses a Python script instead**, run the appropriate entry-point file:

```bash
python <your_script_name>.py
```

Replace the placeholder with the actual script name.

## 🧪 How to Test the Project

1. Ensure the dataset has been downloaded and extracted.
2. Verify that the configured dataset paths are correct.
3. Run the preprocessing and model-loading steps.
4. Execute the project's prediction or evaluation section.
5. Check the predicted class for the input SAR image.

The model should return one of the following outcomes, depending on its implemented labels:

- **Oil Spill Detected**
- **No Oil Spill Detected**

The exact prediction format depends on the implementation.

## 📁 Project Structure

A typical project structure may look like this:

```text
project/
├── requirements.txt
├── README.md
├── notebooks/
├── scripts/
├── models/
└── dataset/          # Download separately
```

Your actual repository structure may differ. Keep the dataset in the location expected by the code.

## ⚠️ Important Notes

- Installing `requirements.txt` alone is not sufficient to run the project.
- The dataset must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset).
- **Model training is required:** The PyTorch checkpoint (`.pth` file) containing the trained model weights is not included in this repository.
- After downloading the dataset and configuring the correct paths, run the model's training notebook or training script to train the model and generate the checkpoint.
- Once training is complete, use the saved checkpoint for inference on SAR images, following the prediction code provided in the repository.
- Training time depends on the model architecture, dataset size, and available hardware.
- Predictions are experimental and should not be treated as definitive confirmation of a real-world oil spill.

## 🙌 Acknowledgements

Thanks to the dataset contributors for making Sentinel-1 SAR oil-spill detection data publicly available through Kaggle.

**Dataset source:** [Sentinel-1 SAR Oil Spill Detection Dataset — Kaggle](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset)
