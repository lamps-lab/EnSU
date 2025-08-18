# Ensemble OADS Classifier

This repository provides an implementation of an ensemble learning framework for classification tasks. Below are the instructions to set up the environment, download the saved model weights, and run the framework.

---

## **Setup Instructions**

### 1. **Download the Saved Model Weights**
The saved model weights are stored on Google Drive. Follow these steps to download and set them up:

1. Download the weights from [Google Drive Link](https://drive.google.com/drive/folders/1bWQhAPrGmcEZF86CcyKaHbLPBZTfikOz?usp=sharing).
2. Create a directory named `saved_weights` in the root folder of this project.
3. Place the downloaded weights inside the `saved_weights` folder.

---

### 2. **Install Required Dependencies**
To avoid dependency conflicts, we recommend setting up a Python virtual environment. Follow these steps:

#### 2.1 **Create and Activate the Virtual Environment**  
```bash
conda create -n myenv python=3.7
conda activate myenv
pip install pytorch-ignite==0.4.2
pip install dgl==0.6.0 -f https://data.dgl.ai/wheels-test/repo.html
pip install numpy==1.17.2
pip install torch==1.11.0
pip install transformers==4.19.2
pip install nltk==3.4.5
pip install scikit-learn==0.22
pip install numpy==1.17.2 torch==1.11.0 transformers==4.19.2
python ensemble_learning.py

