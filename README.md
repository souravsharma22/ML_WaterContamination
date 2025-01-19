# Water Contamination Type Classification

## Overview
This repository contains a machine learning project aimed at classifying water contamination types using image data. The goal is to identify whether a water body is contaminated with:

- **Algae**
- **Plastic**
- **Oil Spill**
- Or if the water body is **clean**

The classification is achieved using a **Support Vector Machine (SVM)** model, which has been fine-tuned to improve performance.

## Project Features
- **Data Preprocessing:** Techniques to clean and prepare image data for model training.
- **Model Training:** Implementation of an SVM-based classifier for image classification.
- **Model Tuning:** Hyperparameter optimization to enhance the model's accuracy and generalization.
- **Evaluation:** Performance evaluation using metrics such as accuracy, precision, recall, and F1-score.

## Getting Started

### Prerequisites
To run this project, you will need the following installed:
- Python (>=3.8)
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `opencv-python`, `matplotlib`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

### Dataset
Ensure you have access to a dataset containing labeled images of the following categories:
1. Algae
2. Plastic
3. Oil Spill
4. Clean Water

Organize your dataset in a directory structure similar to:
```
/dataset
  /train
    /algae
    /plastic
    /oilspill
    /clean
  /test
    /algae
    /plastic
    /oilspill
    /clean
```
## Project Structure
```
ML/
|-- data/
|   |-- dataset        # Image folder as shown above 
|   |-- Mydata.pkl    # Cleaned and preprocessed data stored in pkl format
|
|-- notebooks/
|   |-- dataPrep.ipynb             # Exploratory Data Analysis notebook
|   |-- eda.ipynb   
|   |-- prediction.ipynb  # Model training and evaluation notebook
|   |-- hypertunning.ipynb   # tunning for better accuracy
|
|-- models/
|   |-- model.pkl              # Serialized model
|   |-- tunnedmodel.pkl              # tunned model
|-- app.py                    # web app for deployment
|
|-- README.md                  # Project documentation
```

---

### Running the Project
1. Clone this repository:
```bash
git clone https://github.com/souravsharma22/ML_WaterContamination.git
```
2. Navigate to the project directory:
```bash
cd ML_WaterContamination
```
3. Train and evaluate the model:
```bash
open jupiter NOtebook
```

## Results
The model achieves high accuracy in classifying water contamination types. Detailed results, including confusion matrices and classification reports, are provided in the `results/` directory.

## Future Scope
- Integration of deep learning models (e.g., CNNs) for improved accuracy.
- Deployment of the model as a web application for real-time classification.
- Incorporation of additional contamination categories.

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request to suggest improvements or report bugs.

## Contributor
Sourav Sharma
souravbgp2210@gmail.com
---


