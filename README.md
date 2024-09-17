# Cybersecurity-AI Research Project

## Overview

This repository contains the research work done during my undergraduate program at the University of Dayton Research Institute (UDRI). The project focuses on comparing the accuracy and precision of various AI models in identifying and classifying network packets, primarily using the NSL-KDD dataset. 

The key models explored include:
- **Convolutional Neural Networks (CNN)** using PyTorch.
- **Distributed Random Forest**.
- **Decision Trees**, along with an ensemble of classifiers, including Random Forest, Bagging, AdaBoost, Voting, and Logistic Regression.

## Repository Structure

- **Comparison Plots.ipynb**:
  - This notebook provides visual comparisons of the performance of various models across different metrics, such as accuracy and precision.

- **Distributed Random Forest - New Data.ipynb**:
  - A notebook implementing and fine-tuning the Distributed Random Forest model on the NSL-KDD dataset.

- **Distributed Random Forest.ipynb**:
  - The initial implementation of the Distributed Random Forest model.

- **PyTorch_NSL_KDD_CNN_Start_File_With_Confusion_Matrix.ipynb**:
  - The CNN model implemented in PyTorch, with the results visualized using a confusion matrix.

- **Test_CNN_PyTorch_NSLKDD_5by5_Filters-40E.ipynb**:
  - A CNN model configured with 5x5 filters, trained over 40 epochs on the NSL-KDD dataset.

- **Attack Types.csv**:
  - A CSV file containing different attack types used in the NSL-KDD dataset.

- **ConfusionMatrix.png**:
  - A visual representation of the CNN model's confusion matrix.

- **decisiontree.png**:
  - A diagram of the Decision Tree model used in the study.

- **Field Names.csv**:
  - A list of field names from the NSL-KDD dataset.

- **Group Trial Data.csv**:
  - Contains data collected from trial experiments conducted during the model testing phase.

- **KDDTest+.csv**:
  - A testing dataset derived from NSL-KDD.

- **nsl_kdd_pytorch.csv** and **nsl_kdd_pytorch.mat**:
  - The NSL-KDD dataset used for training and testing the models in both CSV and MAT formats.

- **Predictions_x.csv**:
  - CSV file containing predictions made by the models on test datasets.

- **unique_testing_dataset.csv**:
  - A unique subset of the testing dataset used for additional validation of model performance.

## Models & Libraries Used

- **Decision Tree Models**:
  - The following models were used for the decision tree experiments:
    - DecisionTreeClassifier
    - RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, VotingRegressor
    - LogisticRegression, LinearRegression

- **CNN Model**:
  - Implemented using PyTorch, focused on classifying network packets.

- **Random Forest**:
  - Distributed Random Forest, built using the H2O framework, used for robust classification performance.

## Running the Models

### Setup

1. Clone the repository:

2. Install the necessary libraries.

3. Run the Jupyter notebooks:

   Open any of the notebooks like **PyTorch_NSL_KDD_CNN_Start_File_With_Confusion_Matrix.ipynb** or **Distributed Random Forest - New Data.ipynb** to explore the models.

## Results

- **CNN Model**: Achieved high accuracy in classifying network packets, particularly effective in detecting certain attack types. The confusion matrix visualizes its performance.
  
- **Distributed Random Forest**: Demonstrated strong performance in classifying both normal and attack packets, offering robust accuracy and precision.

- **Decision Tree Models**: Decision Tree and its ensemble counterparts (e.g., Random Forest, AdaBoost) provided a strong baseline comparison against the CNN model.

## Future Work

- Further tuning of hyperparameters for all models.
- Exploring additional datasets for validation.
- Considering other AI architectures such as LSTM or Transformer models for packet classification.

---

Feel free to explore the notebooks for detailed analyses and insights!
