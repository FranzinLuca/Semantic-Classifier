# Semantic Classifier for Cultural Representation

This repository contains the implementation of two distinct machine learning models designed to classify concepts based on their cultural representation. The goal is to determine whether a concept is **Cultural Agnostic**, **Cultural Representative**, or **Cultural Exclusive**.

This project was developed by **Francesco Casacchia** and **Luca Franzin**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Models and Methodology](#models-and-methodology)
  - [Model 1: Traditional ML with Wikidata Properties](#model-1-traditional-ml-with-wikidata-properties)
  - [Model 2: Transformer-based with Wikipedia Summaries](#model-2-transformer-based-with-wikipedia-summaries)
- [Performance Results](#performance-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Models](#running-the-models)
- [License](#license)

---

## Project Overview

The core task of this project is to analyze a given concept (e.g., a person, event, or object) and assign it to one of three categories based on its cultural context:

1.  **Cultural Agnostic:** The concept is universal and not tied to any specific culture.
2.  **Cultural Representative:** The concept is strongly associated with a particular culture but is widely recognized outside of it.
3.  **Cultural Exclusive:** The concept is deeply embedded within a specific culture and is generally unknown or misunderstood outside of it.

To tackle this classification challenge, we developed and evaluated two different models, each leveraging a unique data source and architecture to capture the necessary semantic nuances.

---

## Models and Methodology

We implemented two distinct approaches to compare a traditional machine learning pipeline against a modern, Transformer-based architecture.

### Model 1: Traditional ML with Wikidata Properties

This model uses the structured data available in Wikidata to build a feature set for classification.

-   **Data Source:** Wikidata `Statement-Claim` pairs associated with each concept. For example, the concept "1889 Apia cyclone" might have a pair like `(instance of, cyclone)`.
-   **Feature Engineering:**
    1.  Relevant `Statement-Claim` pairs are extracted for each concept, filtering out overly specific pairs that lack standard Wikidata identifiers (P-codes and Q-codes).
    2.  Dictionaries of all unique statements and claims are created.
    3.  Custom embeddings are generated for each statement and claim.
    4.  For a given concept, the embeddings of its corresponding statement and claim pairs are concatenated.
    5.  These concatenated vectors are averaged to produce a single, final feature vector representing the concept.
-   **Architecture:** The resulting vector is fed into a simple feed-forward network with two dense layers, ReLU activation, and Dropout. The final classification is done using a Softmax output layer.

### Model 2: Transformer-based with Wikipedia Summaries

This model leverages the power of a pre-trained Transformer to understand the unstructured text describing each concept.

-   **Data Source:** The introductory summary of the Wikipedia article for each concept. This provides a rich, descriptive context.
-   **Feature Engineering:** The raw text summaries are tokenized to be compatible with the Transformer model.
-   **Architecture:**
    1.  A pre-trained **`distilroberta-base`** model from the Hugging Face library is used as the base encoder.
    2.  The output from the Transformer is passed to a feed-forward network with three dense layers for the final classification.
    3.  Like Model 1, it uses a Softmax output and is trained with a Categorical Cross-Entropy loss function.

Both models utilized the **Optuna** library for efficient hyperparameter tuning to find the best-performing configurations.

---

## Performance Results

Despite their different approaches, both models achieved remarkably similar performance on the validation set. This suggests that both structured metadata from Wikidata and unstructured text from Wikipedia are viable sources for capturing cultural representation.

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | :------: | :-------: | :----: | :------: |
| **Traditional ML** | `0.7733` | `0.7661`  | `0.7708` | `0.7612`   |
| **Transformer-based** | `0.7600` | `0.7495`  | `0.7542` | `0.7494`   |

---

## Repository Structure\

├── datasets_expanded/  # Contains the processed datasets for each model\
│   ├── all_properties.csv      # Data for Model 1 (Wikidata pairs)\
│   ├── summary.csv             # Data for Model 2 (Wikipedia summaries)\
│   └── ...                     # Train, validation, and test splits\
├── scripts/              # Helper Python scripts for data processing and modeling\
│   ├── dataloader.py         # Custom PyTorch data loaders\
│   ├── modify_dataset.py     # Scripts to process raw data and enrich it\
│   ├── my_model.py           # Model definitions\
│   └── utils.py              # Utility functions\
├── FinalReport.pdf       # The detailed project report\
├── model_1.ipynb         # Jupyter Notebook for training and evaluating the Traditional ML model\
├── model_2.ipynb         # Jupyter Notebook for training and evaluating the Transformer model\
└── README.md             # This file\


---

## Getting Started

Follow these instructions to set up the project environment and run the models.

### Prerequisites

-   Python (version 3.9 or higher recommended)
-   Conda or another virtual environment manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/franzinluca/semantic-classifier.git](https://github.com/franzinluca/semantic-classifier.git)
    cd semantic-classifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    conda create --name semantic_classifier python=3.9
    conda activate semantic_classifier
    ```

3.  **Install the required libraries:**
    (It is recommended to create a `requirements.txt` file from the notebooks for easier installation)
    ```bash
    pip install torch pandas scikit-learn jupyter transformers optuna
    ```

### Running the Models

The entire workflow for training, validation, and testing is contained within the two Jupyter notebooks:

-   To run the **Traditional ML model**, open and execute the cells in `model_1.ipynb`.
-   To run the **Transformer-based model**, open and execute the cells in `model_2.ipynb`.

The notebooks load data from the `datasets_expanded/` directory and use the helper functions and classes defined in the `scripts/` directory.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
