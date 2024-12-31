# Forum Discussions Categorization 

This project focuses on categorizing forum discussions into different categories using various deep learning models. The goal is to classify text data into one of five predefined categories by leveraging different embedding techniques and neural network architectures.

## Project Overview

The project involves several steps, including data preparation, model training, and evaluation. We experimented with multiple deep learning models, each utilizing different embedding techniques and neural network architectures to classify forum discussions.

## Data Preparation

### Steps:
1. **Data Loading and Cleaning**:
   - Removed null values.
   - Handled contractions and abbreviations.
   - Converted text to lowercase and removed punctuations.
   - Dropped texts with length smaller than 20.
   - Replaced mismatched rows with "Media" to handle data imbalance.
   - Replaced numeric rows with the mode of the category.
   - Dropped duplicate rows.
   - Encoded categories using a category mapping dictionary.
   - Applied lemmatization and removed stop words using `preprocess_text`.

2. **Final Data Shape**:
   - After cleaning, the dataset contains **22,707 rows**.

## Models

### 1. Word2Vec + Dense
- **Description**: Uses custom-trained Word2Vec embeddings to transform text into numerical vectors, followed by dense layers for classification.
- **Results**:
  - Validation Accuracy: **67.26%**
  - Training Accuracy: **69.12%**

### 2. Glove Embedding + GRU & LSTM
- **Description**: Utilizes pretrained GloVe embeddings with bidirectional LSTM and GRU layers for sequence processing.
- **Results**:
  - Validation Accuracy: **70.67%**
  - Training Accuracy: **74.59%**

### 3. BERT Feature Extraction + Dense
- **Description**: Leverages pretrained BERT (bert-base-uncased) to generate contextual embeddings, followed by a feed-forward network for classification.
- **Results**:
  - Validation Accuracy: **69.96%**
  - Training Accuracy: **73.85%**

### 4. Embedding all-mpnet-base-v2 + Dense
- **Description**: Uses the **all-mpnet-base-v2** SentenceTransformer to generate embeddings, followed by a feed-forward network with L2 regularization.
- **Results**:
  - Validation Accuracy: **76.13%**
  - Training Accuracy: **77.59%**

### 5. Transformer Model + SentenceTransformer
- **Description**: Combines SentenceTransformer embeddings with handcrafted features (text length, word count, sentence count, average word length) and applies a self-attention block for classification.
- **Results**:
  - Validation Accuracy: **76.86%**
  - Training Accuracy: **80.96%**

## Conclusion

- **Word2Vec + Dense**: Simple model with moderate performance, but room for improvement.
- **Glove Embedding + GRU & LSTM**: Improved accuracy with recurrent layers, but still a gap between training and validation performance.
- **BERT Feature Extraction + Dense**: Utilizes deep contextual embeddings, but may require fine-tuning for better generalization.
- **Embedding all-mpnet-base-v2 + Dense**: Best-performing model with high validation accuracy and excellent generalization.
- **Transformer Model + SentenceTransformer**: Competitive performance with attention mechanisms, but slightly more complex.
- The **Embedding all-mpnet-base-v2 + Dense** model stands out as the most practical and effective choice, offering a balance between simplicity and performance.


