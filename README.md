# Fake-News-Detector
Project implements a Fake News Detection model using Natural Language Processing (NLP) and Machine Learning techniques. The model is trained on a dataset containing fake and real news articles and classifies a given news article as either "Fake News" or "Real News.

## Dataset
The dataset consists of two CSV files:
1. **Fake.csv** - Contains fake news articles labeled as `0`.
2. **True.csv** - Contains real news articles labeled as `1`.

The dataset is combined into a single DataFrame for training and evaluation.

## Preprocessing
To improve model performance, the text undergoes the following preprocessing steps:
- **Lowercasing**: Converts all text to lowercase.
- **Punctuation Removal**: Removes special characters and punctuation marks.
- **Number Removal**: Eliminates numerical digits.
- **Tokenization and Vectorization**: The text is transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

## Model Training
The Naïve Bayes classifier (`MultinomialNB`) is trained using the vectorized text data. The model learns patterns from the training dataset and predicts whether a news article is real or fake.

## Evaluation
The model's performance is assessed using:
- **Accuracy Score**: Measures the percentage of correct predictions.
- **Classification Report**: Displays precision, recall, and F1-score.
- **Confusion Matrix**: Visual representation of prediction errors.

## Prediction Function
The trained model is used to classify new, unseen news articles. The input text undergoes the same preprocessing and is transformed using the TF-IDF vectorizer before being passed to the model for prediction.

## Visualization
A heatmap is generated to display the confusion matrix, showing the model's ability to distinguish between fake and real news.

## Dependencies
Ensure you have the following Python libraries installed:
- pandas
- numpy
- re
- string
- seaborn
- matplotlib
- scikit-learn

To install the required dependencies, use:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn


## Running the Project
1. Load the dataset (`Fake.csv` and `True.csv`).
2. Preprocess the text data.
3. Split the dataset into training and testing sets.
4. Convert text data into numerical vectors using TF-IDF.
5. Train the Naïve Bayes model.
6. Evaluate the model's performance.
7. Use the trained model to predict whether a given news article is real or fake.

## Deployment
To deploy this model as a web application, consider using **Streamlit**. The trained model and vectorizer can be saved using `joblib` and loaded in a Streamlit app to classify user-input news articles in real-time.

## Future Improvements
- Implement deep learning techniques (LSTMs or Transformers) for improved accuracy.
- Enhance text preprocessing with stopword removal and stemming/lemmatization.
- Explore additional machine learning models such as Logistic Regression or Random Forest.
