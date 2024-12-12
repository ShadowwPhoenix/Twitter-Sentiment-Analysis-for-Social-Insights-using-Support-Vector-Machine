# Twitter Sentiment Classification with SVM: Analyzing Public Opinion

## Overview
This project aims to classify the sentiment of tweets using Support Vector Machines (SVM). The goal is to analyze public opinion on Twitter by determining if a tweet is **Positive**, **Negative**, **Neutral**, or **Irrelevant**. The model processes tweet text, applies natural language processing (NLP) techniques, and then uses the SVM algorithm for sentiment prediction.

## Approach

1. **Data Acquisition**:
   - The dataset used in this project is taken from Kaggle's [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) dataset. This dataset contains Twitter text data labeled with sentiment tags such as Positive, Negative, Neutral, and Irrelevant.

2. **Data Preprocessing**:
   - **Handling Missing Values**: Missing text data in both the training and validation sets is filled with a "neutral" label to ensure no data is lost during training.
   - **Text Cleaning**: Text data is cleaned by removing URLs, mentions (@), hashtags (#), and punctuation. The text is then converted to lowercase and tokenized.
   - **Stopwords Removal**: Common words like "the", "is", "in" are removed from the text to avoid skewing the results.

3. **Text Vectorization**:
   - A **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer is used to convert text data into numerical features that can be processed by machine learning models. Only the top 5000 most important features are considered.

4. **Modeling**:
   - **Support Vector Machine (SVM)** with a linear kernel is used to classify the sentiment of the tweets. SVM is a powerful algorithm for classification tasks, especially when the feature space is large.

5. **Evaluation**:
   - The model is evaluated on the **test data** (20% of the training data) and the **validation data** (separate dataset).
   - **Accuracy**, **precision**, **recall**, and **F1-score** are calculated using the `classification_report`.
   - A **confusion matrix** is plotted to visualize how well the model performs.

6. **Prediction on New Text**:
   - A function is provided to classify new input text (e.g., new tweets). This function cleans the text, vectorizes it, and predicts the sentiment using the trained model.

## Sentiment Classes in the Dataset

### 1. **Positive Sentiment**
   - **Description**: Tweets classified as "Positive" express favorable opinions, enthusiasm, happiness, or support. These tweets generally convey optimism or positive emotions about a particular topic, product, person, or event.
   - **Examples**:
     - "I absolutely love this new feature on Twitter! So much fun!"
     - "Great job to the team! You did an amazing job!"
     - "This new movie was fantastic, I really enjoyed it!"
   - **Characterized by**:
     - Positive language, such as "love", "great", "amazing", "fantastic".
     - Expressions of happiness, excitement, or praise.

### 2. **Negative Sentiment**
   - **Description**: Tweets classified as "Negative" express unfavorable opinions, disappointment, anger, frustration, or criticism. These tweets typically convey dissatisfaction or negative emotions about a person, event, or topic.
   - **Examples**:
     - "I can't believe how bad the service was today, very disappointing."
     - "This app is so buggy, I keep crashing every time I try to use it."
     - "The event was a total waste of time, I hated it."
   - **Characterized by**:
     - Negative language, such as "bad", "disappointing", "hate", "worst", "angry".
     - Criticism, complaints, or expressions of frustration or sadness.

### 3. **Neutral Sentiment**
   - **Description**: Tweets classified as "Neutral" are typically factual, descriptive, or objective. These tweets do not express strong positive or negative emotions, but instead provide information, updates, or neutral commentary.
   - **Examples**:
     - "The new version of the app is now available for download."
     - "I just finished reading an interesting article on climate change."
     - "Meeting starts at 3 PM."
   - **Characterized by**:
     - Absence of strong emotional content or opinion.
     - Informative, neutral commentary without any strong positive or negative feelings.

### 4. **Irrelevant Sentiment**
   - **Description**: Tweets classified as "Irrelevant" do not pertain to any sentiment analysis task and might not belong to any of the typical positive, negative, or neutral sentiment categories. They can include random or off-topic text, promotional content, spam, or general chatter that does not provide useful sentiment information.
   - **Examples**:
     - "Check out this link for a great deal on shoes!"
     - "Follow me on Twitter @username."
     - "Win a free iPhone! Click here."
     - "I had lunch at a great place today, check it out!"
   - **Characterized by**:
     - Promotional, spam, or self-promotional content.
     - Irrelevant to the subject being analyzed (for example, the subject of the tweet does not relate to opinions about a product, service, or event).
     - Includes unrelated hashtags, emojis, or generic statements not contributing to sentiment.

## Why These Categories Matter
These sentiment classes are important for the following reasons:
- **Understanding Public Opinion**: By classifying tweets into sentiment categories, we can gain insights into how people feel about various topics, such as a brand, product, service, or event.
- **Improved Decision-Making**: Businesses, marketers, and organizations can use sentiment analysis to monitor public opinion, measure customer satisfaction, and respond to feedback effectively.
- **Fine-grained Analysis**: The inclusion of an "Irrelevant" class allows for better control over the analysis, ensuring that only relevant sentiment data is used for analysis and decision-making.

## Installation

To run this project, you will need to install the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- nltk

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk
