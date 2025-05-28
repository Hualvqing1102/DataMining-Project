# Twitter Sentiment Analysis Project

## Project Overview

This project aims to analyze Twitter data and classify tweets based on sentiment using machine learning techniques. It focuses on cleaning raw tweet text, transforming it with TF-IDF vectorization, and training Support Vector Machine (SVM) models to predict sentiment labels. The goal is to develop an effective sentiment classification pipeline that can be used to understand public opinions from Twitter data.

---

## Business Understanding

### What is Sentiment Analysis?

Sentiment Analysis is the computational process of identifying and categorizing opinions expressed in text, especially to determine whether the writer's attitude is positive, negative, or neutral. Twitter sentiment analysis is widely used in brand monitoring, market research, and social media trend tracking to gain insights into customer attitudes and public mood.

---

## Dataset Description

The dataset contains tweets related to airline experiences. Each tweet is labeled with sentiment classes such as positive, neutral, or negative. The data includes original tweet text along with metadata such as airline sentiment labels.

- **Source:** Public Twitter dataset for airline sentiment analysis  
- **File format:** CSV  
- **Key columns:** `text` (tweet content), `airline_sentiment` (label)  

---

## Directory Structure

├── Data
│ ├──archive.zip # Original dataset zip
│ ├── Tweets.csv # Original dataset
│ ├── cleaned_tweets.csv # Preprocessed dataset after cleaning
│ ├──DataWash.py
│ ├──FeatureExtract.py
├── Picture 
# Visualization of model performance
│ ├── Class Distribution in Dataset.png
│ ├──Classification Metrics by Class.png
│ ├── Confusion Matrix.png 
├──Presentation
│ ├──Twitter_Sentiment_Analysis_Presentation.pptx
├── README.md
├── Twitter_sentiment_classification_modeling.ipynb # SVM and Visualization
├──Twitter_sentiment_exploratory_data_analysis.ipynb #Data cleaning and saving