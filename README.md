# Hakuhodo_ChatGPT Multimodal Behavior and Text Analysis Pipeline

## Overview

This notebook implements an experimental pipeline that integrates:

* Speech recognition from video
* Japanese text processing
* Step counter (pedometer) data
* Weather and time features
* Machine learning prediction
* Text similarity analysis

The purpose of this project is to examine how **spoken content, written ideas, environmental context, and human activity (step counts)** can be transformed into structured data and combined into a single analytical workflow.

---

## Input Data

The system uses three main inputs:

1. **Text file**
   A user-provided text file representing an idea or topic of interest.

2. **Video file (MP4)**
   A video containing spoken Japanese, used as the source for speech transcription.

3. **CSV dataset**
   A dataset containing:

   * month
   * hour
   * rain
   * maxTemp
   * steps (pedometer data)

The step counter data represents human walking activity and is used as the prediction target for the machine learning model.

---

## Processing Flow

### 1. Text Preprocessing

The uploaded text file is processed using Janome for Japanese morphological analysis.
The text is tokenized into word units and rewritten with whitespace to allow vectorization and similarity comparison.

---

### 2. Speech Transcription

Audio is extracted from the uploaded MP4 file and transcribed using OpenAI Whisper.
The transcription is optionally refined using GPT and stored as text.
This text is also tokenized using Janome for further analysis.

---

### 3. Machine Learning Model

A RandomForestClassifier is trained using the following features:

* month
* hour
* rain
* maxTemp

The target variable is:

* steps (pedometer data)

The model learns the relationship between **time and weather conditions** and **walking behavior**.

Model performance is evaluated using:

* Accuracy score
* Classification report

GPT is used to generate a short natural-language interpretation of the model’s performance.

---

### 4. Behavioral Prediction

Using the trained model and the hour extracted from speech transcription, the system predicts step-related behavior and infers whether the user is likely to be at home or active.

---

### 5. Text Similarity Analysis

To compare:

* the original idea text
* the transcribed spoken text

TF-IDF vectors are computed and cosine similarity is calculated.
This produces a similarity score representing how closely the spoken content matches the intended idea.

---

## Outputs

The notebook produces:

* Predicted walking behavior (based on step count data)
* At-home status (derived from prediction)
* Model accuracy
* Classification report summary
* Speech transcription
* Tokenized Japanese text
* Similarity score between idea text and spoken content

---

## Technologies Used

* OpenAI Whisper (speech recognition)
* OpenAI GPT (text interpretation)
* Janome (Japanese morphological analysis)
* scikit-learn (Random Forest, TF-IDF, cosine similarity)
* pandas, NumPy
* Google Colab

---

## Concept

This project explores whether **language and physical activity** can be modeled together by combining:

* what is written
* what is spoken
* what the weather is
* how much the user walks

The pipeline treats human behavior as a multimodal signal composed of language, time, environment, and movement.

---

## Intended Use

This notebook is designed for experimental and educational purposes, including:

* Speech-based text analysis
* Japanese NLP experiments
* Activity prediction using step counter data
* Comparing spoken narratives with written ideas
* Demonstrating hybrid NLP and machine learning workflows
