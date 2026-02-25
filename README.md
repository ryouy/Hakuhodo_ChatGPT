# Hakuhodo_ChatGPT

Personalized TV Program View Prediction
Behavior Data × Interest Vector × CM Content Matching

---

## Overview

This project predicts whether a user will watch a specific TV program by combining:

* Physical state estimation (At-home prediction)
* Personal interest modeling (Google search history)
* CM content semantic similarity

Instead of relying on aggregate ratings, this system operates at the individual level.

The core idea:

```
Watch Probability = f(At-Home Probability, Interest Match Score)
```

---

## Data Sources

### 1. Smartphone Step Counter Data

Source: Personal smartphone pedometer logs

Purpose:
Used to determine whether the user is at home.

Definition:

```
steps == 0 → at home
```

Features:

* date
* hour
* steps

---

### 2. Japan Meteorological Agency Weather Data

Source: Official historical weather datasets

Features:

* month
* hour
* rain
* maxTemp

Merged with step data to improve at-home prediction accuracy.

---

### 3. Google Search History (~100,000 entries)

Source: Personal Google search archive export

Used to construct a high-dimensional interest vector.

Process:

* Text normalization
* Tokenization
* TF-IDF vectorization

Output:
Personal Interest Embedding

---

### 4. TV Program CM (Video Input)

Input: CM video (mp4)

Pipeline:

1. Whisper → speech-to-text
2. GPT → cleanup / extraction
3. Tokenization
4. TF-IDF vectorization

Output:
CM Content Embedding

---

## System Architecture

### Step 1 — At-Home Prediction

Model:

```
RandomForestClassifier
```

Input:

* month
* hour
* rain
* maxTemp

Output:

```
P(at_home)
```

Accuracy:
~0.72

---

### Step 2 — Interest Matching

Compute cosine similarity between:

```
Personal Interest Vector
CM Content Vector
```

Output:

```
Interest Match Score
```

---

### Step 3 — Watch Prediction

Final features:

* At-home probability
* Interest similarity score
* Contextual variables (time, weather)

Output:

```
Watch / Not Watch
```

This integrates physical state and psychological preference into a single predictive model.

---

## Tech Stack

* Python
* pandas
* scikit-learn
* Janome
* OpenAI Whisper
* OpenAI GPT API
* TF-IDF
* Cosine Similarity

---

## Why This Is Different

Traditional TV analytics:

* Demographic-based
* Aggregate-based

This system:

* Individual-level
* Behavior-driven
* Semantic content aware

It connects:
Real-world behavior
Personal intent
Media content

Into one prediction pipeline.

---

## Potential Extensions

* Real-time smart TV integration
* Streaming platform adaptation
* Dynamic ad targeting
* LLM-based interest modeling
* Deep learning embedding replacement (BERT / Sentence Transformers)
