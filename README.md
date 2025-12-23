machine_learning

A lightweight Python project focused on building machine learning tools from scratch and experimenting with data preprocessing, feature engineering, and model design without relying on high-level ML libraries.

The goal of this repository is to better understand how machine learning systems work internally by implementing core components manually using primarily pandas, standard Python libraries, and minimal external dependencies.

Project Motivation

This project is centered around automating parts of the hiring process, specifically estimating whether a CV is relevant for a role.

The idea is to:

extract meaningful features from unstructured data (CV text, images, tabular data)

clean and prepare datasets reliably

train decision trees and random forests from scratch

iteratively improve keyword weighting and feature importance based on real data

Rather than relying on prebuilt ML frameworks, the emphasis is on learning fundamentals, interpretability, and experimentation.

Core Components
1. Keyword-Based CV Scoring (ATS Prototype)

A simple applicant screening system that:

reads CV text files

cleans and tokenizes text

matches words against a keyword list

applies custom weights to calculate a relevance score

filters applications using a configurable threshold

This simulates the behavior of basic Applicant Tracking Systems (ATS).

Future direction:
The keyword weights are currently manually defined. The long-term goal is to learn these weights automatically using the custom random forest implementation in this repository.

2. Decision Trees & Random Forests (From Scratch)

Custom implementations of:

supervised decision trees (entropy / Gini impurity)

unsupervised decision trees (variance-based splits)

random forests with:

bootstrap sampling

feature subsampling

majority voting

Only lightweight dependencies are used (pandas, random, math), allowing full visibility into:

how splits are chosen

how trees grow

how ensembles improve stability

These models are intended to:

analyze feature importance

estimate keyword significance

iteratively update weights as new labeled data is added

3. Data Cleaning & Preprocessing Pipeline (data_clean.py)

A reusable data-cleaning module designed to prepare datasets for machine learning.

Features include:

automatic removal of columns with excessive missing values

user-assisted removal of ID-like columns that may bias models

categorical encoding for non-numeric features

K-Nearest Neighbors (KNN) imputation for missing values

The output is a fully prepared CSV file ready to be used by the tree and forest models.

4. Image Preprocessing Utilities

Basic image handling tools to support future ML experiments:

image loading and resizing

RGB normalization

conversion of images into pandas DataFrames

flattened pixel representations for classical ML models

This allows images to be treated as structured data and integrated into the same pipeline as tabular datasets.

Design Philosophy

Minimal dependencies — avoid black-box ML libraries

Transparency — every algorithm step is visible and modifiable

Modularity — tools can be reused across different data types

Learning-focused — correctness and clarity over optimization

This repository is intended as a learning and experimentation space rather than a production-ready ML system.

Future Work

Planned extensions include:

learning keyword weights automatically using random forests

feature importance analysis across CV datasets

removing interactive steps for full automation

performance evaluation against scikit-learn baselines

improved text preprocessing (TF-IDF, n-grams)

Disclaimer

This project is experimental and educational in nature.
It is not intended to be used as a real hiring system.
