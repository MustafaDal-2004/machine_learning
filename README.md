machine_learning

A lightweight Python project focused on building machine learning tools from scratch and experimenting with data preprocessing, feature engineering, and model design without relying on high-level machine learning libraries.

The goal of this repository is to better understand how machine learning systems work internally by manually implementing core components using primarily pandas, standard Python libraries, and minimal external dependencies.

Project Motivation

This project explores the automation of parts of the hiring process, specifically estimating whether a CV is relevant for a given role.

The objectives include:

Extracting meaningful features from unstructured data (CV text, images, and tabular data)

Cleaning and preparing datasets in a reliable and reusable way

Implementing decision trees and random forests from scratch

Iteratively improving keyword weighting and feature importance using real data

Rather than relying on prebuilt ML frameworks, the emphasis is on learning fundamentals, interpretability, and experimentation.

Core Components
Keyword-Based CV Scoring (ATS Prototype)

A simple applicant screening system that:

Reads CV text files

Cleans and tokenizes text

Matches words against a predefined keyword list

Applies custom weights to calculate a relevance score

Filters applications using a configurable threshold

This component simulates the behavior of basic Applicant Tracking Systems (ATS).

Future direction:
Keyword weights are currently manually defined. The long-term goal is to learn these weights automatically using the custom random forest implementation included in this repository.

Decision Trees & Random Forests (From Scratch)

Custom implementations of:

Supervised decision trees (Entropy / Gini impurity)

Unsupervised decision trees (variance-based splitting)

Random forests using:

Bootstrap sampling

Feature subsampling

Majority voting

Only lightweight dependencies are used (pandas, random, math), providing full visibility into:

How splits are selected

How trees are constructed recursively

How ensemble methods improve stability and performance

These models are designed to:

Analyze feature importance

Estimate keyword significance

Iteratively update weights as new labeled data is added

Data Cleaning & Preprocessing Pipeline (data_clean.py)

A reusable data-cleaning module designed to prepare datasets for machine learning workflows.

Features include:

Automatic removal of columns with excessive missing values

User-assisted removal of ID-like columns that may bias models

Categorical encoding for non-numeric features

K-Nearest Neighbors (KNN) imputation for missing values

The output is a fully prepared CSV dataset ready for use with the tree and forest models.

Image Preprocessing Utilities

Basic image-handling tools to support future ML experiments:

Image loading and resizing

RGB normalization

Conversion of images into pandas DataFrames

Flattened pixel representations for classical ML models

These utilities allow image data to be treated as structured input and integrated into the same pipeline as tabular datasets.

Design Philosophy

Minimal dependencies — avoid black-box ML libraries

Transparency — every algorithmic step is visible and modifiable

Modularity — components can be reused across different data types

Learning-focused — correctness and clarity prioritized over optimization

This repository is intended as a learning and experimentation space, not a production-ready machine learning system.

Future Work

Planned extensions include:

Learning keyword weights automatically using random forests

Feature importance analysis across CV datasets

Removing interactive steps to enable full automation

Performance evaluation against scikit-learn baselines

Enhanced text preprocessing (TF-IDF, n-grams)

Disclaimer

This project is experimental and educational in nature.
It is not intended to be used as a real hiring or applicant screening system.
