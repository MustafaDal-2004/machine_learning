machine_learning

This repository is a lightweight Python project focused on building machine learning tools from scratch and experimenting with data preprocessing, feature engineering, and model design without relying on high-level machine learning libraries. The goal is to better understand how machine learning systems work internally by manually implementing core components using primarily pandas, standard Python libraries, and minimal external dependencies.

Project Motivation

This project explores the automation of parts of the hiring process, specifically estimating whether a CV is relevant for a given role. It focuses on extracting meaningful features from unstructured data such as CV text, images, and tabular datasets, while also emphasizing reliable data cleaning and preparation. Decision trees and random forests are implemented from scratch to allow full control over how models learn and make decisions. Rather than relying on prebuilt frameworks, the project prioritizes learning fundamentals, interpretability, and experimentation.

Core Components
Keyword-Based CV Scoring (ATS Prototype)

The repository includes a simple applicant screening system that reads CV text files, cleans and tokenizes the text, matches words against a predefined keyword list, and applies custom weights to calculate a relevance score. Applications are filtered using a configurable threshold, simulating the behavior of basic Applicant Tracking Systems (ATS). Currently, keyword weights are manually defined, but the long-term goal is to learn these weights automatically using the custom random forest implementation included in the project.

Decision Trees & Random Forests (From Scratch)

Custom implementations of supervised and unsupervised decision trees are provided, using entropy, Gini impurity, and variance-based splitting strategies. These trees are combined into random forests using bootstrap sampling, feature subsampling, and majority voting. Only lightweight dependencies such as pandas, random, and math are used, allowing full visibility into how splits are selected, how trees are constructed recursively, and how ensemble methods improve stability. These models are intended to analyze feature importance, estimate keyword significance, and iteratively update weights as new labeled data is introduced.

Data Cleaning & Preprocessing Pipeline (data_clean.py)

The project includes a reusable data-cleaning module designed to prepare datasets for machine learning workflows. This pipeline automatically removes columns with excessive missing values, allows user-assisted removal of ID-like columns that may bias models, encodes categorical features, and imputes missing values using a K-Nearest Neighbors (KNN) approach. The result is a fully prepared CSV dataset ready to be used by the decision tree and random forest models.

Image Preprocessing Utilities

Basic image preprocessing utilities are included to support future machine learning experiments involving visual data. These tools handle image loading and resizing, RGB normalization, conversion of images into pandas DataFrames, and flattened pixel representations suitable for classical machine learning models. This allows image data to be treated as structured input and integrated into the same pipeline as tabular datasets.

Design Philosophy

The design of this repository emphasizes minimal dependencies to avoid black-box machine learning libraries, transparency so that every algorithmic step is visible and modifiable, modularity to enable reuse across different data types, and a learning-focused approach that prioritizes correctness and clarity over optimization. The project is intended as a space for experimentation and understanding rather than a production-ready system.

Future Work

Planned extensions include learning keyword weights automatically using random forests, performing feature importance analysis across CV datasets, removing interactive steps to enable full automation, evaluating performance against scikit-learn baselines, and improving text preprocessing through techniques such as TF-IDF and n-grams.

Disclaimer

This project is experimental and educational in nature and is not intended to be used as a real hiring or applicant screening system.
