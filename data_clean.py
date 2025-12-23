import pandas as pd
import math

# ============================================================
# Column Handling
# ============================================================

def drop_mostly_empty_columns(df, threshold=0.5):
    """
    Drops columns where the fraction of missing values exceeds the threshold.

    Parameters:
        df (pd.DataFrame): Input dataset
        threshold (float): Maximum allowed fraction of missing values

    Returns:
        pd.DataFrame: Dataset with sparse columns removed
    """
    missing_ratio = df.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=columns_to_drop)


def select_columns_to_remove(df):
    """
    Allows the user to interactively select columns to remove
    (e.g., ID or index-like columns).

    Returns:
        list[str]: List of column names to remove
    """
    print("\nAvailable columns:")
    for col in df.columns:
        print(f" - {col}")

    user_input = input("\nEnter comma-separated column names to remove: ")
    selected = [c.strip() for c in user_input.split(",") if c.strip() in df.columns]

    print(f"\nColumns selected for removal: {selected}")
    return selected


# ============================================================
# Encoding
# ============================================================

def encode_categorical_columns(df):
    """
    Encodes non-numeric columns using category encoding.

    Parameters:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Dataset with categorical columns encoded
    """
    df = df.copy()

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Encoding categorical column: {col}")
            df[col] = df[col].astype("category").cat.codes
        else:
            print(f"Column '{col}' is numeric. Skipping encoding.")

    return df


# ============================================================
# Distance & Imputation
# ============================================================

def euclidean_distance(row_a, row_b):
    """
    Computes Euclidean distance between two rows,
    ignoring missing values.
    """
    squared_diffs = []

    for col in row_a.index:
        if pd.notnull(row_a[col]) and pd.notnull(row_b[col]):
            squared_diffs.append((row_a[col] - row_b[col]) ** 2)

    return math.sqrt(sum(squared_diffs)) if squared_diffs else float("inf")


def knn_impute(df, k=3):
    """
    Imputes missing values using K-Nearest Neighbors (KNN).

    Parameters:
        df (pd.DataFrame): Input dataset
        k (int): Number of nearest neighbors

    Returns:
        pd.DataFrame: Dataset with missing values imputed
    """
    df = df.copy()

    for column in df.columns:
        if df[column].isnull().sum() == 0:
            continue

        print(f"\nImputing missing values in column: {column}")

        for idx in df[df[column].isnull()].index:
            target_row = df.loc[idx]
            candidates = df[df[column].notnull()].copy()

            candidates["distance"] = candidates.drop(columns=[column]).apply(
                lambda row: euclidean_distance(
                    target_row.drop(labels=[column]),
                    row
                ),
                axis=1
            )

            nearest_values = candidates.nsmallest(k, "distance")[column]
            imputed_value = nearest_values.mean()

            df.at[idx, column] = imputed_value
            print(f" - Row {idx} imputed with value {imputed_value}")

    return df


# ============================================================
# Main Cleaning Pipeline
# ============================================================

def clean_dataset(df, missing_threshold=0.5, knn_k=3):
    """
    Full data cleaning pipeline:
    - Drops sparse columns
    - Removes user-selected columns
    - Encodes categorical variables
    - Imputes missing values using KNN

    Parameters:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\nInitial data preview:")
    print(df.head())

    columns_to_remove = select_columns_to_remove(df)

    df = drop_mostly_empty_columns(df, threshold=missing_threshold)
    df = encode_categorical_columns(df)
    df = df.drop(columns=columns_to_remove, errors="ignore")
    df = knn_impute(df, k=knn_k)

    return df

