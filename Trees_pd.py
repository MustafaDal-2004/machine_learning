import pandas as pd
import random
from collections import Counter
import math

# ============================================================
# Unsupervised Decision Trees
# ============================================================

def split_numeric(values, threshold):
    """
    Splits numeric values into two groups based on a threshold.
    """
    left = [v for v in values if v >= threshold]
    right = [v for v in values if v < threshold]
    return left, right


def variance(values):
    """
    Computes variance of a list of values.
    Used as an impurity measure for unsupervised splits.
    """
    if len(values) == 0:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def best_split_numeric_unsupervised(column):
    """
    Finds the best numeric split by minimizing total variance.
    """
    best_threshold = None
    lowest_variance = float("inf")

    for threshold in sorted(column.unique()):
        left, right = split_numeric(column, threshold)
        total_variance = variance(left) + variance(right)

        if total_variance < lowest_variance:
            lowest_variance = total_variance
            best_threshold = threshold

    return best_threshold, lowest_variance


def best_split_categorical_unsupervised(column):
    """
    Finds a one-vs-rest categorical split that balances group sizes.
    """
    categories = column.unique()
    if len(categories) <= 1:
        return None, float("inf")

    best_split = None
    best_score = float("inf")

    for category in categories:
        group_a = column == category
        group_b = ~group_a
        score = abs(group_a.sum() - group_b.sum())

        if score < best_score:
            best_score = score
            best_split = ({category}, set(categories) - {category})

    return best_split, best_score


def build_unsupervised_tree(data, max_depth, depth=0):
    """
    Recursively builds an unsupervised decision tree.
    """
    if depth >= max_depth or len(data) == 0:
        return {"leaf": True, "indices": data.index.tolist()}

    best_column = None
    best_split = None
    best_score = float("inf")
    split_type = None

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            threshold, score = best_split_numeric_unsupervised(data[column])
            if score < best_score:
                best_score = score
                best_split = threshold
                best_column = column
                split_type = "numeric"
        else:
            split, score = best_split_categorical_unsupervised(data[column])
            if score < best_score:
                best_score = score
                best_split = split
                best_column = column
                split_type = "categorical"

    if best_column is None:
        return {"leaf": True, "indices": data.index.tolist()}

    if split_type == "numeric":
        left_data = data[data[best_column] >= best_split]
        right_data = data[data[best_column] < best_split]
    else:
        left_data = data[data[best_column].isin(best_split[0])]
        right_data = data[~data[best_column].isin(best_split[0])]

    return {
        "leaf": False,
        "column": best_column,
        "split_type": split_type,
        "split_value": best_split,
        "left": build_unsupervised_tree(left_data, max_depth, depth + 1),
        "right": build_unsupervised_tree(right_data, max_depth, depth + 1),
    }


def build_unsupervised_random_forest(data, n_trees, depth_range):
    """
    Builds an ensemble of unsupervised decision trees.
    """
    forest = []
    shuffled = data.sample(frac=1).reset_index(drop=True)
    chunk_size = len(shuffled) // n_trees

    for i in range(n_trees):
        subset = shuffled.iloc[i * chunk_size :] if i == n_trees - 1 \
            else shuffled.iloc[i * chunk_size : (i + 1) * chunk_size]

        depth = random.randint(depth_range[0], depth_range[1])
        tree = build_unsupervised_tree(subset, depth)
        forest.append(tree)

    return forest


# ============================================================
# Supervised Decision Trees
# ============================================================

def entropy(y):
    """
    Computes Shannon entropy.
    """
    total = len(y)
    counts = Counter(y)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def gini(y):
    """
    Computes Gini impurity.
    """
    total = len(y)
    counts = Counter(y)
    return 1 - sum((c / total) ** 2 for c in counts.values())


def best_split_numeric_supervised(column, y, criterion):
    """
    Finds the best numeric split using entropy or Gini impurity.
    """
    impurity = entropy if criterion == "entropy" else gini
    best_threshold = None
    best_score = float("inf")

    for threshold in sorted(column.unique()):
        left = y[column >= threshold]
        right = y[column < threshold]

        score = (
            len(left) * impurity(left) +
            len(right) * impurity(right)
        ) / len(y)

        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def best_split_categorical_supervised(column, y, criterion):
    """
    Finds the best categorical split using one-vs-rest strategy.
    """
    impurity = entropy if criterion == "entropy" else gini
    categories = column.unique()

    best_split = None
    best_score = float("inf")

    for category in categories:
        left = y[column == category]
        right = y[column != category]

        score = (
            len(left) * impurity(left) +
            len(right) * impurity(right)
        ) / len(y)

        if score < best_score:
            best_score = score
            best_split = ({category}, set(categories) - {category})

    return best_split, best_score


def build_supervised_tree(X, y, max_depth, depth=0, criterion="entropy"):
    """
    Recursively builds a supervised decision tree.
    """
    if depth >= max_depth or len(set(y)) == 1 or len(X) == 0:
        return {
            "leaf": True,
            "prediction": y.mode()[0],
            "samples": len(y)
        }

    best_column = None
    best_split = None
    best_score = float("inf")
    split_type = None

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            threshold, score = best_split_numeric_supervised(X[column], y, criterion)
            if score < best_score:
                best_score = score
                best_split = threshold
                best_column = column
                split_type = "numeric"
        else:
            split, score = best_split_categorical_supervised(X[column], y, criterion)
            if score < best_score:
                best_score = score
                best_split = split
                best_column = column
                split_type = "categorical"

    if split_type == "numeric":
        left_mask = X[best_column] >= best_split
        right_mask = X[best_column] < best_split
    else:
        left_mask = X[best_column].isin(best_split[0])
        right_mask = ~left_mask

    return {
        "leaf": False,
        "column": best_column,
        "split_type": split_type,
        "split_value": best_split,
        "left": build_supervised_tree(X[left_mask], y[left_mask], max_depth, depth + 1, criterion),
        "right": build_supervised_tree(X[right_mask], y[right_mask], max_depth, depth + 1, criterion),
        "samples": len(y)
    }


def build_supervised_random_forest(
    X, y, n_trees=10, max_depth=5, sample_ratio=0.8, feature_ratio=0.8, criterion="entropy"
):
    """
    Builds a random forest using bootstrap sampling and feature subsampling.
    """
    forest = []

    for _ in range(n_trees):
        indices = random.sample(range(len(X)), int(len(X) * sample_ratio))
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]

        n_features = max(1, int(len(X.columns) * feature_ratio))
        features = random.sample(list(X.columns), n_features)

        tree = build_supervised_tree(X_sample[features], y_sample, max_depth, criterion=criterion)
        forest.append((tree, features))

    return forest


def predict_tree(tree, row):
    """
    Predicts a single sample using a decision tree.
    """
    if tree["leaf"]:
        return tree["prediction"]

    if tree["split_type"] == "numeric":
        branch = "left" if row[tree["column"]] >= tree["split_value"] else "right"
    else:
        branch = "left" if row[tree["column"]] in tree["split_value"][0] else "right"

    return predict_tree(tree[branch], row)


def predict_forest(forest, X):
    """
    Predicts labels using majority voting across trees.
    """
    predictions = []

    for _, row in X.iterrows():
        votes = [
            predict_tree(tree, row[features])
            for tree, features in forest
        ]
        predictions.append(Counter(votes).most_common(1)[0][0])

    return predictions




