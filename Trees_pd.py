import pandas as pd
import random
from collections import Counter
import math

#unsupervised trees
def splitter(data, value):
    left = [item for item in data if item >= value]
    right = [item for item in data if item < value]
    return left, right

def variance(values):
    if len(values) == 0:
        return 0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

def best_split_numeric(data_column):
    smallest_total = float('inf')
    best_split = None

    for value in sorted(data_column.unique()):
        left, right = splitter(data_column, value)
        total_variance = variance(left) + variance(right)

        if total_variance < smallest_total:
            smallest_total = total_variance
            best_split = value

    return best_split, smallest_total

def best_categorical_split(column):
    categories = column.unique()
    if len(categories) <= 1:
        return None, float('inf')

    best_split = None
    best_score = float('inf')

    for category in categories:
        group1 = column == category
        group2 = column != category

        score = abs(group1.sum() - group2.sum())

        if score < best_score:
            best_score = score
            best_split = ({category}, set(categories) - {category})

    return best_split, best_score

def build_unsupervised_tree(data, depth, current_depth=0):
    if current_depth >= depth:
        return {'leaf': data.index.tolist()}

    best_column = None
    best_score = float('inf')
    best_split = None
    split_type = None 

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            value, score = best_split_numeric(data[column])
            if score < best_score:
                best_score = score
                best_split = value
                best_column = column
                split_type = 'numeric'
        else:
            split_groups, score = best_categorical_split(data[column])
            if score < best_score:
                best_score = score
                best_split = split_groups
                best_column = column
                split_type = 'categorical'

    if best_column is None:
        return {'leaf': data.index.tolist()}

    if split_type == 'numeric':
        left_data = data[data[best_column] >= best_split]
        right_data = data[data[best_column] < best_split]
    else:
        left_data = data[data[best_column].isin(best_split[0])]
        right_data = data[~data[best_column].isin(best_split[0])]

    return {
        'split_type': split_type,
        'column': best_column,
        'split_value': best_split,
        'left': build_unsupervised_tree(left_data, depth, current_depth + 1),
        'right': build_unsupervised_tree(right_data, depth, current_depth + 1)
    }

def build_unsupervised_random_forest(data, n_trees, depth_range):
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    part_size = len(data_shuffled) // n_trees
    forest = []

    for i in range(n_trees):
        if i == n_trees - 1:  
            data_part = data_shuffled.iloc[i * part_size :]
        else:
            data_part = data_shuffled.iloc[i * part_size : (i + 1) * part_size]

        random_depth = random.randint(depth_range[0], depth_range[1])
        tree = build_unsupervised_tree(data_part, random_depth)
        forest.append((data_part, tree))

    return forest

#supervised trees
def entropy(y):
    total = len(y)
    counts = Counter(y)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

def best_split_numeric_supervised(X_column, y, criterion='entropy'):
    best_score = float('inf')
    best_value = None

    impurity_func = entropy if criterion == 'entropy' else gini

    for value in sorted(X_column.unique()):
        left_mask = X_column >= value
        right_mask = X_column < value

        y_left = y[left_mask]
        y_right = y[right_mask]

        score = (len(y_left) * impurity_func(y_left) + len(y_right) * impurity_func(y_right)) / len(y)

        if score < best_score:
            best_score = score
            best_value = value

    return best_value, best_score

def best_categorical_split_supervised(X_column, y, criterion='entropy'):
    categories = X_column.unique()
    best_score = float('inf')
    best_split = None

    impurity_func = entropy if criterion == 'entropy' else gini

    for category in categories:
        group1_mask = X_column == category
        group2_mask = ~group1_mask

        y1 = y[group1_mask]
        y2 = y[group2_mask]

        score = (len(y1) * impurity_func(y1) + len(y2) * impurity_func(y2)) / len(y)

        if score < best_score:
            best_score = score
            best_split = ({category}, set(categories) - {category})

    return best_split, best_score

def is_pure(y):
    return len(set(y)) == 1

def build_supervised_tree(X, y, max_depth, current_depth=0, criterion='entropy'):
    if current_depth >= max_depth or is_pure(y) or len(X) == 0:
        return {'leaf': True, 'prediction': y.mode()[0], 'samples': len(y)}

    best_column = None
    best_score = float('inf')
    best_split = None
    split_type = None

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            value, score = best_split_numeric_supervised(X[column], y, criterion)
            if score < best_score:
                best_score = score
                best_split = value
                best_column = column
                split_type = 'numeric'
        else:
            value, score = best_categorical_split_supervised(X[column], y, criterion)
            if score < best_score:
                best_score = score
                best_split = value
                best_column = column
                split_type = 'categorical'

    if best_column is None:
        return {'leaf': True, 'prediction': y.mode()[0], 'samples': len(y)}

    if split_type == 'numeric':
        left_mask = X[best_column] >= best_split
        right_mask = X[best_column] < best_split
    else:
        left_mask = X[best_column].isin(best_split[0])
        right_mask = ~X[best_column].isin(best_split[0])

    left_tree = build_supervised_tree(X[left_mask], y[left_mask], max_depth, current_depth + 1, criterion)
    right_tree = build_supervised_tree(X[right_mask], y[right_mask], max_depth, current_depth + 1, criterion)

    return {
        'leaf': False,
        'split_type': split_type,
        'column': best_column,
        'split_value': best_split,
        'left': left_tree,
        'right': right_tree,
        'samples': len(y)
    }

def build_supervised_random_forest(X, y, n_trees=10, max_depth=5, sample_ratio=0.8, feature_ratio=0.8, criterion='entropy'):
    forest = []

    for _ in range(n_trees):
        # Bootstrap sample (random subset of rows)
        sample_indices = random.sample(range(len(X)), int(len(X) * sample_ratio))
        X_sample = X.iloc[sample_indices].reset_index(drop=True)
        y_sample = y.iloc[sample_indices].reset_index(drop=True)

        # Random subset of features
        n_features = max(1, int(len(X.columns) * feature_ratio))
        feature_subset = random.sample(list(X.columns), n_features)

        # Train a tree on the subset
        tree = build_supervised_tree(X_sample[feature_subset], y_sample, max_depth=max_depth, criterion=criterion, feature_subset=feature_subset)
        forest.append((tree, feature_subset))

    return forest

def predict_with_tree(tree, row):
    if tree['leaf']:
        return tree['prediction']

    col = tree['column']
    if tree['split_type'] == 'numeric':
        if row[col] >= tree['split_value']:
            return predict_with_tree(tree['left'], row)
        else:
            return predict_with_tree(tree['right'], row)
    else:
        if row[col] in tree['split_value'][0]:
            return predict_with_tree(tree['left'], row)
        else:
            return predict_with_tree(tree['right'], row)

def predict_with_forest(forest, X):
    predictions = []

    for _, row in X.iterrows():
        votes = []
        for tree, features in forest:
            row_subset = row[features]
            prediction = predict_with_tree(tree, row_subset)
            votes.append(prediction)

        final_prediction = Counter(votes).most_common(1)[0][0]
        predictions.append(final_prediction)

    return predictions



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_full = pd.read_csv('/home/mustafa/Music/titanic/train.csv')

data_full_test = pd.read_csv('/home/mustafa/Music/titanic/test.csv')


X_unlabeled = data_full.drop('Survived', axis=1)
exit()

print(X_unlabeled[1:5])

y = data_full['Survived']




