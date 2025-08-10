import pandas as pd
import random


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

def build_random_forest(data, n_trees, depth_range):
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
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_full = pd.read_csv('/home/mustafa/Music/titanic/train.csv')

data_full_test = pd.read_csv('/home/mustafa/Music/titanic/test.csv')

# Remove the 'Survived' column (the label)
X_unlabeled = data_full.drop('Survived', axis=1)

# If you want to keep the labels separately:
y = data_full['Survived']

print(build_random_forest(X_unlabeled, 10 ,(3,4)))




