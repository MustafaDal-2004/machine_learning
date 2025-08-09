import pandas as pd


data = pd.read_csv('/home/mustafa/Documents/learning/insurance.csv')

print(data[1:5])

non_numeric_columns = data.select_dtypes(exclude=['number']).columns.tolist()

print(non_numeric_columns)

def variance(values):
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

def splitter(data, value):
    left = [item for item in data if item >= value]
    right = [item for item in data if item < value]
    return left, right

def variance(values):
    if len(values) == 0:
        return 0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

def log2(x):
    if x <= 0:
        return 0  # avoid math domain error
    count = 0
    frac = x
    while frac < 1:
        frac *= 2
        count -= 1
    while frac >= 2:
        frac /= 2
        count += 1
    # Approximate fractional part
    result = count
    frac -= 1
    term = frac
    for i in range(1, 10):  # 10-term Taylor series approximation
        result += ((-1) ** (i + 1)) * (term ** i) / i
    return result

def entropy(rows):
    label_counts = {}
    for row in rows:
        label = row[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    total = len(rows)
    ent = 0
    for label in label_counts:
        p = label_counts[label] / total
        ent -= p * log2(p)
    return ent

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

def get_combinations(items, r):
    def combine(start, path):
        if len(path) == r:
            result.append(path)
            return
        for i in range(start, len(items)):
            combine(i + 1, path + [items[i]])

    result = []
    combine(0, [])
    return result

def best_categorical_split(column):
    categories = list(set(column))
    best_split = None
    smallest_entropy = float('inf')

    for i in range(1, len(categories) // 2 + 1):
        for group in get_combinations(categories, i):
            group_set = set(group)

            group1 = column[column.isin(group_set)]
            group2 = column[~column.isin(group_set)]

            ent1 = entropy(group1)
            ent2 = entropy(group2)

            total_ent = ent1 + ent2

            if total_ent < smallest_entropy:
                smallest_entropy = total_ent
                best_split = (group_set, set(categories) - group_set)

    return best_split, smallest_entropy

def build_unsupervised_tree(data, depth, current_depth=0):
    if current_depth >= depth:
        return {'leaf': data.index.tolist()}

    best_column = None
    best_score = float('inf')
    best_split = None
    split_type = None  # 'numeric' or 'categorical'

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
        return {'leaf': data.index.tolist()}  # no valid split

    # Perform the actual data split
    if split_type == 'numeric':
        left_data = data[data[best_column] >= best_split]
        right_data = data[data[best_column] < best_split]
        question = f"{best_column} >= {best_split}"
    else:  # categorical
        left_data = data[data[best_column].isin(best_split[0])]
        right_data = data[~data[best_column].isin(best_split[0])]
        question = f"{best_column} in {best_split[0]}"

    return {
        'question': question,
        'split_type': split_type,
        'column': best_column,
        'split_value': best_split,
        'left': build_unsupervised_tree(left_data, depth, current_depth + 1),
        'right': build_unsupervised_tree(right_data, depth, current_depth + 1)
    }

print(build_unsupervised_tree(data,3))