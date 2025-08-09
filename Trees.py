def read_csv(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            data.append(line.split(','))
    header = data[0]
    rows = data[1:]
    return header, rows

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

#determines how mixed up the partition is if it really mixed up or the same data reapeted pretty much
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

def is_numeric(value):
    try:
        float(value)
        return True
    except:
        return False

#seperate a column based on a vaule so if >= 30 or <30 split into two partiontion for numeric columns
def partition(rows, column, value):
    true_rows, false_rows = [], []
    for row in rows:
        if is_numeric(value):
            if float(row[column]) >= float(value):
                true_rows.append(row)
            else:
                false_rows.append(row)
        else:
            if row[column] == value:
                true_rows.append(row)
            else:
                false_rows.append(row)
    return true_rows, false_rows

#tell us the entropy change and if it went down as we did our split
def info_gain(left, right, current_uncertainty):
    p = len(left) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

#this will find the best split for the data to catogrorise
def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = {}
        for row in rows:
            values[row[col]] = True  # unique values only

        for val in values:
            true_rows, false_rows = partition(rows, col, val)
            if not true_rows or not false_rows:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain > best_gain:
                best_gain = gain
                best_question = (col, val)
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = {}
        for row in rows:
            label = row[-1]
            if label not in self.predictions:
                self.predictions[label] = 0
            self.predictions[label] += 1

class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question[0], question[1])
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return DecisionNode(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    col, val = node.question
    if is_numeric(val):
        if float(row[col]) >= float(val):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)
    else:
        if row[col] == val:
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)
        
header, data = read_csv("/home/mustafa/Documents/learning/airlines_flights_data.csv")
tree = build_tree(data)

# Test prediction
print("Header:", header)
for i in range(3):  # test first 3 rows
    print(f"Row {i+1} prediction:", classify(data[i], tree))

#you should learn entropy and how to do

