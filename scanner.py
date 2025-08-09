import string

# Function to clean and split text
def reader(filepath):
    with open(filepath, 'r') as file:
        text = file.read().lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

# Function to load keywords and weights from file
def load_keywords(filepath):
    keywords = {}
    with open(filepath, 'r') as file:
        for line in file:
            if ',' in line:
                word, weight = line.strip().lower().split(',')
                keywords[word] = int(weight)
    return keywords

# Function to calculate the weighted score
def calculate_score(cv_words, keyword_weights):
    score = 0
    matched = []
    for word in cv_words:
        if word in keyword_weights:
            score += keyword_weights[word]
            matched.append(word)
    return score, matched

# === RUN ===
cv_words = reader('/home/mustafa/Music/story.txt')
keyword_weights = load_keywords('/home/mustafa/Music/word_list.txt')

score, matched_keywords = calculate_score(cv_words, keyword_weights)

print(f"Matched keywords: {set(matched_keywords)}")
print(f"Total Score: {score}")

# Threshold logic
THRESHOLD = 15
if score >= THRESHOLD:
    print("Application PASSES — forward to employer.")
else:
    print("Application REJECTED — not enough relevant keywords.")

