import string
from typing import Dict, List, Tuple

# ============================================================
# Text Processing
# ============================================================

def read_and_tokenize(filepath: str) -> List[str]:
    """
    Reads a text file, normalizes case, removes punctuation,
    and splits the text into individual words.

    Parameters:
        filepath (str): Path to the text file

    Returns:
        List[str]: List of cleaned words
    """
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read().lower()

    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# ============================================================
# Keyword Loading
# ============================================================

def load_keyword_weights(filepath: str) -> Dict[str, int]:
    """
    Loads keywords and associated weights from a file.
    Each line should be formatted as: keyword,weight

    Parameters:
        filepath (str): Path to keyword file

    Returns:
        Dict[str, int]: Dictionary mapping keywords to weights
    """
    keywords = {}

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if "," not in line:
                continue

            keyword, weight = line.strip().lower().split(",", 1)
            keywords[keyword] = int(weight)

    return keywords


# ============================================================
# Scoring Logic
# ============================================================

def calculate_weighted_score(
    document_words: List[str],
    keyword_weights: Dict[str, int]
) -> Tuple[int, List[str]]:
    """
    Calculates a weighted relevance score for a document
    based on keyword occurrences.

    Parameters:
        document_words (List[str]): Tokenized document text
        keyword_weights (Dict[str, int]): Keywords with weights

    Returns:
        Tuple[int, List[str]]:
            - Total score
            - List of matched keywords
    """
    score = 0
    matched_keywords = []

    for word in document_words:
        if word in keyword_weights:
            score += keyword_weights[word]
            matched_keywords.append(word)

    return score, matched_keywords


# ============================================================
# Execution
# ============================================================

if __name__ == "__main__":
    CV_PATH = "data/story.txt"
    KEYWORD_PATH = "data/word_list.txt"
    THRESHOLD = 15

    cv_words = read_and_tokenize(CV_PATH)
    keyword_weights = load_keyword_weights(KEYWORD_PATH)

    score, matched = calculate_weighted_score(cv_words, keyword_weights)

    print(f"Matched keywords: {sorted(set(matched))}")
    print(f"Total score: {score}")

    if score >= THRESHOLD:
        print("Application PASSES — forward to employer.")
    else:
        print("Application REJECTED — insufficient keyword relevance.")

