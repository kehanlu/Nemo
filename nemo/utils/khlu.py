import os

def check_finename(filepath):
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    i = 1
    while os.path.exists(f"{base}.{i}{ext}"):
        i += 1
    return f"{base}.{i}{ext}"


def check_consecutive_words(text, words):
    # Convert text to lowercase and split into a list of words
    text_words = text.lower().split()
    
    # Convert words to check to lowercase
    words = words.lower().split()
    
    # Check for consecutive sequence
    for i in range(len(text_words) - len(words) + 1):
        if text_words[i:i+len(words)] == words:
            return True
    
    return False


from whisper_normalizer.basic import BasicTextNormalizer
from collections import defaultdict
def calculate_accuracy(results):
    """
    results: List of List of Dict
    """
    # ========================
    # Calculate accuracy
    # ========================
    normalizer = BasicTextNormalizer()
    question_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    outputs = []
    for i, batch in enumerate(results):
        # batch: [{}, {}]
        for result in batch:
            question_type = result["question_type"]
            metric = result["metric"]

            prediction = normalizer(result["prediction"].replace("<|eot_id|>", ""))
            target = normalizer(result["target"].replace("<|eot_id|>", ""))

            if metric == "accuracy":
                if check_consecutive_words(text=prediction, words=target):
                    question_groups[question_type]["correct"] += 1
            question_groups[question_type]["total"] += 1
            
            result["index"] = i
            outputs.append(result)
    accuracies = {}
    for question_type, counts in question_groups.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        accuracies[question_type] = accuracy

    return accuracies, outputs