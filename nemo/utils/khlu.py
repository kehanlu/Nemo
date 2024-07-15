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