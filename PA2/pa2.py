from collections import Counter
from nltk.stem import PorterStemmer
import os
import re
# Load the stopwords
with open('stopwords.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

ps = PorterStemmer()

def process_document(text):
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # The same steps as your previous code
    translator = str.maketrans('', '', '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')
    tokens = text.translate(translator).split()
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return tokens

# Initialize a Counter for document frequency
document_frequency = Counter()

# Iterate through all the files in the dataset directory
for filename in os.listdir('./dataset'):
    if filename.endswith('.txt'):
        filepath = os.path.join('./dataset', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = process_document(text)
        # Update document frequency - only count each term once per document
        document_frequency.update(set(tokens))

# Sort the terms in ascending order
sorted_terms = sorted(document_frequency.items(), key=lambda x: x[0])

# Save the dictionary and document frequency to a file
with open('dictionary.txt', 'w') as f:
    for index, (term, df) in enumerate(sorted_terms, start=1):
        f.write(f"{index}\t{term}\t{df}\n")
