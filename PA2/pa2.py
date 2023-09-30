import os
import re
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer
import math

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

def load_vector(doc_id):
    with open(f'{doc_id}.txt', 'r') as f:
        lines = f.readlines()
    num_terms = int(lines[0].strip())
    vector = np.zeros(num_terms)
    for line in lines[1:]:
        index, tfidf = line.strip().split()
        index = int(index) - 1  # assuming indices start from 1 in the file
        tfidf = float(tfidf)
        vector[index] = tfidf
    return vector

def cosine(docx, docy):
    vector_x = load_vector(docx)
    vector_y = load_vector(docy)
    
    dot_product = np.dot(vector_x, vector_y)
    norm_x = np.linalg.norm(vector_x)
    norm_y = np.linalg.norm(vector_y)
    
    if norm_x > 0 and norm_y > 0:
        cosine_similarity = dot_product / (norm_x * norm_y)
    else:
        cosine_similarity = 0.0  # avoid division by zero
    
    return cosine_similarity




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


# Build a term index dictionary for easy lookup
term_index = {term: index for index, (term, df) in enumerate(sorted_terms, start=1)}

# Now compute the tf-idf vectors for each document
for filename in os.listdir('./dataset'):
    if filename.endswith('.txt'):
        filepath = os.path.join('./dataset', filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = process_document(text)
        tf = Counter(tokens)
        
        # Create a zero vector of length equal to the number of terms
        tfidf_vector = np.zeros(len(term_index))
        
        for term, freq in tf.items():
            if term in term_index:
                tf_t = freq
                df_t = document_frequency[term]
                N = len(os.listdir('./dataset'))  # Assuming all files in the dataset directory are text documents
                idf_t = math.log10(N / df_t)
                tfidf_t = tf_t * idf_t
                tfidf_vector[term_index[term] - 1] = tfidf_t  # -1 because indices start from 1
        
        # Normalize the tf-idf vector to unit length
        norm = np.linalg.norm(tfidf_vector)
        if norm > 0:
            tfidf_vector_unit = tfidf_vector / norm
        else:
            tfidf_vector_unit = tfidf_vector  # avoid division by zero

        # Get non-zero entries for the sparse representation
        non_zero_entries = [(index + 1, tfidf) for index, tfidf in enumerate(tfidf_vector_unit) if tfidf > 0]

        # Save the tf-idf unit vector to a file
        doc_id = os.path.splitext(filename)[0]  # Assuming filename is 'DocID.txt'
        with open(f'./tfidf/{doc_id}.txt', 'w') as f:
            f.write(f"{len(non_zero_entries)}\n")  # Write the number of non-zero entries
            for index, tfidf in non_zero_entries:
                f.write(f"{index}\t{tfidf:.3f}\n")  # Write the term index and tf-idf value, formatted to 3 decimal places

# Example usage:
docx = '1'
docy = '2'
similarity = cosine(docx, docy)
print(f'Cosine similarity between {docx} and {docy}: {similarity:.3f}')