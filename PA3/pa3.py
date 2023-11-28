import os
import string
import nltk
import csv
from collections import defaultdict
from math import log
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from scipy.stats import chi2_contingency
# Download necessary NLTK packages
nltk.download('stopwords')

# Function to process documents: tokenize, remove stop words, and stem
def process_document(text):
    """
    Process the given text by tokenizing, removing stop words, and stemming.
    """
    translator = str.maketrans('', '', string.punctuation + string.digits)
    tokens = text.translate(translator).split()
    tokens = [word.lower() for word in tokens]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens

# Function to calculate the log likelihood ratio
def log_likelihood_ratio(n11, n01, n10, n00):
    """
    Calculate the log likelihood ratio for given frequency counts.
    """
    def safe_log(x):
        return log(x) if x > 0 else 1e-10

    N = n11 + n01 + n10 + n00
    p = (n11 + n01) / N
    p1 = n11 / (n11 + n10) if n11 + n10 > 0 else 0
    p2 = n01 / (n01 + n00) if n01 + n00 > 0 else 0

    log_L1 = n11 * safe_log(p) + n10 * safe_log(1 - p) + n01 * safe_log(p) + n00 * safe_log(1 - p)
    log_L2 = n11 * safe_log(p1) + n10 * safe_log(1 - p1) + n01 * safe_log(p2) + n00 * safe_log(1 - p2)

    return -2 * (log_L1 - log_L2)

# Function to calculate Chi-Square
def chi_square(term_class_freq, overall_term_freq, class_sizes, total_docs, term):
    """
    Calculate the Chi-Square statistic for a given term.
    """
    table = []
    for class_id in class_sizes:
        # Frequency of term in class
        f_term_in_class = term_class_freq[term].get(class_id, 0)
        # Frequency of term not in class
        f_term_not_in_class = overall_term_freq[term] - f_term_in_class
        # Frequency of not term in class
        f_not_term_in_class = class_sizes[class_id] - f_term_in_class
        # Frequency of not term not in class
        f_not_term_not_in_class = total_docs - class_sizes[class_id] - f_term_not_in_class

        table.append([f_term_in_class, f_term_not_in_class])
        table.append([f_not_term_in_class, f_not_term_not_in_class])

    chi2_stat, p, dof, ex = chi2_contingency(table, correction=False)
    return chi2_stat

# Modified feature extraction and selection function
def extract_features(train_data):
    term_class_freq = defaultdict(lambda: defaultdict(int))
    overall_term_freq = defaultdict(int)
    class_sizes = defaultdict(int)
    total_docs = 0

    for class_id, doc_ids in train_data.items():
        for doc_id in doc_ids:
            class_sizes[class_id] += 1
            with open(f'./data/{doc_id}.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = process_document(text)
            for token in set(tokens):
                term_class_freq[token][class_id] += 1
                overall_term_freq[token] += 1

    total_docs = sum(class_sizes.values())
    llr_scores, chi_scores = defaultdict(int), defaultdict(int)
    
    for term, freq in term_class_freq.items():
        for class_id, count in class_sizes.items():
            n11 = freq[class_id]
            n10 = class_sizes[class_id] - n11
            n01 = overall_term_freq[term] - n11
            n00 = total_docs - class_sizes[class_id] - n01

            llr_scores[term] = max(llr_scores[term], log_likelihood_ratio(n11, n01, n10, n00))
            chi_scores[term] = max(chi_scores[term], chi_square(term_class_freq, overall_term_freq, class_sizes, total_docs, term))

    top_llr = set(term for term, _ in sorted(llr_scores.items(), key=lambda item: item[1], reverse=True)[:500])
    top_chi = set(term for term, _ in sorted(chi_scores.items(), key=lambda item: item[1], reverse=True)[:500])

    # Return the intersection of the top terms from each method
    return top_llr & top_chi

# Load training data
train_data = {}
with open('training.txt', 'r') as f:
    for line in f:
        class_id, *doc_ids = line.strip().split()
        train_data[int(class_id)] = [int(doc_id) for doc_id in doc_ids]
# Extract the training document IDs to exclude them from prediction
training_doc_ids = set()
for doc_ids in train_data.values():
    training_doc_ids.update(doc_ids)

# Extract features
top_terms = extract_features(train_data)

# Functions for Multinomial Naive Bayes Classifier
def train_multinomial_nb(C, D):
    """
    Train the Multinomial Naive Bayes classifier.
    """
    # Helper functions for the classifier
    def extract_vocabulary(D):
        """ 
        Extract vocabulary from documents using only the top terms.
        """
        V = set()
        for d in D:
            with open(d, 'r', encoding='utf-8') as f:
                V.update([word for word in process_document(f.read()) if word in top_terms])
        return V


    def concatenate_text_of_all_docs_in_class(D_c):
        """ Concatenate text of all documents in a class """
        text_c = ""
        for d in D_c:
            with open(d, 'r', encoding='utf-8') as f:
                d_text = ' '.join(process_document(f.read()))
                text_c += d_text
        return text_c

    def count_tokens_of_term(t_ct, text_c, V):
        """ Count tokens of term """
        text_c = text_c.split()
        for term in text_c:
            if term in V:
                t_ct[term] += 1
        return t_ct

    V = extract_vocabulary(D)
    N = len(D)
    prior = {}
    condprob = defaultdict(lambda: defaultdict(float))
    T_ct = defaultdict(lambda: defaultdict(int))

    for c in C:
        D_c = [d for d in D if D[d] == c]
        N_c = len(D_c)
        prior[c] = N_c / N
        text_c = concatenate_text_of_all_docs_in_class(D_c)
        T_ct[c] = count_tokens_of_term(T_ct[c], text_c, V)
        for t in V:
            condprob[t][c] = (T_ct[c][t] + 1) / (sum(T_ct[c].values()) + len(V))

    return V, prior, condprob

def apply_multinomial_nb(C, V, prior, condprob, d):
    """
    Apply the trained Multinomial Naive Bayes classifier to a new document.
    """
    def extract_tokens_from_doc(V, d):
        """ Extract tokens from document """
        tokens = process_document(d)
        return [token for token in tokens if token in V]

    W = extract_tokens_from_doc(V, d)
    score = defaultdict(float)
    for c in C:
        score[c] = log(prior[c])
        for t in W:
            score[c] += log(condprob[t][c])
    return max(score, key=score.get)

# Main program
D = {f'./data/{doc_id}.txt': class_id for class_id, doc_ids in train_data.items() for doc_id in doc_ids}
C = set(train_data.keys())
V, prior, condprob = train_multinomial_nb(C, D)

# Predict and write output to a CSV file
with open('hw3_output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Id', 'Value'])
    for filename in os.listdir('./data'):
        if filename.endswith('.txt'):
            doc_id = int(os.path.splitext(filename)[0])
            if doc_id in training_doc_ids:
                continue
            with open(f'./data/{filename}', 'r', encoding='utf-8') as f:
                text = f.read()
            predicted_class = apply_multinomial_nb(C, V, prior, condprob, text)
            csvwriter.writerow([doc_id, predicted_class])