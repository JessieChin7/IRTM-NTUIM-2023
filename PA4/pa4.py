import os
import re
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Function to preprocess documents
def process_document(text, stop_words, ps):
    text = re.sub(r'\d+', '', text)  # Remove digits
    translator = str.maketrans('', '', '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')
    text = text.translate(translator)  # Remove punctuation
    tokens = text.lower().split()  # Convert to lowercase and split
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  # Remove stopwords and single characters
    return [ps.stem(word) for word in tokens]  # Stemming

# Modify the cosine similarity function for normalized vectors
def cosine_similarity(vector1, vector2):
    # No need to divide by the norms, as the vectors are normalized
    return np.dot(vector1, vector2)

# Custom heap push operation
def heap_push(heap, item):
    heap.append(item)
    _sift_up(heap, len(heap) - 1)

# Custom heap pop operation
def heap_pop(heap):
    last_item = heap.pop()
    if heap:
        return_item = heap[0]
        heap[0] = last_item
        _sift_down(heap, 0)
    else:
        return_item = last_item
    return return_item

# Custom sift up operation
def _sift_up(heap, child_idx):
    while child_idx > 0:
        parent_idx = (child_idx - 1) >> 1
        if heap[child_idx][0] < heap[parent_idx][0]:
            heap[child_idx], heap[parent_idx] = heap[parent_idx], heap[child_idx]
            child_idx = parent_idx
        else:
            break

# Custom sift down operation
def _sift_down(heap, parent_idx):
    child_idx = 2 * parent_idx + 1
    while child_idx < len(heap):
        right_idx = child_idx + 1
        if right_idx < len(heap) and not heap[child_idx][0] < heap[right_idx][0]:
            child_idx = right_idx
        if heap[parent_idx][0] > heap[child_idx][0]:
            heap[parent_idx], heap[child_idx] = heap[child_idx], heap[parent_idx]
            parent_idx = child_idx
            child_idx = 2 * parent_idx + 1
        else:
            break

# Custom heapify operation
def heapify(heap):
    n = len(heap)
    for i in reversed(range(n//2)):
        _sift_down(heap, i)

# Function to perform HAC with different linkage options
def hierarchical_agglomerative_clustering(doc_vectors, Ks, linkage='single'):
    N = len(doc_vectors)
    C = {n: {i: {'sim': cosine_similarity(doc_vectors[n], doc_vectors[i]),
                'index': i} for i in range(N) if i != n} for n in range(N)}
    I = np.ones(N, dtype=bool)
    P = {n: [] for n in range(N)}
    clusters = {n: [n] for n in range(N)}  # Initialize clusters

    for n in P:
        P[n] = [(-value['sim'], value['index']) for value in C[n].values()]
        heapify(P[n])
    A = []

    while np.sum(I) > min(Ks):
        # Find the pair of clusters with maximum similarity
        k1, k2 = None, None
        max_sim = -np.inf
        for n, pq in P.items():
            if I[n] and pq:
                # Extract the similarity and index separately
                neg_sim, idx = heap_pop(pq)
                sim = -neg_sim  # Negate the similarity score
                if sim > max_sim:
                    max_sim, k1, k2 = sim, n, idx

        if k1 is None:  # No more clusters to merge
            break
        
        A.append((k1, k2, max_sim))

        I[k2] = False  # Mark the merged cluster as inactive
        P[k1] = []
        
        # Update the priority queues and similarity matrix
        for i in range(N):
            if I[i] and i != k1:
                # Delete old similarities for k1 and k2 from the priority queue of i
                P[i] = [(s, ind) for s, ind in P[i] if ind not in {k1, k2}]
                heapify(P[i])

                # Calculate new similarity for k1 based on the linkage choice
                new_sim = calculate_new_similarity(C, i, k1, k2, doc_vectors, linkage)

                # Update similarities in C
                C[i][k1] = {'sim': new_sim, 'index': k1}
                C[k1][i] = {'sim': new_sim, 'index': i}

                # Insert new similarity into the priority queue
                heap_push(P[i], (-new_sim, k1))

        # Reconstruct the priority queue for k1

        for i in range(N):
            if I[i] and i != k1:
                heap_push(P[k1], (-C[k1][i]['sim'], i))
                heapify(P[k1])

        # Update cluster membership
        clusters[k1].extend(clusters[k2])
        del clusters[k2]

    return A, clusters

# Helper function to calculate new similarity based on linkage choice
def calculate_new_similarity(C, i, k1, k2, doc_vectors, linkage):
    if linkage == 'single':
        return min(C[i][k1]['sim'], C[i][k2]['sim'])
    elif linkage == 'complete':
        return max(C[i][k1]['sim'], C[i][k2]['sim'])
    elif linkage == 'average':
        return (C[i][k1]['sim'] + C[i][k2]['sim']) / 2
    else:
        raise ValueError("Unknown linkage type: {}".format(linkage))

# Function to extract clusters for a given K from merge history
def get_clusters_for_K(merge_history, K, total_documents):
    clusters = {i: [i] for i in range(total_documents)}
    for merge in merge_history[-(total_documents-K):]:
        k1, k2, _ = merge
        clusters[k1].extend(clusters[k2])
        del clusters[k2]
    return clusters

# Function to save clusters
def save_clusters(clusters, filename):
    # Write to file
    with open(filename, 'w', encoding='utf-8') as file:
        for cluster in clusters.values():
            # Sort the document indices within each cluster in ascending order
            sorted_cluster_indices = sorted(cluster)
            # Write each document index to the file, followed by a newline character
            for doc_index in sorted_cluster_indices:
                file.write(f"{doc_index + 1}\n")
            # Write an extra newline character after each cluster to separate them
            file.write("\n")

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize document frequency dictionary and Porter Stemmer
document_frequency = {}
ps = PorterStemmer()

# Process documents and compute document frequency
data_dir = './data'
doc_vectors = []  # To store final normalized TF-IDF vectors
for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            tokens = process_document(f.read(), stop_words, ps)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

# Sort terms and build term index
sorted_terms = sorted(document_frequency.keys())
term_index = {term: idx for idx, term in enumerate(sorted_terms)}

# Compute TF-IDF for each document
N = len(os.listdir(data_dir))  # Total number of documents
for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            tokens = process_document(f.read(), stop_words, ps)
            tf = {token: tokens.count(token) for token in set(tokens)}
            tfidf_vector = np.zeros(len(term_index))
            for term, freq in tf.items():
                df_t = document_frequency[term]
                idf_t = math.log10(N / df_t)
                tfidf_vector[term_index[term]] = freq * idf_t
            # Normalize
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector /= norm
            doc_vectors.append(tfidf_vector)

# Convert doc_vectors to a numpy array
doc_vectors = np.array(doc_vectors)

# Run HAC and get clusters
merge_history, _ = hierarchical_agglomerative_clustering(doc_vectors, [8, 13, 20], linkage='complete')

clusters_k8 = get_clusters_for_K(merge_history, 8, N)
clusters_k13 = get_clusters_for_K(merge_history, 13, N)
clusters_k20 = get_clusters_for_K(merge_history, 20, N)

# Save clusters for different Ks
save_clusters(clusters_k8, './8.txt')
save_clusters(clusters_k13, './13.txt')
save_clusters(clusters_k20, './20.txt')