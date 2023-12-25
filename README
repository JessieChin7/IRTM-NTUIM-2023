# Information Retrieval and Text Mining
2023 Spring NTUIM

## Programming Assignment 1: Term Extraction

- **Objective**: Extract terms from a single English news document.
- **Process**:
  - Tokenization of the document text.
  - Conversion of all text to lowercase.
  - Stemming using the Porter Stemmer algorithm.
  - Removal of stopwords.

## Programming Assignment 2: TF-IDF Vectorization

- **Objective**: Convert a set of documents into TF-IDF vectors.
- **Process**:
  - Construction of a dictionary based on extracted terms.
  - Recording of document frequency for each term.
  - Transformation of each document into a TF-IDF unit vector.
  - Implementation of a cosine similarity function `cosine(Docx, Docy)` to calculate the cosine similarity between any two documents.

## Programming Assignment 3: Multinomial Naive Bayes Classifier

- **Objective**: Implement and test a Multinomial Naive Bayes Classifier.
- **Process**:
  - Classification of documents into 13 classes using a training set of 15 documents per class.
  - Feature selection employed to reduce the vocabulary to the top 500 terms using methods like Χ2 test and likelihood ratio.
  - Add-one smoothing used to avoid zero probabilities in the classification.

## Programming Assignment 4: Hierarchical Agglomerative Clustering (HAC)

- **Objective**: Perform HAC on the collection of documents.
- **Process**:
  - Documents represented as normalized TF-IDF vectors (from Assignment 2).
  - Use of cosine similarity for pairwise document similarity.
  - Exploration of different similarity measures between clusters including single-link, complete-link, group-average, and centroid similarity.
  - Clustering results for K = 8, 13, and 20.
  - Implementation of a custom HEAP to optimize the retrieval of the cluster pair with the maximal similarity.