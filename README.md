<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Information Retrieval and Text Mining Assignments</title>
</head>
<body>
    <h1>Information Retrieval and Text Mining</h1>
    <p>2023 Spring NTUIM</p>
    <h2>Programming Assignment 1: Term Extraction</h2>
    <ul>
        <li><strong>Objective</strong>: Extract terms from a single English news document.</li>
        <li><strong>Process</strong>:
            <ul>
                <li>Tokenization of the document text.</li>
                <li>Conversion of all text to lowercase.</li>
                <li>Stemming using the Porter Stemmer algorithm.</li>
                <li>Removal of stopwords.</li>
            </ul>
        </li>
    </ul>
    <h2>Programming Assignment 2: TF-IDF Vectorization</h2>
    <ul>
        <li><strong>Objective</strong>: Convert a set of documents into TF-IDF vectors.</li>
        <li><strong>Process</strong>:
            <ul>
                <li>Construction of a dictionary based on extracted terms.</li>
                <li>Recording of document frequency for each term.</li>
                <li>Transformation of each document into a TF-IDF unit vector.</li>
                <li>Implementation of a cosine similarity function <code>cosine(Docx, Docy)</code> to calculate the cosine similarity between any two documents.</li>
            </ul>
        </li>
    </ul>
    <h2>Programming Assignment 3: Multinomial Naive Bayes Classifier</h2>
    <ul>
        <li><strong>Objective</strong>: Implement and test a Multinomial Naive Bayes Classifier.</li>
        <li><strong>Process</strong>:
            <ul>
                <li>Classification of documents into 13 classes using a training set of 15 documents per class.</li>
                <li>Feature selection employed to reduce the vocabulary to the top 500 terms using methods like Χ2 test and likelihood ratio.</li>
                <li>Add-one smoothing used to avoid zero probabilities in the classification.</li>
            </ul>
        </li>
    </ul>
    <h2>Programming Assignment 4: Hierarchical Agglomerative Clustering (HAC)</h2>
    <ul>
        <li><strong>Objective</strong>: Perform HAC on the collection of documents.</li>
        <li><strong>Process</strong>:
            <ul>
                <li>Documents represented as normalized TF-IDF vectors (from Assignment 2).</li>
                <li>Use of cosine similarity for pairwise document similarity.</li>
                <li>Exploration of different similarity measures between clusters including single-link, complete-link, group-average, and centroid similarity.</li>
                <li>Clustering results for K = 8, 13, and 20.</li>
                <li>Implementation of a custom HEAP to optimize the retrieval of the cluster pair with the maximal similarity.</li>
            </ul>
        </li>
    </ul>
</body>
</html>
