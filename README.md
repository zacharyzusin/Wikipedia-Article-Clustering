# Wikipedia-Article-Clustering

## Introduction

This project was inspired by the **Wikipedia SpeedRuns** game, which encouraged me to think of Wikipedia as a directed graph, where each page represents an article, and links between them serve as directed edges. Initially, I thought it would an interesting idea to try to cluster Wikipedia articles based on network properties, such as clustering coefficients and betweenness centrality, as well as the Louvain method for community detection. However, these approaches did not produce satisfactory results.

I then shifted my focus to a textual-based approach wherein which I attempted to use word frequencies as a heuristic for determining page similarity, but this method proved too simplistic and ultimately unsuccessful. It became clear that a more sophisticated analysis was necessary, one that utilized semantic understanding rather than mere frequency counts.

## Methodology

To achieve a more meaningful clustering of Wikipedia articles, I implemented a comprehensive Natural Language Processing pipeline, which involved several key steps:

1. **Data Collection**: I gathered articles from various categories and for each article, I used the Wikipedia API and parsed the relevant text.

2. **Text Preprocessing**: The retrived text underwent preprocessing to remove references, lowercasing, punctuation, and stopwords. I employed lemmatization to reduce words to their base forms, enhancing the semantic consistency of the data.

3. **Semantic Embeddings**: To capture the semantic nuances of the text, I utilized **Sentence-BERT**, a pre-trained transformer model specifically designed for generating contextualized sentence embeddings. This step transformed the cleaned text data into high-dimensional embeddings that preserve semantic relationships and produced vectors which could be used to represent each article's contents.

4. **Dimensionality Reduction**: Given the high dimensionality of the embeddings, I applied **UMAP** (Uniform Manifold Approximation and Projection) for dimensionality reduction, allowing for visualization in a two-dimensional space while mostly preserving the structure of the original data.

5. **Clustering**: To identify distinct groups within the articles, I employed **K-Means clustering**., determining the optimal number of clusters using silhouette scores, which measure the quality of the clusters formed.

6. **Visualization**: Finally, I created an interactive scatter plot using **Plotly** to visualize the clustered articles in the UMAP-reduced space. The plot allows for easy exploration of the clusters, with hover functionality displaying the titles of the articles.

## Conclusion

The final pipeline proved mostly successful in clustering the Wikipedia articles into distinct groups based on their semantic content, however would always group a few seemingly disparate articles together incorrectly. And so, this pipeline certainly can be further improved by exploring additional NLP techniques and perhaps even integrating graph properties with textual analysis for a more robust outcome.
