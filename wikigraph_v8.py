import requests
import string
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Download NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define articles by category
articles = {
    'Science': [
        'Quantum Mechanics',
        'Theory of Relativity',
        'Evolution',
        'Photosynthesis',
        'Plate Tectonics',
        'Genetics',
        'Neuroscience',
        'Black Holes',
        'Nanotechnology',
        'Astrobiology',
        'Astrophysics',
        'Climate Change',
        'Quantum Computing',
        'Evolutionary Biology',
        'Stem Cell Research',
        'Dark Matter',
        'Genetic Engineering',
        'Biochemistry',
        'Artificial Intelligence',
        'Biophysics',
        'Microbiology',
        'Astrobiology',
        'Botany',
        'Chemistry',
        'Ecology',
        'Geology',
        'Physics',
        'Mathematics',
        'Statistics',
        'Pharmacology',
        'Forensic Science'
    ],
    'Historical Events': [
        'World War II',
        'American Civil War',
        'French Revolution',
        'Fall of the Berlin Wall',
        'Moon Landing',
        'Renaissance',
        'Great Depression',
        'Industrial Revolution',
        'The Crusades',
        'Civil Rights Movement',
        'Cold War',
        'American Revolution',
        'Ancient Rome',
        'Byzantine Empire',
        'World War I',
        'Civil War in Spain',
        'The Enlightenment',
        'The Great Wall of China',
        'The Fall of the Roman Empire',
        'The Vietnam War',
        'The Spanish Inquisition',
        'The Ottoman Empire',
        'The Silk Road',
        'The Age of Exploration',
        'The American Frontier',
        'The Rise of Fascism',
        'The Battle of Hastings',
        'The French and Indian War',
        'The Gold Rush',
        'The Suffrage Movement',
        'The Cold War'
    ],
    'Food and Drink': [
        'Pizza',
        'Sushi',
        'Tacos',
        'Chocolate',
        'Coffee',
        'Pasta',
        'Curry',
        'Wine',
        'Bread',
        'Ice Cream',
        'Cheesecake',
        'Sushi Roll',
        'Hamburger',
        'Tiramisu',
        'Pho',
        'Tapas',
        'Salad',
        'Barbecue',
        'Croissant',
        'Muffin',
        'Pancakes',
        'Lasagna',
        'Korean BBQ',
        'Fried Rice',
        'Biryani',
        'Pudding',
        'Macarons',
        'Clam Chowder',
        'Falafel',
        'Goulash',
        'Ceviche',
        'Chili'
    ],
    'Technology and Innovation': [
        'Internet of Things',
        'Blockchain',
        'Virtual Reality',
        'Augmented Reality',
        '3D Printing',
        'Cybersecurity',
        'Self-driving Cars',
        'Renewable Energy',
        'Biometrics',
        'Wearable Technology',
        '5G Technology',
        'Smart Homes',
        'Quantum Computing',
        'Machine Learning',
        'Robotics',
        'Cloud Computing',
        'Drones',
        'Cryptocurrency',
        'Genomics',
        'Edge Computing',
        'Artificial Intelligence',
        'Augmented Analytics',
        'Fintech',
        'Smart Cities',
        'Space Exploration',
        'Nanotechnology',
        'Telemedicine',
        'Voice Assistants',
        'Big Data',
        'Digital Twins',
        'Quantum Cryptography'
    ],
    'Sports and Recreation': [
        'Soccer',
        'Basketball',
        'Tennis',
        'Cricket',
        'Baseball',
        'Golf',
        'Swimming',
        'Cycling',
        'Running',
        'Hiking',
        'Skiing',
        'Snowboarding',
        'Surfing',
        'Rock Climbing',
        'Yoga',
        'Martial Arts',
        'Table Tennis',
        'Badminton',
        'Volleyball',
        'Rugby',
        'American Football',
        'Ice Hockey',
        'Formula 1',
        'Motocross',
        'Gymnastics',
        'Wrestling',
        'Field Hockey',
        'Lacrosse',
        'Surf Lifesaving',
        'CrossFit',
        'Ultimate Frisbee'
    ]
}

# Function to fetch Wikipedia article content by title
def fetch_wikipedia_article_content(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'parse' in data:
            raw_html = data['parse']['text']['*']
            soup = BeautifulSoup(raw_html, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return content
        else:
            return None
    except requests.RequestException as e:
        return None

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(lemmatized_tokens)

# Function to get Sentence-BERT embeddings
def get_sentence_bert_embeddings(texts, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

# Function to fetch and preprocess articles
def fetch_and_preprocess_articles(articles):
    article_titles = [title for category in articles.values() for title in category]
    articles_content = [fetch_wikipedia_article_content(title) for title in article_titles]
    valid_articles, valid_titles = [], []
    for title, content in zip(article_titles, articles_content):
        if content:
            processed_article = preprocess_text(content)
            valid_articles.append(processed_article)
            valid_titles.append(title)
    return valid_articles, valid_titles

# Function to determine optimal number of clusters using silhouette score
def determine_optimal_clusters(data, cluster_range=range(2, 20)):
    silhouette_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters

# Main pipeline
def main():
    # Fetch and preprocess articles
    valid_articles, valid_titles = fetch_and_preprocess_articles(articles)
    
    # Get Sentence-BERT embeddings
    embeddings = get_sentence_bert_embeddings(valid_articles)

    # Normalize the features
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Dimensionality Reduction with UMAP
    umap = UMAP(n_components=2, random_state=42)
    reduced_data_umap = umap.fit_transform(normalized_embeddings)

    # Determine optimal number of clusters using silhouette score
    optimal_clusters = determine_optimal_clusters(reduced_data_umap)

    # Clustering with K-Means
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=0)
    clusters = kmeans.fit_predict(reduced_data_umap)

    # Create a DataFrame for visualization
    df = pd.DataFrame(reduced_data_umap, columns=['Component 1', 'Component 2'])
    df['Cluster'] = clusters
    df['Article Title'] = valid_titles  # Add valid article titles for hover info

    # Interactive Visualization with Plotly
    fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', 
                     title='Clustered Wikipedia Articles in UMAP-reduced Space',
                     labels={'Component 1': 'UMAP Component 1', 'Component 2': 'UMAP Component 2'},
                     hover_name='Article Title')

    fig.update_traces(textposition='top center')
    fig.show()

# Run the main function
if __name__ == "__main__":
    main()
