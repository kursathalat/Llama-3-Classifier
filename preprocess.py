from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
import gensim.downloader as api
from sklearn.cluster import KMeans
from collections import Counter


dataset = load_dataset("microsoft/MeetingBank-QA-Summary")["test"]

#nltk.download('stopwords') # Uncomment this line if you haven't downloaded the stopwords

stop_words = set(stopwords.words('english'))

def sub_punct(text):
    """
    Remove punctuation from text but not numbers
    """
    if not text.isnumeric():
        return re.sub(r'[^a-z\s]', '', text)
    else:    
        return text

def preprocess_text(text):
    """
    Preprocess the text
    """
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    words = text.split()
    # Remove stop words
    words = [sub_punct(word) for word in words if word not in stop_words]
    return words


glove_model = api.load('glove-wiki-gigaword-50')

def get_embeddings(words):
    return {word: glove_model[word] for word in words if word in glove_model}




kmeans = KMeans(n_clusters=5, n_init=10) # let's keep 5 clusters for now

def kmeans_cluster(embeddings):
    """
    Cluster word embeddings using KMeans
    """

    word_vectors = list(embeddings.values())
    kmeans.fit(word_vectors)

    clusters = {word: kmeans.labels_[i] for i, word in enumerate(embeddings.keys())}

    cluster_counts = Counter(clusters.values())
    central_cluster = cluster_counts.most_common(1)[0][0]

    important_words = [word for word, cluster in clusters.items() if cluster == central_cluster]

    return important_words

# Let's classify the summaries into 3 classes: glacherry, vintrailly, and borriness
classes = []

for item in dataset:
    text = item["prompt"]
    cleaned_words = preprocess_text(text)
    embeddings = get_embeddings(cleaned_words)
    important_words = kmeans_cluster(embeddings)
    summary = [word.lower() for word in item["gpt4_summary"] .split()]

    common_word_count = 0
    for word in summary:
        if word in important_words:
            common_word_count += 1
    
    if common_word_count <= len(important_words)/5 :
        classes.append("glacherry")
    elif common_word_count > len(important_words)/5 and common_word_count <= 2*len(important_words)/5:
        classes.append("vintrailly")
    elif common_word_count > 2*len(important_words)/5:
        classes.append("borriness")

dataset = dataset.add_column("label", classes)

# If you want to push the dataset to the hub, uncomment the following line
#dataset.push_to_hub("meetingbank_features")