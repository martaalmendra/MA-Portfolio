#### IMPORTS ####
# General Purpose
import pandas as pd
import numpy as np 
import time
import re
import string
import spacy
import random

# Preprocessing
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
sent_tokenizer = PunktSentenceTokenizer()
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from collections import defaultdict
from collections import Counter
from num2words import num2words
from gensim.models import Word2Vec

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score
from hdbscan import HDBSCAN
from sklearn.decomposition import TruncatedSVD

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import os
import networkx as nx
from tqdm import tqdm
tqdm.pandas()


#Topic Modelling
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.models import LdaModel, LsiModel
from gensim.models.coherencemodel import CoherenceModel

# Classification and Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay



##### PRE-PROCESSING AND CLEANING ####

# General Visualizations
def plot_bar(data, title='', xlabel='', ylabel='', color='violet'):
    plt.figure(figsize=(8, 4))
    if isinstance(data, pd.Series):
        data.plot(kind='bar', color=color)
    else:
        data.value_counts().plot(kind='bar', color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.show()


def plot_histogram(data, column, bins=20, color='violet', title='', xlabel='', ylabel='', avg_line=False):
    plt.figure(figsize=(8, 4))
    
    sns.histplot(data[column], kde=False, bins=bins, color=color, edgecolor='black')
    if avg_line:
        avg_value = data[column].mean()
        plt.axvline(avg_value, color='red', linestyle='dashed', linewidth=2, label=f'Average {column}: {avg_value:.2f}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    
    if avg_line:
        plt.legend()
    
    plt.show()

# Data Cleaning

def remove_closed_parts(timings):
    parts = timings.split('),')
    open_parts = []
    for part in parts:
        if 'Closed' not in part:
            open_parts.append(part)
    return '),'.join(open_parts)

def times_split(timings, restaurants):
    open_hours = []
    open_days = []
    for i, text in enumerate(timings):
        if pd.notna(text): 
            text = remove_closed_parts(text)  
            if '(' not in text: 
                open_hours.append(text)
                open_days.append(None)
            elif text.count('(') == 1:  
                hours, days = text.split('(', 1)
                open_hours.append(hours.strip())
                open_days.append(days.replace(')', '').strip()) 
            else:   
                parts = text.split('),')
                hours_list = []
                days_list = []
                for part in parts:
                    if '(' in part:
                        hours, days = part.split('(', 1)
                        hours_list.append(hours.strip())
                        days_list.append(days.replace(')', '').strip())
                open_hours.append('. '.join(hours_list))
                open_days.append('. '.join(days_list))
        else:
            open_hours.append(None)
            open_days.append(None)

    restaurants['Open_Hours'] = open_hours
    restaurants['Open_Days'] = open_days

# Function to expand day ranges
def expand_days(day_range):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if '-' in day_range:
        start_day, end_day = day_range.split('-')
        start_index = days.index(start_day)
        end_index = days.index(end_day)
        return ', '.join(days[start_index:end_index + 1])
    return day_range

# Function to transform the open_days column
def transform_open_days(open_days):
    if pd.isna(open_days):
        return open_days
    parts = open_days.split('. ')
    expanded_parts = [', '.join([expand_days(day_range) for day_range in part.split(', ')]) for part in parts]
    return '. '.join(expanded_parts)


emoji_symbols = re.compile("["                
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\u2600-\u26FF"          # Miscellaneous Symbols
        u"\u2700-\u27BF"          # Dingbat Symbols
        "]", flags=re.UNICODE)


def process_emojis(review):
    review = emoji_symbols.sub(" ", review)
    return review
    
    
def basic_clean(review):
    pattern = (
        r"(\b[A-Za-z0-9.%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)"     # Email addresses
        r"|(@[A-Za-z0-9]+)"                                          # Social media tags
        r"|(\w+://\S+|http.+?)"                                      # Website links
        r"|(<.*?>)"                                                  # HTML tags
        r"|([^0-9A-Za-z{} ])"                                        # Non-alphanumeric characters (except emojis)
    )

    # Substitute all patterns with a space
    review = re.sub(pattern, " ", review, flags=re.MULTILINE)

    # Remove punctuation, replace multiple spaces with a single space, and strip whitespace in one line
    review = re.sub(f"[{re.escape(string.punctuation)}]", " ", review).strip() 
    review = re.sub(r'\s+', ' ', review)                                        

    return review



#Convert Numbers into their word form 
def numbs_to_txt(review):
    # Replace '/' between two numbers with 'out of'
    review = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', review)

    # Replace '-' between two numbers with 'out of'
    review = re.sub(r'(\d+)-(\d+)', r'\1 out of \2', review)

    # Replace float numbers with 'point'
    review = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', review)
    
    # Find all numeric values in the text and convert them to words
    review = re.sub(r'\b\d+\b', lambda x: num2words(x.group()), review)
    
    return review



def correct_repetition(text):
    # If the text is a number (ex: "5"), return it.
    if text.isdigit():
        return text
    corrected_text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    return corrected_text



# Pre-processing pipeline


nlp = spacy.load("en_core_web_sm")
wordnet_lem = WordNetLemmatizer()
# stopwords and any special negation words
stopwords = set(stopwords.words("english"))
neg_words = {'no', 'not'}



def text_pre_process_pipeline(raw_text, no_emojis = True, no_puctuation = True, no_stopwords=True, lower=True, lemmatized=True):
    if lower:
        raw_text = raw_text.lower()

    if no_emojis:
        raw_text = process_emojis(raw_text)

    clean_text = numbs_to_txt(raw_text)

    if no_puctuation:
        #Remove isolated chracters with exception to ponctuation:
        clean_text = re.sub(r'\b(?![\W_])\w\b', '', clean_text)
        clean_text = basic_clean(clean_text)

    clean_text = correct_repetition(clean_text)
   
    doc = nlp(clean_text)
    processed_text = []
    tokenized = []

    for token in doc:
        word = token.lemma_ if lemmatized else token.text  
        if no_stopwords and word not in stopwords or word in neg_words:
            processed_text.append(word) 
            tokenized.append(token.text)

    clean_text = ' '.join(processed_text)
    
    return clean_text, tokenized



##### DATA VISUALIZATIONS ####

# Word Cloud

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    hue = random.choice([330, 310, 290, 270]) 
    saturation = random.randint(60, 100)       
    lightness = random.randint(60, 80)        
    return "hsl({}, {}%, {}%)".format(hue, saturation, lightness)


def wordcloud_generator(data, column, unique_labels=None, vectorisation='tfidf', export=False, before=True):
   
    if export and unique_labels:
        corpus = []
        for label in unique_labels:
            label_doc = ""
            for text in data[column].loc[data[f"has_label_{label}"] == 1]:
                label_doc += " " + text
            corpus.append(label_doc)
        print(f"Length of unique_labels: {len(unique_labels)}; Length of corpus: {len(corpus)}")
    else:
        corpus = [data[column].str.cat(sep=' ')]  

    if export:
        folder = "WordClouds_before" if before else "WordClouds_after"
        os.makedirs(folder, exist_ok=True)
    
    for idx, label_doc in enumerate(corpus):
        if vectorisation == 'tfidf':
            title = 'tfidf'
            vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
        else:
            title = 'BoW Frequencies'
            vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        
        bigrams = vectorizer.fit_transform([label_doc])
        bigrams_freq = dict(zip(vectorizer.get_feature_names_out(), bigrams.sum(axis=0).A1))
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            color_func=color_func
        ).generate_from_frequencies(bigrams_freq)
        
        if export:
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {unique_labels[idx]} ({title})')
            plt.savefig(f'{folder}/WordCloud_{unique_labels[idx]}_{title}.png', format='png')
            plt.close()
        else:
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Bigram {title}')
            plt.show()


# Co-occurrence

def cooccurrence_matrix_review_generator(preproc_sentences):

    co_occurrences = defaultdict(Counter)
    for sentence in tqdm(preproc_sentences):
        for token_1 in sentence:
            for token_2 in sentence:
                if token_1 != token_2:
                    co_occurrences[token_1][token_2] += 1

    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    return co_matrix_df


def cooccurrence_matrix_window_generator(preproc_sentences, window_size):
    co_occurrences = defaultdict(Counter)

    for sentence in tqdm(preproc_sentences):
        for i, word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    co_occurrences[word][sentence[j]] += 1

    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    return co_matrix_df


# Network Graph

def cooccurrence_network_generator(cooccurrence_matrix_df, label, n_highest_words=20, figsize=(8, 6), export=False):
   
    filtered_df = cooccurrence_matrix_df.iloc[:n_highest_words, :n_highest_words]
    if filtered_df.empty:
        print(f"Co-occurrence matrix for label {label} is empty. Skipping graph generation...")
        return
    graph = nx.Graph()

    for word in filtered_df.columns:
        graph.add_node(word, size=filtered_df[word].sum())

    for word1 in filtered_df.columns:
        for word2 in filtered_df.columns:
            if word1 != word2:
                weight = filtered_df.loc[word1, word2]
                if weight > 0: 
                    graph.add_edge(word1, word2, weight=weight)

    pos = nx.spring_layout(graph, k=0.5)

    node_sizes = [data['size'] * 0.5 for _, data in graph.nodes(data=True)]
    edge_weights = [0.01 * graph[u][v]['weight'] for u, v in graph.edges()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=edge_weights)
    nx.draw_networkx_labels(graph, pos, font_weight='bold', font_size=10)
    plt.title(f"Co-occurrence Network for {label}")
    plt.tight_layout()

    if export:
        folder = "co_occurrences_cuisines"
        os.makedirs(folder, exist_ok=True)

        save_path = os.path.join(folder, f"Cooccurrence_{label}.png")
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.close()
        print(f"Saved co-occurrence graph for {label} at {save_path}")
    else:
        plt.show()


# Donut Chart 

def donut_chart(data, column):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(data[column])

    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.sum(axis=0).A1))

    most_common_tokens = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    donut_input = pd.DataFrame(most_common_tokens, columns=['words', 'frequency'])

    fig = px.pie(donut_input, values='frequency', names='words', hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()



# Tree Map

def tree_map(data, column):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(data[column])

    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.sum(axis=0).A1))

    words_df = pd.DataFrame(word_freq.items(), columns=['words', 'frequency'])
    words_df = words_df.sort_values(by='frequency', ascending=False).head(500)

    words_df['pos_tag'] = words_df['words'].apply(lambda word: pos_tag([word])[0][1])

    fig = px.treemap(words_df,
                    path=[px.Constant('Restaurant Reviews'), 'pos_tag', 'words'],
                    values='frequency',
                    color='frequency',
                    color_continuous_scale='pinkyl')

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show()



##### VECTORIZATION #####

def remove_rare_words_min_df(dataframe, text_column, min_df=0.002):

    vectorizer = CountVectorizer(min_df=min_df)
    matrix = vectorizer.fit_transform(dataframe[text_column])
    vocabulary = set(vectorizer.get_feature_names_out())
    all_words = set(word for text in dataframe[text_column] for word in text.split())

    removed_words = list(all_words - vocabulary)

    def filter_words(text):
        return ' '.join([word for word in text.split() if word in vocabulary])

    dataframe['cleaned_reviews_reduced_words'] = dataframe[text_column].apply(filter_words)

    return dataframe, removed_words



def vectorization(data, text_column, vector_column_name, vectorizer_type="bow", norm='l2', ngram_range=(1, 2), binary=False):

    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=binary) 
        
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, norm=norm)
    
    matrix = vectorizer.fit_transform(data[text_column])

    data[vector_column_name] = list(matrix.toarray())

    return data, vectorizer, matrix



##### Targets of Opportunity - Co-occurences, Clustering and Topic Modelling #####


#Clustering
def inertia_and_elbow_plotter(tf_matrix, max_k=10, verbose=True):
    x_k_nr = []
    y_inertia = []
    
    for k in tqdm(range(2, max_k+1)):
        x_k_nr.append(k)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tf_matrix)
        y_inertia.append(kmeans.inertia_)
        
        if verbose:
            print(f"For k = {k}, inertia = {round(kmeans.inertia_, 3)}")

    fig = px.line(x=x_k_nr, y=y_inertia, markers=True, labels={'x': 'Number of clusters (k)', 'y': 'Inertia'})
    fig.show()

    x = np.array([x_k_nr[0], x_k_nr[-1]])  
    y = np.array([y_inertia[0], y_inertia[-1]])
    coefficients = np.polyfit(x, y, 1)
    line = np.poly1d(coefficients)

    a = coefficients[0]
    c = coefficients[1]

    elbow_point = max(range(len(y_inertia)), key=lambda i: abs(y_inertia[i] - line(x_k_nr[i])) / np.sqrt(a**2 + 1)) + 2
    print(f'Optimal value of k according to the elbow method: {elbow_point}')
    
    return elbow_point


def plotter_3d_cluster_with_labels_for_dishes(dataset_org, co_occ_matrix_filtered, dishes_column_name, cluster_label_name, n_top_tokens=5):
    dataset = dataset_org.copy()

    svd_n3 = TruncatedSVD(n_components=3)
    svd_result = svd_n3.fit_transform(co_occ_matrix_filtered)  

    for component in range(3):
        col_name = f"svd_d3_x{component}"
        dataset[col_name] = svd_result[:, component]

    def get_top_n_tokens_for_cluster(cluster_id, n_top_tokens, dataset, cluster_label_name, dishes_column_name):
        cluster_docs = dataset[dataset[cluster_label_name] == cluster_id][dishes_column_name]

        all_dishes = [dish for doc in cluster_docs for dish in doc] 
        dish_counts = Counter(all_dishes)
        top_dishes = [dish for dish, _ in dish_counts.most_common(n_top_tokens)]
        return "_".join(top_dishes)

    cluster_labels = {cluster_id: get_top_n_tokens_for_cluster(cluster_id, n_top_tokens, dataset, cluster_label_name, dishes_column_name)
                      for cluster_id in dataset[cluster_label_name].unique()}

    dataset["cluster_name"] = dataset[cluster_label_name].map(cluster_labels)

    fig = px.scatter_3d(dataset,
                        x='svd_d3_x0',
                        y='svd_d3_x1',
                        z='svd_d3_x2',
                        color="cluster_name", 
                        title=f"{cluster_label_name} Clusters",
                        opacity=0.7,
                        hover_name="cluster_name",  
                        color_discrete_sequence=px.colors.qualitative.Alphabet)

    fig.show()


#Topic Modelling

def train_topic_model(data, model_type='lda', vector_type='bow', num_topics=100, iterations=50):
    
    dictionary = corpora.Dictionary(data["tokenized"])
    corpus = [dictionary.doc2bow(doc) for doc in data["tokenized"]]

    if vector_type == 'tfidf':
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus] 

    if model_type == 'lda':
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=iterations)
    elif model_type == 'lsi':
        model = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    topics = model.print_topics(num_words=5)
    return model, topics, corpus





##### MODELING #####

#Metrics 

def fold_score_calculator(y_pred, y_test, verbose=False):
    #code from classes
    #6. Compute the binary classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    

    if verbose == True:
        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(acc,prec,recall,f1))
    return (acc, prec, recall, f1)


def plot_metrics(ovr_train, ovr_test, cchain_train, cchain_test, model_name="Model"):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    approaches = ['One-vs-Rest Train', 'One-vs-Rest Test', 'Classifier Chain Train', 'Classifier Chain Test']

    data = {
        'Accuracy': [ovr_train[0], ovr_test[0], cchain_train[0], cchain_test[0]],
        'Precision': [ovr_train[1], ovr_test[1], cchain_train[1], cchain_test[1]],
        'Recall': [ovr_train[2], ovr_test[2], cchain_train[2], cchain_test[2]],
        'F1': [ovr_train[3], ovr_test[3], cchain_train[3], cchain_test[3]]}

    x = np.arange(len(metrics_names))  
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#87CEEB', '#4682B4', '#FFB6C1', '#FF69B4']

    for i, approach in enumerate(approaches):
        ax.bar(x + i * bar_width - 1.5 * bar_width, [data[metric][i] for metric in metrics_names],
               bar_width, label=approach, color=colors[i])

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(f"{model_name}'s Performance: Train vs Test for One-vs-Rest and Classifier Chain")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(title="Approach", loc="upper right")

    for i, approach in enumerate(approaches):
        for j, metric in enumerate(metrics_names):
            score = data[metric][i]
            ax.text(j + i * bar_width - 1.5 * bar_width, score + 0.02, f'{score:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()




##### Sentiment Analysis #####
def sent_pre_process_pipeline(raw_text):
    
    clean_text = numbs_to_txt(raw_text)

    clean_text = re.sub(r'\b(?![\W_])\w\b', '', clean_text)
    clean_text = correct_repetition(clean_text)

    pattern = (
        r"(\b[A-Za-z0-9.%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)"     # Email addresses
        r"|(@[A-Za-z0-9]+)"                                          # Social media tags
        r"|(\w+://\S+|http.+?)"                                      # Website links
        r"|(<.*?>)"                                                  # HTML tags
    )

    # Substitute all patterns with a space
    clean_text = re.sub(pattern, " ", clean_text, flags=re.MULTILINE)

    return clean_text
