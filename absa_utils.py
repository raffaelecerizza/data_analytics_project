import re
import ast
import nltk
import spacy
import string
import contractions
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import dataset_utils

from tqdm import tqdm
from afinn import Afinn
from wordcloud import WordCloud
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from itertools import chain

from pyabsa import ATEPCCheckpointManager
from pyabsa import available_checkpoints


ASSETS_PATH = "./assets"
ASUM_BIN = "./asum/bin/"
ASUM_INPUT = "./asum/input/"
ASUM_OUTPUT =  "./asum/output/"



def preprocess_asum_dataset(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We split reviews in sentences.
    new_dataset['sentences'] = new_dataset['reviewText'].progress_map(lambda sentence: sent_tokenize(sentence))
    
    # We preprocess the sentences.
    new_dataset['prep_sentences'] = new_dataset['sentences'].progress_map(lambda text: preprocess_sentences(text))
    
    # We extract all the tokens and we build a dictionary that assigns an id to
    # each unique token.
    all_tokens = list(chain.from_iterable(chain.from_iterable(new_dataset['prep_sentences'])))
    all_tokens = set(all_tokens)
    all_tokens = sorted(all_tokens)
    tokens_dict = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    
    # We convert the tokens in the sentences into the ids.
    new_dataset['prep_sentences_ids'] = new_dataset['prep_sentences'].progress_map(lambda prep_sentences: convert_tokens_in_ids(prep_sentences, tokens_dict))
    
    # We save the WordList for ASUM with the tokens.
    with open(ASUM_INPUT + "WordList.txt", "w") as file:
        for token in all_tokens:
            file.write(token + "\n")
    
    # We save the sentences (converted in token ids) for ASUM.
    prep_sentences_ids = new_dataset['prep_sentences_ids'].values
    with open(ASUM_INPUT + "BagOfSentences.txt", "w") as file:
        for sentences in prep_sentences_ids:
            file.write(str(len(sentences)) + "\n")
            for sentence in sentences:
                file.write(' '.join([str(token_id) for token_id in sentence]) + '\n')
    
    return new_dataset


def preprocess_sentences(sentences):
    prep_sentences = []
    for i in range(len(sentences)):
        prep_sentence = preprocess_asum_text(sentences[i])
        prep_sentences.append(prep_sentence)
    return prep_sentences


def preprocess_review_dataset(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We preprocess the reviews in the dataset.
    new_dataset['preprocessedReviewText'] = new_dataset['reviewText'].progress_map(lambda text: preprocess_text(text))
    
    # We convert the list of tokens in a string text.
    new_dataset['preprocessedReviewText'] = new_dataset['preprocessedReviewText'].apply(lambda tokens: ' '.join(tokens))
    
    # We correct the punctuation.
    new_dataset['preprocessedReviewText'] = new_dataset['preprocessedReviewText'].apply(correct_punctuation)
    
    return new_dataset


def preprocess_summary_dataset(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We preprocess the reviews in the dataset.
    new_dataset['preprocessedSummary'] = new_dataset['summary'].progress_map(lambda text: preprocess_text(text))
    
    # We convert the list of tokens in a string text.
    new_dataset['preprocessedSummary'] = new_dataset['preprocessedSummary'].apply(lambda tokens: ' '.join(tokens))
    
    # We correct the punctuation.
    new_dataset['preprocessedSummary'] = new_dataset['preprocessedSummary'].apply(correct_punctuation)
    
    return new_dataset


def preprocess_text(text):    
    # We expand contractions. Example: don't -> do not.
    expanded_text = contractions.fix(text)
    
    # We tokenize the text.
    tokens = word_tokenize(expanded_text)

    # We convert the tokens to lower case.
    #tokens = [token.lower() for token in tokens]
    
    # We remove repetitions.
    tokens = [remove_repetitions(token) for token in tokens]
    
    # We remove stop words.
    stop_words = ["nbsp", "ie=utf8", "data-hook=",
                  "product-link-linked", "a-link-normal", "href=",
                  "class="]
    stop_words = set(stop_words)
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens


def preprocess_asum_text(text):
    # We expand contractions. Example: don't -> do not.
    expanded_text = contractions.fix(text)
    
    # We tokenize the text.
    tokens = word_tokenize(expanded_text)

    # We convert the tokens to lower case.
    tokens = [token.lower() for token in tokens]
    
    # We remove repetitions.
    tokens = [remove_repetitions(token) for token in tokens]
    
    # We specify the punctuation.
    punctuation = string.punctuation
    
    # We remove the punctuation.
    tokens = [token for token in tokens if token not in punctuation]
    
    # We use the Snowball Stemmer.
    snow_stemmer = SnowballStemmer(language='english')
    tokens = [snow_stemmer.stem(token) for token in tokens]
    
    # We specify negations.
    negations = ['not']
    negations_set = set(negations)
    tokens = connect_not(tokens)
    
    # Lambda function to check if the token contains numbers.
    contains_number = lambda x: not re.search(r'\d', x)
    
    # Lambda function to check if the token contains only alphabetic characters.
    is_alpha = lambda x: all(c.isalpha() for c in x)
    
    # We remove stop words, numbers and single characters.
    stop_words = set(stopwords.words("english"))
    #stop_words.difference_update(negations_set)
    stop_words_to_add = ["\'s", "n\'t", "``", "\'ve", "\'m", "--", "\'re",
                        "\'ll", "\'d", "nbsp", "ie=utf8", "data-hook=",
                        "product-link-linked", "a-link-normal", "href=",
                        "class=", "/a", "and/or", "..", "...", "....",
                        ".....", "\'\'"]
    for new_stop_word in stop_words_to_add:
        stop_words.add(new_stop_word)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if is_alpha(token)] 
    tokens = [token for token in tokens if contains_number(token) and len(token) > 1] 

    return tokens


def remove_repetitions(word):
    # We remove characters that are repeated more than 2 times in a word.
    # Example: "waaaaayyyy" to "waayy".
    correct_word = re.sub(r'(.)\1+', r'\1\1', word)
    return correct_word


def correct_punctuation(text):
    # We remove incorrect space before punctuation.
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace(' :', ':')
    text = text.replace(' ;', ';')
    text = text.replace(' %', '%')
    text = text.replace(' \'', '\'')
    # We remove incorrect space for brackets.
    text = text.replace('( ', '(')
    text = text.replace(' )', ')')
    text = text.replace('[ ', '[')
    text = text.replace(' ]', ']')
    return text


def connect_not(tokens):
    connected_tokens = []
    i = 0

    while i < len(tokens):
        if tokens[i] == "not" and i + 1 < len(tokens):
            connected_tokens.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            connected_tokens.append(tokens[i])
            i += 1

    return connected_tokens


def convert_tokens_in_ids(prep_sentences, tokens_dict):
    converted_sentences = []
    for sublist in prep_sentences:
        converted_sentence = []
        for token in sublist:
            converted_sentence.append(tokens_dict[token])
        converted_sentences.append(converted_sentence)
    return converted_sentences


def get_reviews_aspects(dataset):
    new_dataset = dataset.copy()
    
    # We get the preprocessed reviews as a list.
    reviews = dataset['preprocessedReviewText'].values
    reviews = list(reviews)
    
    # We load the PyABSA model.
    # ATEPC = Aspect Term Extraction and Sentiment Classification
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='english', 
        auto_device=True  # False means load model on CPU
    )
    
    # We extract the aspects and their sentiment.
    atepc_result = aspect_extractor.extract_aspect(
        inference_source=reviews,  
        pred_sentiment=True,  # predict the sentiment of extracted aspect terms
    )
    
    # We collect all the aspects and sentiments.
    aspects = []
    sentiments = []
    for i in range(len(atepc_result)):
        aspects.append(atepc_result[i]['aspect'])
        sentiments.append(atepc_result[i]['sentiment'])
        
    # We add specific columns for aspects and sentiments.
    new_dataset['aspects'] = aspects
    new_dataset['sentiments'] = sentiments
    
    return new_dataset


def plot_most_frequent_aspects(dataset, k=20):
    new_dataset = dataset.copy()
    
    # The list of aspects is imported as a string. We convert it into a list of aspects.
    new_dataset['aspects'] = new_dataset['aspects'].apply(ast.literal_eval)
    
    # We get the aspects of the dataset.
    aspects = new_dataset['aspects'].values
    
    # We keep only one list with all the aspects.
    flat_list = [aspect for sublist in aspects for aspect in sublist]
    
    # We count the frequency of each aspect.
    aspect_frequency = Counter(flat_list)
    
    # We get the most common aspects.
    top_aspects = dict(aspect_frequency.most_common(k))
    df_top_aspects = pd.DataFrame(list(top_aspects.items()), columns=['Aspect', 'Frequency'])

    # We create a histogram plot with aspects and frequencies.
    histogram = px.bar(
        df_top_aspects, 
        x='Aspect', 
        y='Frequency', 
        title=f'Top {k} most frequent aspects',
        color_discrete_sequence=['skyblue']
    )
    histogram.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array'),  
        showlegend=False, 
    )
    histogram.write_image(f'{ASSETS_PATH}/most_frequent_aspects.png')
    histogram.show()
    
    
def plot_top_and_bottom_aspects(dataset, k=20):
    new_dataset = dataset.copy()
    
    # We compute a dictionary for the relevant information for each aspect.
    result_dict = compute_aspects_dictionary(new_dataset)
        
    # We create a dataframe from the dictionary.    
    aspects_df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    # We print the quartiles for the frequency of the aspects.
    first_quartile = np.percentile(aspects_df['Frequency'], 25)
    second_quartile = np.percentile(aspects_df['Frequency'], 50)
    third_quartile = np.percentile(aspects_df['Frequency'], 75)
    print(f"First quartile: {first_quartile}")
    print(f"Second quartile: {second_quartile}")
    print(f"Third quartile: {third_quartile}")

    # We print the 95 percentile.
    ninetyfive_percentile = np.percentile(aspects_df['Frequency'], 95)
    print(f"Percentile 95: {ninetyfive_percentile}")
    
    # We keep only the aspects of the specified percentile.
    filtered_aspects_df = aspects_df[aspects_df['Frequency'] > ninetyfive_percentile]
    
    # We sort the dataframe by the mean sentiment.
    filtered_aspects_df = filtered_aspects_df.sort_values(by='Mean', ascending=False)
        
    # We plot the top aspects by mean sentiment.
    top_aspects = filtered_aspects_df.head(k)
    fig = px.bar(
        top_aspects, 
        x=top_aspects.index, 
        y='Mean', 
        title=f'Top {k} aspects by mean sentiment',
        labels={'index': 'Aspect', 'Mean': 'Mean sentiment'},
        color_discrete_sequence=['mediumseagreen']
    )
    fig.update_layout(
        xaxis_title='Aspect', 
        yaxis_title='Mean sentiment'
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array'),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/top_aspects_by_sentiment.png')
    fig.show()
    
    # We plot the bottom aspects by mean sentiment.
    bottom_aspects = filtered_aspects_df.tail(k)
    bottom_aspects = bottom_aspects.sort_values(by='Mean', ascending=True)
    fig = px.bar(
        bottom_aspects, 
        x=bottom_aspects.index, 
        y='Mean', 
        title=f'Bottom {k} aspects by mean sentiment',
        labels={'index': 'Aspect', 'Mean': 'Mean sentiment'},
        color_discrete_sequence=['salmon']
    )
    fig.update_layout(
        xaxis_title='Aspect', 
        yaxis_title='Mean sentiment'
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array'),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/bottom_aspects_by_sentiment.png')
    fig.show()
    
    return aspects_df, filtered_aspects_df


def compute_aspects_dictionary(dataset):
    new_dataset = dataset.copy()
    
    # The list of aspects is imported as a string. We convert it into a list of aspects.
    new_dataset['aspects'] = new_dataset['aspects'].apply(ast.literal_eval)
        
    # We get the aspects of the dataset.
    aspects = new_dataset['aspects'].values
    
    # The list of sentiments is imported as a string. We convert it into a list of sentiments.
    new_dataset['sentiments'] = new_dataset['sentiments'].apply(ast.literal_eval)

    # We get the sentiments of the aspects.
    sentiments = new_dataset['sentiments'].values
    
    # We convert sentiments to numbers.
    sentiments = [[1 if sentiment == 'Positive' else -1 if sentiment == 'Negative' else 0 for sentiment in sublist] for sublist in sentiments]
    
    # We keep only one list with all the aspects.
    aspects_flat_list = [aspect for sublist in aspects for aspect in sublist]
        
    # We count the frequency of each aspect.
    aspect_frequency = Counter(aspects_flat_list)
    
    # We keep only one list with all the sentiments.
    sentiments_flat_list = [sentiment for sublist in sentiments for sentiment in sublist]
    
    # We create lists for aspects and their frequency.
    aspects_names = []
    aspects_counts = []
    for aspect in aspect_frequency.keys():
        aspects_names.append(aspect)
        aspects_counts.append(aspect_frequency[aspect])
        
    # For each aspect we build a dictionary with:
    #   - frequency;
    #   - sum of sentiments;
    #   - mean sentiment;
    #   - standard deviation of sentiments;
    #   - sentiments list.
    result_dict = {}
    for i in range(len(aspects_names)):
        aspect = aspects_names[i]
        aspect_dict = {'Frequency': aspects_counts[i], 'Sum': 0, 'Mean': 0, 'Std': 0, 'Sentiments': []}
        result_dict[aspect] = aspect_dict
    
    for i in range(len(aspects_flat_list)):
        aspect = aspects_flat_list[i]
        sentiment = sentiments_flat_list[i]
        sum = result_dict[aspect]['Sum']
        sum += sentiment
        result_dict[aspect]['Sum'] = sum
        result_dict[aspect]['Sentiments'].append(sentiment)
        
    for aspect in aspects_names:
        sentiments = result_dict[aspect]['Sentiments']
        mean = np.mean(sentiments)
        std = np.std(sentiments)
        result_dict[aspect]['Mean'] = mean
        result_dict[aspect]['Std'] = std
        
    return result_dict
        
    
def plot_aspect_sentiment_in_time(dataset, aspect):
    new_dataset = dataset.copy()
    
    # We filter the dataset in order to consider only the reviews with the specific aspect.
    filtered_df = new_dataset[new_dataset['aspects'].apply(lambda x: aspect in eval(x))]
    
    # We add the year attribute.
    filtered_df = filtered_df.copy()
    filtered_df['reviewTime'] = pd.to_datetime(filtered_df['reviewTime'])
    filtered_df['year'] = filtered_df['reviewTime'].dt.year
    
    # We find the years where the aspect is present.
    years = pd.unique(filtered_df['year'].values)
    years = np.sort(years)
    
    # For each year we compute the mean sentiment with standard deviation.
    year_list, mean_list, std_list = [], [], []
    for year in years:
        # We compute a dictionary for the relevant information for each aspect.
        year_df = filtered_df[filtered_df['year'] == year]  
        result_dict = compute_aspects_dictionary(year_df)
        #print("result_dict: ", result_dict)
        year_list.append(year)
        mean_list.append(result_dict[aspect]['Mean'])
        std_list.append(result_dict[aspect]['Std'])
        
    # Now we plot the data with an error plot.
    aspect_df = pd.DataFrame({'Year': year_list, 'Mean': mean_list, 'Std': std_list})
    aspect_df = aspect_df.copy()
    aspect_df['Year'] = aspect_df['Year'].astype(int)
    fig = px.line(
        aspect_df, 
        x='Year', 
        y='Mean', 
        error_y='Std', 
        title=f'Mean sentiment for {aspect} for each year',
        labels={'Mean': 'Mean sentiment', 'Year': 'Year'})
    fig.update_traces(error_y_thickness=0.8)
    fig.show()  
        
    
def plot_price_information(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    aspects = ['bargain', 'convenience', 'cost', 'costs', 'deal', 'discount', 'expensive', 'money', 'price', 'priced', 'prices', 'pricey', 'pricing']
    
    filtered_df = new_dataset[new_dataset['aspects'].apply(lambda x: any(aspect in eval(x) for aspect in aspects))]
    
    result_dict = compute_aspects_dictionary(filtered_df)
    
    frequencies = []
    sums = []
    sentiments = []
    for i in range(len(aspects)):
        print("Aspect: ", aspects[i])
        print("Frequency: ", result_dict[aspects[i]]['Frequency'])
        print("Sum: ", result_dict[aspects[i]]['Sum'])
        print("Mean: ", result_dict[aspects[i]]['Mean'])
        print("Std: ", result_dict[aspects[i]]['Std'])
        print()
        frequencies.append(result_dict[aspects[i]]['Frequency'])
        sums.append(result_dict[aspects[i]]['Sum'])
        sentiments.extend(result_dict[aspects[i]]['Sentiments'])
        
    print("Total frequencies: ", np.sum(frequencies))
    print("Mean sentiments: ", np.mean(sentiments))
    print("Std sentiments: ", np.std(sentiments))
        
    aspects = ['bargain', 'convenience', 'cost', 'deal', 'discount', 'expensive', 'money', 'price']
    
    # We add the year attribute.
    filtered_df = filtered_df.copy()
    filtered_df['reviewTime'] = pd.to_datetime(filtered_df['reviewTime'])
    filtered_df['year'] = filtered_df['reviewTime'].dt.year

    # We find the years where the aspect is present.
    years = pd.unique(filtered_df['year'].values)
    years = np.sort(years)
    
    total_aspect_list = []
    total_year_list = []
    total_mean_list = []
    total_std_list = []
    # For each aspect we find the relevant information.
    for aspect in aspects:
        # For each year we compute the mean sentiment with standard deviation.
        year_list, mean_list, std_list, aspect_list = [], [], [], []
        for year in years:
            # We compute a dictionary for the relevant information for each aspect.
            year_df = filtered_df[filtered_df['year'] == year]  
            result_dict = compute_aspects_dictionary(year_df)
            if aspect in result_dict:
                aspect_list.append(aspect)
                year_list.append(year)
                mean_list.append(result_dict[aspect]['Mean'])
                std_list.append(result_dict[aspect]['Std'])
        total_aspect_list.append(aspect_list)
        total_year_list.append(year_list)
        total_mean_list.append(mean_list)
        total_std_list.append(std_list)
        
    flat_total_aspect_list = [value for sublist in total_aspect_list for value in sublist]
    flat_total_year_list = [value for sublist in total_year_list for value in sublist]
    flat_total_mean_list = [value for sublist in total_mean_list for value in sublist]
    flat_total_std_list = [value for sublist in total_std_list for value in sublist]
    
    aspect_df = pd.DataFrame({'Aspect': flat_total_aspect_list, 
                          'Year': flat_total_year_list, 
                          'Mean': flat_total_mean_list, 
                          'Std': flat_total_std_list})
    fig = px.line(aspect_df, 
              x='Year', 
              y='Mean', 
              color='Aspect',
              error_y='Std', 
              title='Mean sentiment for each aspect for each year',
              labels={'Mean': 'Mean sentiment', 'Year': 'Year'})

    fig.update_traces(error_y_thickness=0.8)
    fig.show()
    
    
    
    
    
    
    
