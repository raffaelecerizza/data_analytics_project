import re
import nltk
import spacy
import string
import contractions
import numpy as np
import pandas as pd
import seaborn as sns
import scattertext as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dataset_utils

from tqdm import tqdm
from afinn import Afinn
from wordcloud import WordCloud
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ASSETS_PATH = "./assets"


def preprocess_review_dataset(dataset, negation_handling=False, lemmatization=False, vader=False):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We preprocess the reviews in the dataset.
    #new_dataset['preprocessedReviewText'] = new_dataset['reviewText'].progress_apply(preprocess_text)
    new_dataset['preprocessedReviewText'] = new_dataset['reviewText'].progress_map(lambda text: preprocess_text(text, negation_handling, lemmatization, vader))
    
    # We convert the list of tokens in a string text.
    new_dataset['preprocessedReviewText'] = new_dataset['preprocessedReviewText'].apply(lambda tokens: ' '.join(tokens))
    
    return new_dataset


def preprocess_summary_dataset(dataset, negation_handling=False, lemmatization=False, vader=False):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We preprocess the summaries in the dataset.
    #new_dataset['preprocessedSummary'] = new_dataset['summary'].progress_apply(preprocess_text)
    new_dataset['preprocessedSummary'] = new_dataset['summary'].progress_map(lambda text: preprocess_text(text, negation_handling, lemmatization, vader))
    
    # We convert the list of tokens in a string text.
    new_dataset['preprocessedSummary'] = new_dataset['preprocessedSummary'].apply(lambda tokens: ' '.join(tokens))
    
    return new_dataset


def preprocess_text(text, negation_handling=False, lemmatization=False, vader=False):
    # We expand contractions. Example: don't -> do not.
    expanded_text = contractions.fix(text)
    
    # We tokenize the text.
    tokens = word_tokenize(expanded_text)

    if vader == True:
        # We remove repetitions.
        tokens = [remove_repetitions(token) for token in tokens]
        
        # We remove stop words.
        stop_words = ["nbsp", "ie=utf8", "data-hook=",
                    "product-link-linked", "a-link-normal", "href=",
                    "class="]
        stop_words = set(stop_words)
        tokens = [token for token in tokens if token not in stop_words]
    else:
    
        # We convert the tokens to lower case.
        tokens = [token.lower() for token in tokens]
        
        # We remove repetitions.
        tokens = [remove_repetitions(token) for token in tokens]

        # We specify the punctuation.
        punctuation = string.punctuation
        
        # We specify negations.
        negations = ['not', 'nor', 'against', 'no', 'never']
        negations_set = set(negations)

        # If required, we handle negations with the NOT_ prefix.
        if negation_handling:
            tokens = handle_negations(tokens, negations, punctuation)

        # We remove the punctuation.
        tokens = [token for token in tokens if token not in punctuation]

        # Lambda function to check if the token contains numbers.
        contains_number = lambda x: not re.search(r'\d', x)
        
        # Lambda function to check if the token contains only alphabetic characters.
        is_alpha = lambda x: all(c.isalpha() for c in x)
        
        # We remove stop words, numbers and single characters.
        stop_words = set(stopwords.words("english"))
        stop_words.difference_update(negations_set)
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
    
        # We lemmatize the tokens.
        if lemmatization == True:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def remove_repetitions(word):
    # We remove characters that are repeated more than 2 times in a word.
    # Example: "waaaaayyyy" to "waayy".
    correct_word = re.sub(r'(.)\1+', r'\1\1', word)
    return correct_word


def handle_negations(tokens, negations, punctuation):
    result = []
    add_not_prefix = False
    for token in tokens:
        if token in negations:
            add_not_prefix = True
        else:
            if token in punctuation:
                add_not_prefix = False

            if add_not_prefix:
                result.append("NOT_" + token)
            else:
                result.append(token)
    return result


def compute_review_afinn_scores(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We compute the Afinn scores.
    afinn = Afinn()
    negations = ['not', 'nor', 'against', 'no', 'never']
    #new_dataset['afinn'] = new_dataset['preprocessedReviewText'].progress_apply(afinn.score)
    new_dataset['afinn'] = new_dataset['preprocessedReviewText'].progress_map(lambda text: compute_afinn_score(afinn, text, negations))
    
    # We plot a histogram of the Afinn scores.
    histogram = px.histogram(
        new_dataset, 
        x='afinn',
        title='Histogram of Afinn scores for reviews',
        labels={'afinn': 'Afinn score', 'count': 'Number of reviews'},
        color_discrete_sequence=['mediumseagreen'],
        nbins=len(new_dataset['afinn'].value_counts()),
        opacity=0.7,
        width=800,
        height=600
    )
    histogram.update_traces(
        marker_line_color='black', 
        marker_line_width=1, 
    )
    histogram.update_layout(
        xaxis_title='Afinn score',
        yaxis_title='Number of reviews',
    )
    histogram.write_image(f'{ASSETS_PATH}/reviews_afinn_scores.png')
    histogram.show()

    # We replace the opinion value with a number.
    new_dataset['opinion'] = new_dataset['opinion'].replace({'negative': -1, 'neutral': 0, 'positive': 1})
    
    # We plot the confusion matrix.
    y_true = np.sign(new_dataset.opinion)
    y_pred = np.sign(new_dataset.afinn).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    
    # We print classification metrics.
    print_classification_metrics(y_true, y_pred)
    
    return new_dataset


def compute_summary_afinn_scores(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We compute the Afinn scores.
    afinn = Afinn()
    negations = ['not', 'nor', 'against', 'no', 'never']
    #new_dataset['afinn'] = new_dataset['preprocessedSummary'].progress_apply(afinn.score)
    new_dataset['afinn'] = new_dataset['preprocessedSummary'].progress_map(lambda text: compute_afinn_score(afinn, text, negations))
    
    # We plot a histogram of the Afinn scores.
    # We save the plotly.express plot.
    histogram = px.histogram(
        new_dataset, 
        x='afinn',
        title='Histogram of Afinn scores for summaries',
        labels={'afinn': 'Afinn score', 'count': 'Number of summaries'},
        color_discrete_sequence=['mediumseagreen'],
        nbins=len(new_dataset['afinn'].value_counts()),
        opacity=0.7,
        width=800,
        height=600
    )
    histogram.update_traces(
        marker_line_color='black', 
        marker_line_width=1, 
    )
    histogram.update_layout(
        xaxis_title='Afinn score',
        yaxis_title='Number of summaries',
    )
    histogram.write_image(f'{ASSETS_PATH}/summaries_afinn_scores.png')
    histogram.show()
    
    # We replace the opinion value with a number.
    new_dataset['opinion'] = new_dataset['opinion'].replace({'negative': -1, 'neutral': 0, 'positive': 1})
    
    # We plot the confusion matrix.
    y_true = np.sign(new_dataset.opinion)
    y_pred = np.sign(new_dataset.afinn).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    
    # We print classification metrics.
    print_classification_metrics(y_true, y_pred)

    return new_dataset


def compute_afinn_score(afinn, text, negations):
    afinn_score = []

    tokens = word_tokenize(text)

    invert_sign = False

    for token in tokens:
        # If there is a negation, the next word has it value inverted.
        if token in negations:
            invert_sign = True
        else:
            score = afinn.score(token)
            if invert_sign:
                score *= -1  
                invert_sign = False  
            afinn_score.append(score)

    return np.sum(afinn_score)
    
    
def compute_review_vader_scores(dataset, preprocessed=True):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We compute the VADER scores.
    analyzer = SentimentIntensityAnalyzer()
    if preprocessed == True:
        new_dataset['vader'] = new_dataset['preprocessedReviewText'].progress_map(lambda text: compute_vader_score(analyzer, text))
    else:
        new_dataset['vader'] = new_dataset['reviewText'].progress_map(lambda text: compute_vader_score(analyzer, text))
    
    # We replace the opinion value with a number.
    new_dataset['opinion'] = new_dataset['opinion'].replace({'negative': -1, 'neutral': 0, 'positive': 1})
    
    # We plot the confusion matrix.
    y_true = np.sign(new_dataset.opinion)
    y_pred = np.sign(new_dataset.vader).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    
    # We print classification metrics.
    print_classification_metrics(y_true, y_pred)
    
    return new_dataset


def compute_summary_vader_scores(dataset, preprocessed=True):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We compute the VADER scores.
    analyzer = SentimentIntensityAnalyzer()
    if preprocessed == True:
        new_dataset['vader'] = new_dataset['preprocessedSummary'].progress_map(lambda text: compute_vader_score(analyzer, text))
    else:
        new_dataset['vader'] = new_dataset['summary'].progress_map(lambda text: compute_vader_score(analyzer, text))
    
    # We replace the opinion value with a number.
    new_dataset['opinion'] = new_dataset['opinion'].replace({'negative': -1, 'neutral': 0, 'positive': 1})
    
    # We plot the confusion matrix.
    y_true = np.sign(new_dataset.opinion)
    y_pred = np.sign(new_dataset.vader).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    
    # We print classification metrics.
    print_classification_metrics(y_true, y_pred)
    
    return new_dataset
    
    
def compute_vader_score(analyzer, text):
    sentiment_dict = analyzer.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        return 1    # positive
    elif sentiment_dict['compound'] <= - 0.05:
        return -1   # negative
    else:
        return 0    # neutral
    
    
def plot_confusion_matrix(y_true, y_pred):
    label_mapping = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    y_true_mapped = [label_mapping[label] for label in y_true]
    y_pred_mapped = [label_mapping[label] for label in y_pred]
    confusion_matrix = pd.crosstab(y_pred_mapped, y_true_mapped)
    #confusion_matrix = pd.crosstab(y_pred, y_true)
    #row_sums = confusion_matrix.sum(axis=1)
    #normalized_confusion_matrix = confusion_matrix.div(row_sums, axis=0)
    #normalized_confusion_matrix = confusion_matrix / np.sum(confusion_matrix.values)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d',
        cmap='Reds', 
        cbar=True,
        annot_kws={"size": 16})
    plt.xlabel('True labels', fontsize=16)
    plt.ylabel('Predicted labels', fontsize=16)
    plt.title('Confusion matrix', fontsize=18)
    plt.show()
    
    
def print_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive'], digits=3)
    print("Accuracy: ", accuracy)
    print("Classification report: ")
    print(report)

    
def create_reviews_scattertext(dataset, year):
    new_dataset = dataset.copy()
    
    # We binarize opinions:
    #   - positive opinions: ratings >= 4;
    #   - negative opinions: ratings < 4.
    new_dataset['opinion_binary'] = new_dataset['rating'] >= 4
    new_dataset['opinion_binary'] = new_dataset['opinion_binary'].astype('category')
    new_dataset.replace({'opinion_binary': {True: 'Positive', False: 'Negative'}}, inplace = True)
    
    # We create the scattertext for the specified year.
    nlp = spacy.load("en_core_web_sm")
    print("Started scattertext")
    create_scattertext(new_dataset, nlp, year)
    print("Finished scattertext")
    
    
def create_scattertext(dataset, nlp, year):
    # We filter the dataset with the relevant year.
    year_dataset = dataset[pd.DatetimeIndex(dataset['reviewTime']).year == year]
  
    # We create the corpus.
    corpus = st.CorpusFromPandas(year_dataset, 
                                 category_col='opinion_binary', 
                                 text_col="preprocessedReviewText",
                                 #text_col="reviewText",
                                 nlp=nlp).build()

    # We create the HTML scattertext.
    html = st.produce_scattertext_explorer(corpus,
                                           category='Positive',
                                           category_name='Positive opinion',
                                           not_category_name='Negative opinion',
                                           width_in_pixels=1000,
                                           metadata=year_dataset["rating"])
    open("./scattertexts/year_" + str(year) + ".html", 'wb').write(html.encode('utf-8'))
    
    
def get_train_test_reviews(dataset, undersampling=False, tfidf=False, split=0.8):
    # We get the preprocessed reviews and the opinions.
    preprocessed_reviews = dataset['preprocessedReviewText'].values
    opinions = dataset['opinion'].values
    
    # We compute the Bag of Words.
    #count_vectorizer = CountVectorizer(stop_words=None, lowercase=True, max_features=15000)
    count_vectorizer = CountVectorizer(stop_words=None, lowercase=True)
    bow = count_vectorizer.fit_transform(preprocessed_reviews)
    
    '''
    # We split between train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        bow.toarray(),
        dataset.opinion,
        train_size=split,
        random_state=1)
    '''
    
    # We shuffle the indices of the dataset.
    indices = np.arange(len(preprocessed_reviews))
    np.random.seed(1)
    np.random.shuffle(indices)
    
    # We split between train and test.
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    bow = bow[indices]
    X_train = bow[:train_size]
    X_test = bow[train_size:]
    opinions = opinions[indices]
    y_train = opinions[:train_size]
    y_test = opinions[train_size:]
    
    # If required, we undersample all the classes to the number of instances
    # of the minority class.
    if undersampling == True:
        undersampler = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        
    # If required, we apply TF-IDF to the train set.
    if tfidf == True:
        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_train = tfidf_transformer.fit_transform(X_train)  
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def get_train_test_summaries(dataset, undersampling=False, tfidf=False, split=0.8):
    # We get the preprocessed summaries and the opinions.
    preprocessed_summaries = dataset['preprocessedSummary'].values
    opinions = dataset['opinion'].values
    
    # We compute the Bag of Words.
    #count_vectorizer = CountVectorizer(stop_words=None, lowercase=True, max_features=15000)
    count_vectorizer = CountVectorizer(stop_words=None, lowercase=True)
    bow = count_vectorizer.fit_transform(preprocessed_summaries)
    
    '''
    # We split between train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        bow.toarray(),
        dataset.opinion,
        train_size=split,
        random_state=1)
    '''
    
    # We shuffle the indices of the dataset.
    indices = np.arange(len(preprocessed_summaries))
    np.random.seed(1)
    np.random.shuffle(indices)
    
    # We split between train and test.
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    bow = bow[indices]
    X_train = bow[:train_size]
    X_test = bow[train_size:]
    opinions = opinions[indices]
    y_train = opinions[:train_size]
    y_test = opinions[train_size:]
    
    # If required, we undersample all the classes to the number of instances
    # of the minority class.
    if undersampling == True:
        undersampler = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        
    # If required, we apply TF-IDF to the train set.
    if tfidf == True:
        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_train = tfidf_transformer.fit_transform(X_train)
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices
    

def train_logistic_regression_model(X_train, y_train, iterations=100):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=iterations)  
    model = model.fit(X=X_train, y=y_train)
    return model


def test_logistic_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    return y_pred


def cross_validation_dataset(dataset, reviews=True, undersampling=False, tfidf=False, iterations=1000):
    new_dataset = dataset.copy()
    
    # We get the preprocessed summaries and the opinions.
    if reviews:
        preprocessed_reviews = new_dataset['preprocessedReviewText'].values
    else:
        preprocessed_summaries = dataset['preprocessedSummary'].values
    opinions = new_dataset['opinion'].values
    
    # We compute the Bag of Words.
    count_vectorizer = CountVectorizer(stop_words=None, lowercase=True)
    if reviews:
        bow = count_vectorizer.fit_transform(preprocessed_reviews)
    else:
        bow = count_vectorizer.fit_transform(preprocessed_summaries)
    
    # We shuffle the indices of the dataset.
    row_indices = np.arange(bow.shape[0])
    np.random.seed(1)
    np.random.shuffle(row_indices)
    
    # We split the indices in 3 sets. 
    fold_size = bow.shape[0] // 3
    fold1_indices = row_indices[:fold_size]
    fold2_indices = row_indices[fold_size: 2 * fold_size]
    fold3_indices = row_indices[2 * fold_size:]
    
    # We extract features and labels for each set.
    X_fold1 = bow[fold1_indices, :]
    X_fold2 = bow[fold2_indices, :]
    X_fold3 = bow[fold3_indices, :]
    y_fold1 = opinions[fold1_indices]
    y_fold2 = opinions[fold2_indices]
    y_fold3 = opinions[fold3_indices]
    
    # We build the train and test data for each cross validation.
    X_train_1 = vstack([X_fold1, X_fold2])
    X_train_2 = vstack([X_fold1, X_fold3])
    X_train_3 = vstack([X_fold2, X_fold3])
    X_test_1 = X_fold3.copy()
    X_test_2 = X_fold2.copy()
    X_test_3 = X_fold1.copy()
    y_train_1 = np.vstack((y_fold1, y_fold2)).reshape(-1)
    y_train_2 = np.vstack((y_fold1, y_fold3)).reshape(-1)
    y_train_3 = np.vstack((y_fold2, y_fold3)).reshape(-1)
    y_test_1 = y_fold3.copy()
    y_test_2 = y_fold2.copy()
    y_test_3 = y_fold1.copy()
    
    # If required, we undersample all the classes to the number of instances
    # of the minority class.
    if undersampling == True:
        undersampler = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
        X_train_1, y_train_1 = undersampler.fit_resample(X_train_1, y_train_1)
        X_train_2, y_train_2 = undersampler.fit_resample(X_train_2, y_train_2)
        X_train_3, y_train_3 = undersampler.fit_resample(X_train_3, y_train_3)
        
    # If required, we apply TF-IDF to the train set.
    if tfidf == True:
        tfidf_transformer = TfidfTransformer(use_idf=True)
        X_train_1 = tfidf_transformer.fit_transform(X_train_1)
        X_train_2 = tfidf_transformer.fit_transform(X_train_2)
        X_train_3 = tfidf_transformer.fit_transform(X_train_3)
    
    # We train and test a logistic regression model for each cross validation.
    model1 = train_logistic_regression_model(X_train_1, y_train_1, iterations=iterations)
    y_pred_1 = test_logistic_regression_model(model1, X_test_1, y_test_1)
    model2 = train_logistic_regression_model(X_train_2, y_train_2, iterations=iterations)
    y_pred_2 = test_logistic_regression_model(model2, X_test_2, y_test_2)
    model3 = train_logistic_regression_model(X_train_3, y_train_3, iterations=iterations)
    y_pred_3 = test_logistic_regression_model(model3, X_test_3, y_test_3)
    
    # We print the classification metrics.
    y_true = np.vstack((y_test_1, y_test_2, y_test_3)).reshape(-1)
    y_pred = np.vstack((y_pred_1, y_pred_2, y_pred_3)).reshape(-1)
    print_classification_metrics(y_true, y_pred)
    
    # Test indices start from the third fold since we use the first two folds
    # for the training in the cross validation. And so on.
    test_indices = np.vstack((fold3_indices, fold2_indices, fold1_indices)).reshape(-1)
    
    return y_true, y_pred, test_indices
    
    
def print_mismatch_examples(dataset, y_true, y_pred, test_indices, num_examples=5):
    # We get opinions, ratings and reviews of the whole dataset.
    opinions = dataset['opinion'].values
    ratings = dataset['rating'].values
    reviews = dataset['reviewText'].values
    
    # We get opinions of the test set.
    test_opinions = opinions[test_indices]
    
    # We get the indices of mismatches between test opinions and predictions.
    different_sentiment_indices = np.where(test_opinions != y_pred)[0]
    different_sentiment_test_indices = test_indices[different_sentiment_indices]
    
    # We get opinions, ratings and reviews of the mismatches.
    different_opinions = opinions[different_sentiment_test_indices]
    different_ratings = ratings[different_sentiment_test_indices]
    different_reviews = reviews[different_sentiment_test_indices]
    
    # We also get the predicted sentiments for the mismatches.
    different_predicted_sentiments = y_pred[different_sentiment_indices]
    
    # We create dictionaries for the mismatches.
    dict_negative_mismatches = {
        'rating': [],           
        'opinion': [], 
        'pred_sentiment': [],       
        'review': []          
    }
    dict_neutral_mismatches = {
        'rating': [],           
        'opinion': [], 
        'pred_sentiment': [],       
        'review': []          
    }
    dict_positive_mismatches = {
        'rating': [],           
        'opinion': [], 
        'pred_sentiment': [],       
        'review': []          
    }
    
    # We count the number of mismatches for each opinion.
    count_negatives = 0
    count_neutrals = 0
    count_positives = 0
    
    # We fill the dictionaries.
    for i in range(len(different_sentiment_indices)):
        if different_opinions[i] == 'negative' and count_negatives < num_examples:
            dict_negative_mismatches['rating'].append(different_ratings[i])
            dict_negative_mismatches['opinion'].append(different_opinions[i])
            dict_negative_mismatches['pred_sentiment'].append(different_predicted_sentiments[i])
            dict_negative_mismatches['review'].append(different_reviews[i])
            count_negatives += 1
        if different_opinions[i] == 'neutral' and count_neutrals < num_examples:
            dict_neutral_mismatches['rating'].append(different_ratings[i])
            dict_neutral_mismatches['opinion'].append(different_opinions[i])
            dict_neutral_mismatches['pred_sentiment'].append(different_predicted_sentiments[i])
            dict_neutral_mismatches['review'].append(different_reviews[i])
            count_neutrals += 1
        if different_opinions[i] == 'positive' and count_positives < num_examples:
            dict_positive_mismatches['rating'].append(different_ratings[i])
            dict_positive_mismatches['opinion'].append(different_opinions[i])
            dict_positive_mismatches['pred_sentiment'].append(different_predicted_sentiments[i])
            dict_positive_mismatches['review'].append(different_reviews[i])
            count_positives += 1
    
    # We print the contents of the dictionaries.
    print("Examples of negative mismatches: ")
    for i in range(num_examples):
        for key, values in dict_negative_mismatches.items():
            print(f"{key}: {values[i]}")
        print("\n")
        
    print("Examples of neutral mismatches: ")
    for i in range(num_examples):
        for key, values in dict_neutral_mismatches.items():
            print(f"{key}: {values[i]}")
        print("\n")
    
    print("Examples of positive mismatches: ")
    for i in range(num_examples):
        for key, values in dict_positive_mismatches.items():
            print(f"{key}: {values[i]}")
        print("\n")


def save_dataset_with_sentiment(dataset, y_pred, test_indices, DATASET_WITH_SENTIMENT_PATH):
    new_dataset = dataset.copy()
    
    # We reorder the dataset with test indices.
    new_dataset = new_dataset.iloc[test_indices]
    
    # We add a new column for the predicted sentiment.
    new_dataset['predictedSentiment'] = y_pred
    
    # We convert the predicted sentiment to a number:
    #   - negative: -1;
    #   - neutral: 0;
    #   - positive: 1.
    numeric_sentiment = np.where(y_pred == 'positive', 1, np.where(y_pred == 'neutral', 0, -1))
    new_dataset['numericSentiment'] = numeric_sentiment
    
    # We change the order of columns.
    columns = ['rating', 'opinion', 'predictedSentiment', 'numericSentiment',
               'reviewTime', 'unixReviewTime',
               'reviewText', 'reviewLength', 'summary', 'summaryLength',
               'asin', 'title', 'description', 'price', 
               'reviewerID', 'reviewerName', 'verified',
               'vote', 'style', 'image', 'imageURLHighRes'] 
    new_dataset = new_dataset.loc[:, columns]
    
    new_dataset.to_csv(DATASET_WITH_SENTIMENT_PATH, encoding='utf-8', index=False)
    
    return new_dataset


def plot_top_products_by_sentiment(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(['asin', 'title']).size().reset_index(name='num_reviews')
    product_reviews_distribution = product_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    reviews_counts = product_reviews_distribution['num_reviews'].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)
    print(f"First quartile: {q1}")
    print(f"Second quartile: {q2}")
    print(f"Third quartile: {q3}")

    # We keep only the products with a number of reviews that exceeds the third quartile.
    product_reviews_distribution = product_reviews_distribution[product_reviews_distribution['num_reviews'] >= q3]
    asins = product_reviews_distribution['asin'].values
    sentiments_per_product = dataset.groupby('asin')['numericSentiment'].agg(['mean', 'std']).reset_index()
    sentiments_per_product = sentiments_per_product[sentiments_per_product['asin'].isin(asins)]

    # We compute the mean sentiment for each product.
    mean_sentiments = sentiments_per_product['mean'].values
    mean_sentiments_sorted_indices = np.argsort(mean_sentiments)[::-1]
    mean_sentiments = mean_sentiments[mean_sentiments_sorted_indices]
    
    # We sort the products by the average sentiment in descending order.
    sentiments_per_product = sentiments_per_product.sort_values(by='mean', ascending=False)
    
    # We plot the top products.
    top_products = sentiments_per_product.head(k)
    fig = px.bar(
        top_products, 
        x='asin', 
        y='mean',
        orientation='v', 
        labels={'mean': 'Average sentiment', 'asin': 'Product'},
        title=f'Top {k} products by average sentiment',
        color_discrete_sequence=['mediumseagreen'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_products['asin'], ticktext=top_products['asin']),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/top_products_by_average_sentiment.png')
    fig.show()
    
    # We also plot the bottom products.
    bottom_products = sentiments_per_product.tail(k)
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    fig = px.bar(
        bottom_products, 
        x='asin', 
        y='mean',
        orientation='v', 
        labels={'mean': 'Average sentiment', 'asin': 'Product'},
        title=f'Bottom {k} products by average sentiment',
        color_discrete_sequence=['salmon'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=bottom_products['asin'], ticktext=bottom_products['asin']),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/bottom_products_by_average_sentiment.png')
    fig.show()
    
    # We print the data.
    sentiments_per_product = pd.merge(sentiments_per_product, product_reviews_distribution, on='asin', how='left')
    columns = ['asin', 'title', 'mean', 'std', 'num_reviews'] 
    sentiments_per_product = sentiments_per_product.loc[:, columns]
    print(f"Top {k} products by average sentiment: ")
    print(sentiments_per_product[:k])
    bottom_products = sentiments_per_product[-k:]
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    print(f"Bottom {k} products by average sentiment: ")
    print(bottom_products)
    
    
def plot_sentiment_price_relation(dataset):
    # We print a descriptive analysis of prices and sentiments.
    dataset[['price', 'numericSentiment']].describe()
    
    # We plot a scatter plot between rating and price.
    scatter_plot = px.scatter(
        dataset, 
        x='price', 
        y='numericSentiment', 
        title='Scatter plot between sentiment and price')
    scatter_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Sentiment',
    )
    scatter_plot.write_image(f'{ASSETS_PATH}/scatter_plot_sentiment_price.png')
    scatter_plot.show()
    
    # We plot the Pearson correlation coefficient between sentiment and price.
    correlation = dataset['numericSentiment'].corr(dataset['price'], method='pearson')
    print(f'Pearson correlation between sentiment and price: {correlation}')
    
    # We split prices in intervals.
    price_bins = [0, 8, 12, 18, 24, 34, 50, 200, float('inf')]
    price_labels = ['[0, 8)', '[8, 12)', '[12, 18)', '[18, 24)', '[24, 34)', '[34, 50)', '[50, 200)', 'Greater than 200']
    dataset['price_range'] = pd.cut(dataset['price'], bins=price_bins, labels=price_labels, include_lowest=True)  
    
    # We plot a box plot for the analysis of the price intervals.
    box_plot = px.box(
        dataset, 
        x='price_range', 
        y='numericSentiment', 
        title='Sentiment analysis of price intervals')
    box_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Sentiment',
    )
    # We assign colors to each sentiment value based on the legend.
    sentiment_colors = {-1: 'salmon', 0: 'skyblue', 1: 'mediumseagreen'}

    # We add color-coded markers for each sentiment value.
    for sentiment, color in sentiment_colors.items():
        print("sentiment: ", sentiment)
        if sentiment == -1:
            sentiment_word = "Negative"
        elif sentiment == 0:
            sentiment_word = "Neutral"
        else:
            sentiment_word = "Positive"
        box_plot.add_trace(
            go.Scatter(
                x=dataset.loc[dataset['numericSentiment'] == sentiment, 'price_range'],
                y=dataset.loc[dataset['numericSentiment'] == sentiment, 'numericSentiment'],
                mode='markers',
                marker_symbol='x',
                marker_color=color,
                name=f'{sentiment_word} sentiment',
            )
        )
    scatter_plot.write_image(f'{ASSETS_PATH}/box_plot_sentiment_price.png')
    box_plot.show()
    
    
def plot_sentiments_distribution(dataset):
    sentiment_counts = dataset['predictedSentiment'].value_counts()
    sentiment_order = ['negative', 'neutral', 'positive']
    sentiment_counts = sentiment_counts.loc[sentiment_order]
    df_sentiment = pd.DataFrame({'Sentiment': sentiment_order, 'Frequency': sentiment_counts.values})
    fig = px.bar(
        df_sentiment,
        x='Sentiment',
        y='Frequency',
        color='Sentiment',
        color_discrete_sequence=['salmon', 'skyblue', 'mediumseagreen'],
        labels={'Frequency': 'Frequency', 'Sentiment': 'Sentiment'},
        title='Distribution of sentiments'
    )
    for i, row in df_sentiment.iterrows():
        fig.add_annotation(
            x=row['Sentiment'],
            y=row['Frequency'] + 0.1,
            text=str(row['Frequency']),
            showarrow=False,
            font=dict(color='black', size=12),
            yshift=10
        )
    fig.show()
    fig.write_image(f'{ASSETS_PATH}/sentiments_distribution.png')
    

def plot_intersection_most_common_words(negative_dataset, neutral_dataset, positive_dataset, k=20):
    tqdm.pandas()
    new_negative_dataset = negative_dataset.copy()
    new_neutral_dataset = neutral_dataset.copy()
    new_positive_dataset = positive_dataset.copy()
    
    # We collect the tokens for each dataset.
    negative_tokens = [token for tokens in new_negative_dataset['preprocessedReviewText'] for token in word_tokenize(tokens)]
    neutral_tokens = [token for tokens in new_neutral_dataset['preprocessedReviewText'] for token in word_tokenize(tokens)]
    positive_tokens = [token for tokens in new_positive_dataset['preprocessedReviewText'] for token in word_tokenize(tokens)]
    
    # We find the most common words for each dataset.
    negative_word_counts = Counter(negative_tokens)
    negative_top_words = negative_word_counts.most_common(k)
    neutral_word_counts = Counter(neutral_tokens)
    neutral_top_words = neutral_word_counts.most_common(k)
    positive_word_counts = Counter(positive_tokens)
    positive_top_words = positive_word_counts.most_common(k)
    negative_top_words_keys = []
    neutral_top_words_keys = []
    positive_top_words_keys = []
    for i in range(k):
        negative_top_words_keys.append(negative_top_words[i][0])
        neutral_top_words_keys.append(neutral_top_words[i][0])
        positive_top_words_keys.append(positive_top_words[i][0])
        
    # We create a single set with alle the most common words.
    common_top_words = set(negative_top_words_keys + neutral_top_words_keys + positive_top_words_keys)
    
    # We count the frequency for each word.
    negative_top_words_counts = []
    neutral_top_words_counts = []
    positive_top_words_counts = []
    for word in common_top_words:
        negative_top_words_counts.append(negative_word_counts[word])
        neutral_top_words_counts.append(neutral_word_counts[word])
        positive_top_words_counts.append(positive_word_counts[word])
        
    # We create a dataframe with words and frequencies.
    dataframe_common_top_words = pd.DataFrame(list(zip(common_top_words, negative_top_words_counts, neutral_top_words_counts, positive_top_words_counts)), 
                                 columns =['word', 'negative', 'neutral', 'positive'])
    dataframe_common_top_words = dataframe_common_top_words.sort_values(by='positive',ascending=False)
    dataframe_common_top_words.reset_index(drop=True, inplace=True)
    
    # We make a plot with words and frequencies.
    fig = px.bar(
        dataframe_common_top_words,
        x='word',
        y=['negative', 'neutral', 'positive'],
        color_discrete_map={'negative': 'red', 'neutral': 'blue', 'positive': 'green'},
        labels={'x': 'Word', 'y': 'Frequency', 'variable': 'Predicted sentiment'},
        title=f'Top {k} common words for negative, neutral and positive sentiments',
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=list(range(len(dataframe_common_top_words)))),
        legend_title_text='Predicted sentiment',
    )
    fig.update_layout(xaxis_title='Word', yaxis_title='Frequency')
    fig.write_image(f'{ASSETS_PATH}/most_common_words_sentiment_intersection.png')
    fig.show()

    dataframe_common_top_words['total'] = dataframe_common_top_words.sum(axis=1)
    dataframe_common_top_words['negative'] = dataframe_common_top_words['negative'] / dataframe_common_top_words['total'] * 100
    dataframe_common_top_words['neutral'] = dataframe_common_top_words['neutral'] / dataframe_common_top_words['total'] * 100
    dataframe_common_top_words['positive'] = dataframe_common_top_words['positive'] / dataframe_common_top_words['total'] * 100
    fig = px.bar(
        dataframe_common_top_words,
        x='word',
        y=['negative', 'neutral', 'positive'],
        labels={'value': 'Predicted sentiment percentage', 'variable': 'Predicted sentiment'},
        title='Predicted sentiment percentages for the most common words',
        color_discrete_map={'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'mediumseagreen'},
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=list(range(len(dataframe_common_top_words)))),
    )
    fig.update_layout(barmode='stack', legend_title_text='Predicted sentiment')
    fig.update_layout(xaxis_title='Word')
    fig.write_image(f'{ASSETS_PATH}/sentiment_percentage_per_word.png')
    fig.show()
    

