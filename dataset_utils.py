import re
import nltk
import string
import contractions
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from wordcloud import WordCloud
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


ASSETS_PATH = "./assets"


def preprocess_reviews_dataset(dataset):
    # We format the date of the reviews.
    dataset = format_review_date(dataset)

    # We remove duplicates.
    dataset = remove_reviews_duplicates(dataset)
    
    # We remove the rows with NaN values for 'reviewText' and 'summary'.
    dataset = drop_reviews_nan_values(dataset)
    
    # We add an attribute for the review's length.
    dataset = add_review_length(dataset)
    
    # We add an attribute for the summary's length.
    dataset = add_summary_length(dataset)
    
    # We rename the 'overall' attribute to 'rating'.
    dataset = rename_overall_attribute(dataset)
    
    # We add the 'opinion' attribute.
    dataset = add_opinion(dataset)
    
    # We sort the dataset by the reviews' time.
    dataset = dataset.sort_values(by='unixReviewTime')
    
    return dataset
    
    
def format_review_date(dataset):
    new_dataset = dataset.copy()
    new_dataset['reviewTime'] = pd.to_datetime(new_dataset.reviewTime)
    return new_dataset


def remove_reviews_duplicates(dataset):
    columns_subset=['overall', 'reviewTime', 'unixReviewTime', 'reviewerID', 'asin', 'reviewText']
    new_dataset = dataset.drop_duplicates(subset=columns_subset, keep='first')
    return new_dataset


def drop_reviews_nan_values(dataset):
    new_dataset = dataset.copy()
    new_dataset = new_dataset.dropna(subset=['reviewText'], how='any')
    new_dataset = new_dataset.dropna(subset=['summary'], how='any')
    return new_dataset


def add_review_length(dataset):
    new_dataset = dataset.copy()
    new_dataset['reviewLength'] = new_dataset['reviewText'].apply(lambda x: len(str(x).split()))
    # We remove the reviews with no text.
    new_dataset = new_dataset[new_dataset['reviewLength'] != 0]
    return new_dataset


def add_summary_length(dataset):
    new_dataset = dataset.copy()
    new_dataset['summaryLength'] = new_dataset['summary'].apply(lambda x: len(str(x).split()))
    # We remove the reviews without summary.
    new_dataset = new_dataset[new_dataset['summaryLength'] != 0]
    return new_dataset


def rename_overall_attribute(dataset):
    new_dataset = dataset.copy()
    # We convert the values of the 'overall' column to integers.
    new_dataset['overall'] = new_dataset['overall'].astype(int)
    new_dataset = new_dataset.rename(columns={'overall': 'rating'})
    return new_dataset


def add_opinion(dataset):
    new_dataset = dataset.copy()
    # We add an opinion value with the following rules:
    #   - negative: rating 1 or 2;
    #   - netrual: rating 3;
    #   - positive: rating 4 or 5.
    new_dataset['opinion'] = pd.cut(
        new_dataset['rating'], 
        bins=[0, 2, 3, 5], 
        labels=['negative', 'neutral', 'positive'], 
        include_lowest=False
    )
    return new_dataset


def merge_reviews_and_products(reviews_dataset, products_dataset):
    # We keep only the relevant information from the products dataset.
    products_dataset = keep_products_subset_columns(products_dataset)
    
    # We remove duplicates from the products dataset.
    products_dataset = remove_products_duplicates(products_dataset)
    
    # We merge the datasets.
    dataset = pd.merge(reviews_dataset, products_dataset, on='asin', how='left')
    
    # We convert prices to floats.
    dataset = convert_price_to_float(dataset)
    
    # We remove the reviews without product information.
    dataset = remove_reviews_without_product_information(dataset)
    
    # We rearrange the columns.
    dataset = rearrange_dataset_columns(dataset)
    
    # We sort the dataset by the reviews' time.
    dataset = dataset.sort_values(by='unixReviewTime')
    
    return dataset
    
    
def keep_products_subset_columns(products_dataset):
    products_columns = ["asin", "title", "description", "price", "imageURLHighRes"]
    new_dataset = products_dataset.loc[:, products_columns]
    new_dataset['description'] = new_dataset['description'].astype(str)
    return new_dataset


def remove_products_duplicates(dataset):
    columns_subset=['asin', 'title', 'description', 'price']
    new_dataset = dataset.drop_duplicates(subset=columns_subset, keep='first')
    return new_dataset
    
    
def remove_reviews_without_product_information(dataset):
    # We fill empty values of price with NaN values in order to remove them.
    dataset['price'].replace('', np.nan, inplace=True)
    
    columns_subset=['title', 'description', 'price']
    new_dataset = dataset.dropna(subset=columns_subset, how='any')
    return new_dataset


def convert_price_to_float(dataset):
    # We remove the '$' symbol and convert values to floats. 
    dataset['price'] = dataset['price'].replace('[^\d.]', '', regex=True)
    dataset['price'] = pd.to_numeric(dataset['price'], errors='coerce')
    return dataset
    
    
def rearrange_dataset_columns(dataset):
    columns = ['rating', 'opinion', 'reviewTime', 'unixReviewTime',
               'reviewText', 'reviewLength', 'summary', 'summaryLength',
               'asin', 'title', 'description', 'price', 
               'reviewerID', 'reviewerName', 'verified',
               'vote', 'style', 'image', 'imageURLHighRes'] 
    dataset = dataset.loc[:, columns]
    return dataset
    

def plot_ratings_distribution(dataset, verified=False):
    ratings_counts = dataset['rating'].value_counts()
    ratings_order = [1, 2, 3, 4, 5] 
    ratings_counts = ratings_counts.loc[ratings_order]
    df_ratings = pd.DataFrame({'Rating': ratings_order, 'Frequency': ratings_counts.values})
    df_ratings['Rating'] = pd.Categorical(df_ratings['Rating'], categories=ratings_order, ordered=True)
    fig = px.bar(
        df_ratings,
        x='Rating',
        y='Frequency',
        color='Rating',
        color_discrete_sequence=['skyblue', 'mediumseagreen', 'darkorange', 'salmon', 'plum'],
        labels={'x': 'Rating', 'y': 'Frequency'},
    )
    # We add the value for each rating at the top of the bars.
    for i, row in df_ratings.iterrows():
        fig.add_annotation(
            x=row['Rating'],
            y=row['Frequency'] + 0.1,
            text=str(row['Frequency']),
            showarrow=False,
            font=dict(color='black', size=12),
            yshift=10
        )
    fig.show()
    if verified:
        fig.write_image(f'{ASSETS_PATH}/verified_ratings_distribution.png')
    else:
        fig.write_image(f'{ASSETS_PATH}/ratings_distribution.png')
    
     
def plot_opinions_distribution(dataset, verified=False):
    opinion_counts = dataset['opinion'].value_counts()
    opinion_order = ['negative', 'neutral', 'positive']
    opinion_counts = opinion_counts.loc[opinion_order]
    df_opinions = pd.DataFrame({'Opinion': opinion_order, 'Frequency': opinion_counts.values})
    fig = px.bar(
        df_opinions,
        x='Opinion',
        y='Frequency',
        color='Opinion',
        color_discrete_sequence=['salmon', 'skyblue', 'mediumseagreen'],
        labels={'Frequency': 'Frequency', 'Opinion': 'Opinion'},
        title='Distribution of opinions'
    )
    for i, row in df_opinions.iterrows():
        fig.add_annotation(
            x=row['Opinion'],
            y=row['Frequency'] + 0.1,
            text=str(row['Frequency']),
            showarrow=False,
            font=dict(color='black', size=12),
            yshift=10
        )

    fig.show()
    if verified:
        fig.write_image(f'{ASSETS_PATH}/verified_opinions_distribution.png')
    else:
        fig.write_image(f'{ASSETS_PATH}/opinions_distribution.png')
        
    
def plot_reviews_length_distribution(dataset):
    # We plot the reviews length distribution for the whole dataset.
    reviews_length = dataset['reviewLength'].values
    fig = px.histogram(
        x=reviews_length,
        nbins=200,
        labels={'count': 'Frequency', 'x': 'Review length'},
        title='Distribution of the number of words in the reviews'
    )
    fig.update_layout(
        xaxis_title='Review length',
        yaxis_title='Frequency',
        bargap=0.05,  # space between bars
        bargroupgap=0.1  # space between bar groups
    )
    fig.show()
    fig.write_image(f'{ASSETS_PATH}/reviews_length_distribution.png')
    
    
    # We plot the reviews length distribution for each opinion.
    grouped_dataset = dataset.groupby('opinion')[['reviewLength']].agg(['mean', 'std', 'min']).reset_index()
    grouped_dataset.columns = ['opinion', 'mean', 'std', 'min']
    colors = {'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'mediumseagreen'}
    fig = px.bar(
        grouped_dataset,
        x='opinion',
        y='mean',
        error_y='std',
        color='opinion',
        color_discrete_map=colors,
        title='Average review length with standard deviation for each opinion'
    )
    fig.update_layout(
        xaxis_title='Opinion',
        yaxis_title='Average review length',
        bargap=0.1,  # space between bars
        showlegend=False  # hide legend
    )
    # We add the mean value for each opinion.
    for i, mean_value in enumerate(grouped_dataset['mean']):
        fig.add_annotation(
            x=i,
            y=mean_value,
            text=f'{mean_value:.2f}',
            showarrow=True,
            arrowhead=3,
            ax=-20,
            ay=-20,
            font=dict(color='black', size=10)
        )
    # We add a line for the minimum value.
    for i, min_value in enumerate(grouped_dataset['min']):
        fig.add_shape(
            type='line',
            x0=i - 0.2,
            x1=i + 0.2,
            y0=min_value,
            y1=min_value,
            line=dict(color='black', dash='dash', width=1)
        )
        fig.add_annotation(
            x=i,
            y=min_value,
            text='minimum value',
            showarrow=True,
            arrowhead=3,
            ax=-20,
            ay=-20,
            font=dict(color='black', size=10)
        )
    fig.show()
    fig.write_image(f'{ASSETS_PATH}/average_review_length_for_opinions.png')
    
    
def plot_summaries_length_distribution(dataset):
    # We plot the summaries length distribution for the whole dataset.
    summaries_length = dataset['summaryLength'].values
    fig = px.histogram(
        x=summaries_length,
        nbins=200,
        labels={'count': 'Frequency', 'x': 'Summary length'},
        title='Distribution of the number of words in the summaries'
    )
    fig.update_layout(
        xaxis_title='Summary length',
        yaxis_title='Frequency',
        bargap=0.05,  # space between bars
        bargroupgap=0.1  # space between bar groups
    )
    fig.show()
    fig.write_image(f'{ASSETS_PATH}/summaries_length_distribution.png')
    
    # We plot the summaries length distribution for each opinion.
    grouped_dataset = dataset.groupby('opinion')[['summaryLength']].agg(['mean', 'std', 'min']).reset_index()
    grouped_dataset.columns = ['opinion', 'mean', 'std', 'min']
    colors = {'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'mediumseagreen'}
    fig = px.bar(
        grouped_dataset,
        x='opinion',
        y='mean',
        error_y='std',
        color='opinion',
        color_discrete_map=colors,
        title='Average summary length with standard deviation for each opinion'
    )
    fig.update_layout(
        xaxis_title='Opinion',
        yaxis_title='Average summary length',
        bargap=0.1,  # space between bars
        showlegend=False  # hide legend
    )
    # We add the mean value for each opinion.
    for i, mean_value in enumerate(grouped_dataset['mean']):
        fig.add_annotation(
            x=i,
            y=mean_value,
            text=f'{mean_value:.2f}',
            showarrow=True,
            arrowhead=3,
            ax=-20,
            ay=-20,
            font=dict(color='black', size=10)
        )
    # We add a line for the minimum value.
    for i, min_value in enumerate(grouped_dataset['min']):
        fig.add_shape(
            type='line',
            x0=i - 0.2,
            x1=i + 0.2,
            y0=min_value,
            y1=min_value,
            line=dict(color='black', dash='dash', width=1)
        )
        fig.add_annotation(
            x=i,
            y=min_value,
            text='minimum value',
            showarrow=True,
            arrowhead=3,
            ax=-20,
            ay=-20,
            font=dict(color='black', size=10)
        )
    fig.show()
    fig.write_image(f'{ASSETS_PATH}/average_summary_length_for_opinions.png')
    
    
def plot_reviews_price_distribution(dataset):
    new_dataset = dataset.copy()
    
    # We define the price intervals to consider.
    intervals = [0, 8, 12, 18, 24, 34, 50, 200, float('inf')]
    
    # For each product we assign an interval.
    new_dataset['bin'] = pd.cut(new_dataset['price'], bins=intervals, right=False, 
                                labels=[f'[{intervals[i]}, {intervals[i+1]})' for i in range(len(intervals) - 1)])
    
    # We count the number of products for each interval.
    counts_per_interval = new_dataset['bin'].value_counts().sort_index()
    
    # We plot the distribution.
    fig = px.bar(
        x=counts_per_interval.index,  
        y=counts_per_interval.values,  
        labels={'x': 'Price bin', 'y': 'Number of reviews'},
        title='Distribution of reviews by price',
        color_discrete_sequence=['salmon'],  
    )
    fig.update_layout(
        xaxis_title='Price intervals',
        yaxis_title='Number of reviews',
        bargap=0.2,  
        showlegend=False,  
    )
    fig.write_image(f'{ASSETS_PATH}/reviews_price_distribution.png')
    fig.show() 

    
def plot_products_price_distribution(dataset):
    new_dataset = dataset.copy()
    
    # We define the price intervals to consider.
    intervals = [0, 8, 12, 18, 24, 34, 50, 200, float('inf')]
    
    # We remove duplicates.
    new_dataset = new_dataset.drop_duplicates(subset=['asin', 'price'])
    
    # For each product we assign an interval.
    new_dataset['bin'] = pd.cut(new_dataset['price'], bins=intervals, right=False, 
                                labels=[f'[{intervals[i]}, {intervals[i+1]})' for i in range(len(intervals) - 1)])
    
    # We count the number of products for each interval.
    counts_per_interval = new_dataset['bin'].value_counts().sort_index()
    
    # We plot the distribution.
    fig = px.bar(
        x=counts_per_interval.index,  
        y=counts_per_interval.values,  
        labels={'x': 'Price bin', 'y': 'Number of products'},
        title='Distribution of products by price',
        color_discrete_sequence=['salmon'],  
    )
    fig.update_layout(
        xaxis_title='Price intervals',
        yaxis_title='Number of products',
        bargap=0.2,  
        showlegend=False,  
    )
    fig.write_image(f'{ASSETS_PATH}/products_price_distribution.png')
    fig.show()
    
    
def plot_rating_price_relation(dataset):
    # We print a descriptive analysis of prices and ratings.
    dataset[['price', 'rating']].describe()
    
    # We plot a scatter plot between rating and price.
    scatter_plot = px.scatter(
        dataset, 
        x='price', 
        y='rating', 
        title='Scatter plot between ratings and prices')
    scatter_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Rating',
    )
    scatter_plot.write_image(f'{ASSETS_PATH}/scatter_plot_rating_price.png')
    scatter_plot.show()
    
    # We plot the Pearson correlation coefficient between rating and price.
    correlation = dataset['rating'].corr(dataset['price'], method='pearson')
    print(f'Pearson correlation between rating and price: {correlation}')
    
    # We split prices in intervals.
    price_bins = [0, 8, 12, 18, 24, 34, 50, 200, float('inf')]
    price_labels = ['[0, 8)', '[8, 12)', '[12, 18)', '[18, 24)', '[24, 34)', '[34, 50)', '[50, 200)', 'Greater than 200']
    dataset['price_range'] = pd.cut(dataset['price'], bins=price_bins, labels=price_labels, include_lowest=True)  
    
    # We plot a box plot for the analysis of the price intervals.
    box_plot = px.box(
        dataset, 
        x='price_range', 
        y='rating', 
        title='Analysis of price intervals')
    box_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Rating',
    )
    scatter_plot.write_image(f'{ASSETS_PATH}/box_plot_rating_price.png')
    box_plot.show()
    
    
def plot_reviews_year_distribution(dataset):
    # We convert the reviewTime attribute to a DateTime format.
    dataset['reviewTime'] = pd.to_datetime(dataset['reviewTime'])
    dataset['year'] = dataset['reviewTime'].dt.year.astype(int)
    # We compute the number of reviews for each year.
    reviews_per_year = dataset.groupby('year').size().reset_index(name='num_reviews')
    fig = px.bar(
        reviews_per_year,
        x='year',
        y='num_reviews',
        labels={'year': 'Year', 'num_reviews': 'Number of reviews'},
        title='Distribution of reviews for each year',
        color_discrete_sequence=['salmon'],  
        width=800,  
    )
    fig.update_layout(
        showlegend=False,  
        xaxis=dict(tickmode='linear'),  # set tick mode to linear for better x-axis display
    )
    fig.write_image(f'{ASSETS_PATH}/reviews_year_distribution.png')
    fig.show()
    
    
def plot_reviewers(dataset, k=20):
    # We compute the number of reviews for each reviewer and we sort in descending order.
    reviewer_reviews_distribution = dataset.groupby('reviewerID').size().reset_index(name='num_reviews')
    reviewer_reviews_distribution = reviewer_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    
    # We plot the top reviewers for number of reviews.
    top_reviewers = reviewer_reviews_distribution[:k]
    fig_top = px.bar(
        top_reviewers,
        x='reviewerID',
        y='num_reviews',
        labels={'reviewerID': 'Reviewer', 'num_reviews': 'Number of reviews'},
        title=f'Top {k} reviewers for number of reviews',
        color_discrete_sequence=['mediumseagreen'],  
    )
    fig_top.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_reviewers['reviewerID'], ticktext=top_reviewers['reviewerID']),  
        showlegend=False,  
    )
    fig_top.write_image(f'{ASSETS_PATH}/top_reviewers.png')
    fig_top.show()
    
    # We plot the bottom reviewers for number of reviews.
    bottom_reviewers = reviewer_reviews_distribution[-k:]
    fig_bottom = px.bar(
        bottom_reviewers,
        x='reviewerID',
        y='num_reviews',
        labels={'reviewerID': 'Reviewer', 'num_reviews': 'Number of reviews'},
        title=f'Bottom {k} reviewers for number of reviews',
        color_discrete_sequence=['salmon'],  
    )
    fig_bottom.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=bottom_reviewers['reviewerID'], ticktext=bottom_reviewers['reviewerID']),  
        showlegend=False,  
    )
    fig_bottom.write_image(f'{ASSETS_PATH}/bottom_reviewers.png')
    fig_bottom.show()
    
    # We also print the data.
    print(f"Top {k} reviewers: ")
    print(reviewer_reviews_distribution[:k])
    print(f"Bottom {k} reviewers: ")
    print(reviewer_reviews_distribution[-k:])
    
    
def plot_products(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(['asin', 'title']).size().reset_index(name='num_reviews')
    product_reviews_distribution = product_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    
    # We plot the top products for number of reviews.
    top_products = product_reviews_distribution[:k]
    fig_top = px.bar(
        top_products,
        x='asin',
        y='num_reviews',
        labels={'asin': 'Product', 'num_reviews': 'Number of reviews'},
        title=f'Top {k} products for number of reviews',
        color_discrete_sequence=['mediumseagreen'],  
    )
    fig_top.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_products['asin'], ticktext=top_products['asin']),  
        showlegend=False, 
    )
    fig_top.write_image(f'{ASSETS_PATH}/top_products.png')
    fig_top.show()
    
    # We plot the bottom products for number of reviews.
    bottom_products = product_reviews_distribution[-k:]
    fig_bottom = px.bar(
        bottom_products,
        x='asin',
        y='num_reviews',
        labels={'asin': 'Product', 'num_reviews': 'Number of reviews'},
        title=f'Bottom {k} products for number of reviews',
        color_discrete_sequence=['salmon'],  
    )
    fig_bottom.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=bottom_products['asin'], ticktext=bottom_products['asin']),  
        showlegend=False, 
    )
    fig_bottom.write_image(f'{ASSETS_PATH}/bottom_products.png')
    fig_bottom.show()
    
    # We also print the data.
    print(f"Top {k} products: ")
    print(product_reviews_distribution[:k])
    print(f"Bottom {k} products: ")
    print(product_reviews_distribution[-k:])
    
    
def compute_average_reviews_per_product(dataset):
    reviews_per_product = dataset.groupby('asin')['reviewerID'].count()
    mean = reviews_per_product.mean()
    std = reviews_per_product.std()
    print(f"There is an average of {mean} ± {std} reviews per product.")
    '''
    stats_df = pd.DataFrame({
        'Average reviews': [mean],
        'Std reviews': [std]
    })
    fig = px.scatter(
        stats_df,
        y='Average reviews',
        error_y='Std reviews',
        title='Average number of reviews per product',
        labels={'value': 'Reviews'},
    )
    fig.update_traces(error_y_thickness=0.8)
    fig.update_xaxes(title=None, showticklabels=False)
    fig.write_image('./assets/average_reviews_per_product.png')
    fig.show()
    '''
    

def compute_average_reviews_per_reviewer(dataset):
    reviews_per_reviewer = dataset.groupby('reviewerID')['asin'].count()
    mean = reviews_per_reviewer.mean()
    std = reviews_per_reviewer.std()
    print(f"There is an average of {mean} ± {std} reviews per reviewer.")
    '''
    stats_df = pd.DataFrame({
        'Average reviews': [mean],
        'Std reviews': [std]
    })
    fig = px.scatter(
        stats_df,
        y='Average reviews',
        error_y='Std reviews',
        title='Average number of reviews per reviewer',
        labels={'value': 'Reviews'},
    )
    fig.update_traces(error_y_thickness=0.8)
    fig.update_xaxes(title=None, showticklabels=False)
    fig.write_image('./assets/average_reviews_per_reviewer.png')
    fig.show()
    '''
    
    
def plot_average_rating_per_product_distribution(dataset):
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
    print("Number of products that exceeds the third quartile: ", len(product_reviews_distribution))
    asins = product_reviews_distribution['asin'].values
    ratings_per_product = dataset.groupby('asin')['rating'].agg(['mean', 'std']).reset_index()
    ratings_per_product = ratings_per_product[ratings_per_product['asin'].isin(asins)]
    
    # We compute the mean rating for each product.
    mean_ratings = ratings_per_product['mean'].values
    mean_ratings_sorted_indices = np.argsort(mean_ratings)[::-1]
    mean_ratings = mean_ratings[mean_ratings_sorted_indices]
    
    # We plot the distribution of the mean ratings.
    fig = px.histogram(
        x=mean_ratings, 
        nbins=200, 
        labels={'x': 'Average rating'},
        title='Distribution of the average rating per product',
        color_discrete_sequence=['salmon'])
    fig.update_layout(yaxis_title='Number of products')
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.write_image(f'{ASSETS_PATH}/average_rating_per_product.png')
    fig.show()
    
    
def plot_average_rating_per_reviewer_distribution(dataset):
    # We compute the number of reviews for each reviewer and we sort in descending order.
    reviewer_reviews_distribution = dataset.groupby('reviewerID').size().reset_index(name='num_reviews')
    reviewer_reviews_distribution = reviewer_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    reviews_counts = reviewer_reviews_distribution['num_reviews'].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)
    print(f"First quartile: {q1}")
    print(f"Second quartile: {q2}")
    print(f"Third quartile: {q3}")

    # We keep only the reviewers with a number of reviews that exceeds the second quartile.
    reviewer_reviews_distribution = reviewer_reviews_distribution[reviewer_reviews_distribution['num_reviews'] >= q2]
    reviewer_ids = reviewer_reviews_distribution['reviewerID'].values
    ratings_per_reviewer = dataset.groupby('reviewerID')['rating'].agg(['mean', 'std']).reset_index()
    ratings_per_reviewer = ratings_per_reviewer[ratings_per_reviewer['reviewerID'].isin(reviewer_ids)]

    # We compute the mean rating for each reviewer.
    mean_ratings = ratings_per_reviewer['mean'].values
    mean_ratings_sorted_indices = np.argsort(mean_ratings)[::-1]
    mean_ratings = mean_ratings[mean_ratings_sorted_indices]

    # We plot the distribution of the mean ratings.
    fig = px.histogram(
        x=mean_ratings, 
        nbins=200, 
        labels={'x': 'Average rating'},
        title='Distribution of the average rating per reviewer',
        color_discrete_sequence=['salmon'])
    fig.update_layout(yaxis_title='Number of reviewers')
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.write_image(f'{ASSETS_PATH}/average_rating_per_reviewer.png')
    fig.show()
    
    
def plot_top_rated_products(dataset, k=20):
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
    print("Number of products that exceeds the third quartile: ", len(product_reviews_distribution))
    asins = product_reviews_distribution['asin'].values
    ratings_per_product = dataset.groupby('asin')['rating'].agg(['mean', 'std']).reset_index()
    ratings_per_product = ratings_per_product[ratings_per_product['asin'].isin(asins)]

    # We compute the mean rating for each product.
    mean_ratings = ratings_per_product['mean'].values
    mean_ratings_sorted_indices = np.argsort(mean_ratings)[::-1]
    mean_ratings = mean_ratings[mean_ratings_sorted_indices]
    
    # We sort the products by the average rating in descending order.
    ratings_per_product = ratings_per_product.sort_values(by='mean', ascending=False)
    
    # We plot the top products.
    top_products = ratings_per_product.head(k)
    fig = px.bar(
        top_products, 
        x='asin', 
        y='mean',
        orientation='v', 
        labels={'mean': 'Average rating', 'asin': 'Product'},
        title=f'Top {k} products by average rating',
        color_discrete_sequence=['mediumseagreen'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_products['asin'], ticktext=top_products['asin']),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/top_products_by_average_rating.png')
    fig.show()
    
    # We also plot the bottom products.
    bottom_products = ratings_per_product.tail(k)
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    fig = px.bar(
        bottom_products, 
        x='asin', 
        y='mean',
        orientation='v', 
        labels={'mean': 'Average rating', 'asin': 'Product'},
        title=f'Bottom {k} products by average rating',
        color_discrete_sequence=['salmon'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=bottom_products['asin'], ticktext=bottom_products['asin']),  
        showlegend=False, 
    )
    fig.write_image(f'{ASSETS_PATH}/bottom_products_by_average_rating.png')
    fig.show()
    
    # We print the data.
    ratings_per_product = pd.merge(ratings_per_product, product_reviews_distribution, on='asin', how='left')
    columns = ['asin', 'title', 'mean', 'std', 'num_reviews'] 
    ratings_per_product = ratings_per_product.loc[:, columns]
    print(f"Top {k} products by average rating: ")
    print(ratings_per_product[:k])
    bottom_products = ratings_per_product[-k:]
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    print(f"Bottom {k} products by average rating: ")
    print(bottom_products)
    
    
def plot_information_over_time(dataset):
    # We compute the relevant information for each year.
    # In particular we compute the mean and the standard deviation.
    dataset['reviewTime'] = pd.to_datetime(dataset['reviewTime'])
    dataset['year'] = dataset['reviewTime'].dt.year.astype(int)
    grouped_dataset = dataset.groupby('year').agg({
        'rating': ['mean', 'std'],
        'reviewLength': ['mean', 'std'],
        'summaryLength': ['mean', 'std']
    }).reset_index()
    grouped_dataset.columns = ['year', 'rating_mean', 'rating_std', 
                               'reviewLength_mean', 'reviewLength_std', 
                               'summaryLength_mean', 'summaryLength_std']
    
    # We plot the error bar for the ratings.
    fig_rating = px.line(
        grouped_dataset,
        x='year',
        y='rating_mean',
        error_y='rating_std',
        labels={'rating_mean': 'Average rating', 'year': 'Year'},
        title='Average rating for each year with standard deviations'
    )
    fig_rating.update_traces(error_y_thickness=0.8)
    fig_rating.write_image(f'{ASSETS_PATH}/average_rating_per_year.png')
    fig_rating.show()
    
    # We plot the error bar for reviews' length.
    fig_review_length = px.line(
        grouped_dataset,
        x='year',
        y='reviewLength_mean',
        error_y='reviewLength_std',
        labels={'reviewLength_mean': 'Average review length', 'year': 'Year'},
        title='Average review length for each year with standard deviations'
    )
    fig_review_length.update_traces(error_y_thickness=0.8)
    fig_review_length.write_image(f'{ASSETS_PATH}/average_review_length_per_year.png')
    fig_review_length.show()
    
    # We plot the error bar for summaries' length.
    fig_summary_length = px.line(
        grouped_dataset,
        x='year',
        y='summaryLength_mean',
        error_y='summaryLength_std',
        labels={'summaryLength_mean': 'Average summary length', 'year': 'Year'},
        title='Average summary length for each year with standard deviations'
    )
    fig_summary_length.update_traces(error_y_thickness=0.8)
    fig_summary_length.write_image(f'{ASSETS_PATH}/average_summary_length_per_year.png')
    fig_summary_length.show()
    
    # We plot the percentages of opinions over the years.
    grouped_dataset = dataset.groupby(['year', 'opinion']).size().unstack(fill_value=0)
    grouped_dataset['total'] = grouped_dataset.sum(axis=1)
    grouped_dataset['negative'] = grouped_dataset['negative'] / grouped_dataset['total'] * 100
    grouped_dataset['neutral'] = grouped_dataset['neutral'] / grouped_dataset['total'] * 100
    grouped_dataset['positive'] = grouped_dataset['positive'] / grouped_dataset['total'] * 100
    fig = px.bar(
        grouped_dataset,
        x=grouped_dataset.index,
        y=['negative', 'neutral', 'positive'],
        labels={'value': 'Opinion percentage', 'variable': 'Opinion'},
        title='Opinion percentages for each year',
        color_discrete_map={'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'mediumseagreen'},
    )
    fig.update_layout(barmode='stack', legend_title_text='Opinion')
    fig.write_image(f'{ASSETS_PATH}/opinion_percentage_per_year.png')
    fig.show()
    
    
def plot_correlations(dataset):
    ratings = dataset['rating'].values
    reviews_length = dataset['reviewLength'].values
    summaries_length = dataset['summaryLength'].values
    prices = dataset['price'].values
    correlation_ratings_reviews_length = compute_pearson_coefficient(ratings, reviews_length)
    correlation_ratings_summaries_length = compute_pearson_coefficient(ratings, summaries_length)
    correlation_ratings_prices = compute_pearson_coefficient(ratings, prices)
    print("Pearson correlation coefficient between ratings and reviews length: ", correlation_ratings_reviews_length)
    print("Pearson correlation coefficient between ratings and summaries length: ", correlation_ratings_summaries_length)
    print("Pearson correlation coefficient between ratings and prices: ", correlation_ratings_prices)
    '''
    pairplot = sns.pairplot(
        data=dataset[["rating", "reviewLength", "summaryLength", "price"]], 
        diag_kind='kde', # kernel density estimation
        plot_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'k'}
    )
    pairplot.savefig(f'{ASSETS_PATH}/feature_correlations.png')
    plt.show()
    '''
    fig = px.scatter_matrix(
        dataset,
        dimensions=["rating", "reviewLength", "summaryLength", "price"],
        title="Feature correlations",
        color="rating",  
        opacity=0.7, 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(width=800, height=800)
    fig.write_image(f'{ASSETS_PATH}/feature_correlations.png')
    fig.show()
    
    
def compute_pearson_coefficient(x, y, **kwargs):
    coef = np.corrcoef(x, y)[0][1]
    return round(coef, 5)
    

def plot_most_common_words(dataset, opinion="", k=20):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    all_tokens = [token for tokens in new_dataset['preprocessedReviewText'] for token in word_tokenize(tokens)]
    
    # We plot a word cloud of the most common words.
    wordcloud = WordCloud(width=1600, height=800, background_color="black").generate_from_frequencies(Counter(all_tokens))
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='antialiased')
    if opinion != "":
        plt.savefig(f'{ASSETS_PATH}/most_common_{opinion}_words_wordcloud.png')
    else:
        plt.savefig(f'{ASSETS_PATH}/most_common_words_wordcloud.png')
        
    if opinion == "":
        title = "Most common words"
    else:
        title = f"Most common words for {opinion} reviews" 
        
    # We print the most common words with their count.
    word_counts = Counter(all_tokens)
    top_words = word_counts.most_common(k)
    words, counts = zip(*top_words)
    fig = px.bar(
        x=words,
        y=counts,
        color=words,  
        title=title,
        labels={'x': 'Word', 'y': 'Frequency'},
        color_discrete_sequence=['salmon']
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=list(range(len(words)))),
        showlegend=False
    )
    if opinion != "":
        fig.write_image(f'{ASSETS_PATH}/most_common_{opinion}_words_histogram.png')
    else:
        fig.write_image(f'{ASSETS_PATH}/most_common_words_histogram.png')
    fig.show()
    
    
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
        labels={'x': 'Word', 'y': 'Frequency', 'variable': 'Opinion'},
        title=f'Top {k} common words for negative, neutral and positive opinions',
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=list(range(len(dataframe_common_top_words)))),
        legend_title_text='Opinion',
    )
    fig.update_layout(xaxis_title='Word', yaxis_title='Frequency')
    fig.write_image(f'{ASSETS_PATH}/most_common_words_intersection.png')
    fig.show()

    dataframe_common_top_words['total'] = dataframe_common_top_words.sum(axis=1)
    dataframe_common_top_words['negative'] = dataframe_common_top_words['negative'] / dataframe_common_top_words['total'] * 100
    dataframe_common_top_words['neutral'] = dataframe_common_top_words['neutral'] / dataframe_common_top_words['total'] * 100
    dataframe_common_top_words['positive'] = dataframe_common_top_words['positive'] / dataframe_common_top_words['total'] * 100
    fig = px.bar(
        dataframe_common_top_words,
        x='word',
        y=['negative', 'neutral', 'positive'],
        labels={'value': 'Opinion percentage', 'variable': 'Opinion'},
        title='Opinion percentages for the most common words',
        color_discrete_map={'negative': 'salmon', 'neutral': 'skyblue', 'positive': 'mediumseagreen'},
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=list(range(len(dataframe_common_top_words)))),
    )
    fig.update_layout(barmode='stack', legend_title_text='Opinion')
    fig.update_layout(xaxis_title='Word')
    fig.write_image(f'{ASSETS_PATH}/opinion_percentage_per_word.png')
    fig.show()


def preprocess_text(text, lemmatization=False):
    # We expand contractions. Example: don't -> do not.
    expanded_text = contractions.fix(text)
    
    # We tokenize the text.
    tokens = word_tokenize(expanded_text)
    
    # We convert the tokens to lower case.
    tokens = [token.lower() for token in tokens]
    
    # We remove repetitions.
    tokens = [remove_repetitions(token) for token in tokens]
    
    # We remove the punctuation.
    punctuation = string.punctuation
    tokens = [token for token in tokens if token not in punctuation]
    
    # We specify negations.
    negations = ['not', 'nor', 'against', 'no', 'never']
    negations_set = set(negations)
    
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
                        ".....", "\'\'",
                        "not", "nor", "against", "no", "never"]
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
    

def find_frequent_tokens_in_reviews(dataset, threshold):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We create a sparse matrix for the frequency of tokens in the reviews (Bag of Words).
    vectorizer = TfidfVectorizer()
    bow = vectorizer.fit_transform(new_dataset['preprocessedReviewText'].astype(str))
    
    # We compute the percentage of reviews where each token is present.
    tokens = vectorizer.get_feature_names_out()
    percentage_present = bow.sign().sum(axis=0).A1 / len(new_dataset)
    
    # We get the tokens that are present in a percentage of reviews that exceeds the threshold.
    common_tokens = [token for token, percentage in zip(tokens, percentage_present) if percentage > threshold]
    print(f"Tokens present in more than {threshold*100}% of reviews:", common_tokens)
    
    
def preprocess_reviews(dataset):
    tqdm.pandas()
    new_dataset = dataset.copy()
    
    # We preprocess the reviews in the dataset.
    new_dataset['preprocessedReviewText'] = new_dataset['reviewText'].progress_map(lambda text: preprocess_text(text))
    
    # We convert the list of tokens in a string text.
    new_dataset['preprocessedReviewText'] = new_dataset['preprocessedReviewText'].apply(lambda tokens: ' '.join(tokens))
    
    return new_dataset
    
    
    
    