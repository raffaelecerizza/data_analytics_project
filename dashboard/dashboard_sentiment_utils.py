import os
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, Input, Output, dcc, html, dash_table


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
        labels={'Frequency': 'Frequency', 'Sentiment': 'Sentiment'}
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
    return fig
    
    
def plot_top_products_by_sentiment(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(['asin', 'title']).size().reset_index(name='num_reviews')
    product_reviews_distribution = product_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    reviews_counts = product_reviews_distribution['num_reviews'].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)

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
        color_discrete_sequence=['mediumseagreen'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_products['asin'], ticktext=top_products['asin']),  
        showlegend=False, 
    )
    return fig


def plot_bottom_products_by_sentiment(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(['asin', 'title']).size().reset_index(name='num_reviews')
    product_reviews_distribution = product_reviews_distribution.sort_values(by='num_reviews', ascending=False)
    reviews_counts = product_reviews_distribution['num_reviews'].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)

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
    
    # We plot the bottom products.
    bottom_products = sentiments_per_product.tail(k)
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    fig = px.bar(
        bottom_products, 
        x='asin', 
        y='mean',
        orientation='v', 
        labels={'mean': 'Average sentiment', 'asin': 'Product'},
        color_discrete_sequence=['salmon'])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=bottom_products['asin'], ticktext=bottom_products['asin']),  
        showlegend=False, 
    )
    return fig


def plot_scatter_plot_sentiments_prices(dataset): 
    # We plot a scatter plot between sentiment and price.
    # We print a descriptive analysis of prices and sentiments.
    dataset[['price', 'numericSentiment']].describe()
    
    # We plot a scatter plot between rating and price.
    scatter_plot = px.scatter(
        dataset, 
        x='price', 
        y='numericSentiment', 
    )
    scatter_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Sentiment',
    )

    return scatter_plot


def plot_sentiment_analysis_price_intervals(dataset):
    # We split prices in intervals.
    price_bins = [0, 8, 12, 18, 24, 34, 50, 200, float('inf')]
    price_labels = ['[0, 8)', '[8, 12)', '[12, 18)', '[18, 24)', '[24, 34)', '[34, 50)', '[50, 200)', 'Greater than 200']
    dataset['price_range'] = pd.cut(dataset['price'], bins=price_bins, labels=price_labels, include_lowest=True)  
    
    # We plot a box plot for the analysis of the price intervals.
    box_plot = px.box(
        dataset, 
        x='price_range', 
        y='numericSentiment', 
    )
    box_plot.update_layout(
        xaxis_title='Price',
        yaxis_title='Sentiment',
    )
    # We assign colors to each sentiment value based on the legend.
    sentiment_colors = {-1: 'salmon', 0: 'skyblue', 1: 'mediumseagreen'}

    # We add color-coded markers for each sentiment value.
    for sentiment, color in sentiment_colors.items():
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
      
    return box_plot
    
    