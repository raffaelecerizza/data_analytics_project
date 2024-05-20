import os
import ast
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
from dash import Dash, Input, Output, dcc, html, dash_table



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
        color_discrete_sequence=['salmon']
    )
    histogram.update_layout(
        xaxis=dict(tickangle=-45, tickmode='array'),  
        showlegend=False, 
    )
    
    return histogram


def plot_top_aspects_by_sentiment(dataset, k=20):
    new_dataset = dataset.copy()
    
    # We compute a dictionary for the relevant information for each aspect.
    result_dict = compute_aspects_dictionary(new_dataset)
        
    # We create a dataframe from the dictionary.    
    aspects_df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    # We print the quartiles for the frequency of the aspects.
    first_quartile = np.percentile(aspects_df['Frequency'], 25)
    second_quartile = np.percentile(aspects_df['Frequency'], 50)
    third_quartile = np.percentile(aspects_df['Frequency'], 75)

    # We keep only the last quartile of aspects.
    filtered_aspects_df = aspects_df[aspects_df['Frequency'] > third_quartile]
    
    # We sort the dataframe by the mean sentiment.
    filtered_aspects_df = filtered_aspects_df.sort_values(by='Mean', ascending=False)
        
    # We plot the top aspects by mean sentiment.
    top_aspects = filtered_aspects_df.head(k)
    fig = px.bar(
        top_aspects, 
        x=top_aspects.index, 
        y='Mean', 
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
    return fig


def plot_bottom_aspects_by_sentiment(dataset, k=20):
    new_dataset = dataset.copy()
    
    # We compute a dictionary for the relevant information for each aspect.
    result_dict = compute_aspects_dictionary(new_dataset)
        
    # We create a dataframe from the dictionary.    
    aspects_df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    # We print the quartiles for the frequency of the aspects.
    first_quartile = np.percentile(aspects_df['Frequency'], 25)
    second_quartile = np.percentile(aspects_df['Frequency'], 50)
    third_quartile = np.percentile(aspects_df['Frequency'], 75)

    # We keep only the last quartile of aspects.
    filtered_aspects_df = aspects_df[aspects_df['Frequency'] > third_quartile]
    
    # We sort the dataframe by the mean sentiment.
    filtered_aspects_df = filtered_aspects_df.sort_values(by='Mean', ascending=False)
    
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
    return fig
    
    
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


def plot_aspect_sentiment_over_time(dataset, aspect):
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
        labels={'Mean': 'Mean sentiment', 'Year': 'Year'})
    fig.update_traces(error_y_thickness=0.8)
    return fig




    