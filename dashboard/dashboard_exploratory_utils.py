import os
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, Input, Output, dcc, html, dash_table


def plot_ratings_distribution(dataset, verified=False):
    ratings_counts = dataset["rating"].value_counts()
    ratings_order = [1, 2, 3, 4, 5] 
    ratings_counts = ratings_counts.loc[ratings_order]
    df_ratings = pd.DataFrame({"Rating": ratings_order, "Frequency": ratings_counts.values})
    df_ratings["Rating"] = pd.Categorical(df_ratings["Rating"], categories=ratings_order, ordered=True)
    fig = px.bar(
        df_ratings,
        x="Rating",
        y="Frequency",
        color="Rating",
        color_discrete_sequence=["skyblue", "mediumseagreen", "darkorange", "salmon", "plum"],
        labels={"x": "Rating", "y": "Frequency"},
    )
    # We add the value for each rating at the top of the bars.
    for i, row in df_ratings.iterrows():
        fig.add_annotation(
            x=row["Rating"],
            y=row["Frequency"] + 0.1,
            text=str(row["Frequency"]),
            showarrow=False,
            font=dict(color="black", size=12),
            yshift=10
        )
    return fig


def plot_opinions_distribution(dataset, verified=False):
    opinion_counts = dataset["opinion"].value_counts()
    opinion_order = ["negative", "neutral", "positive"]
    opinion_counts = opinion_counts.loc[opinion_order]
    df_opinions = pd.DataFrame({"Opinion": opinion_order, "Frequency": opinion_counts.values})
    fig = px.bar(
        df_opinions,
        x="Opinion",
        y="Frequency",
        color="Opinion",
        color_discrete_sequence=["salmon", "skyblue", "mediumseagreen"],
        labels={"Frequency": "Frequency", "Opinion": "Opinion"},
    )
    for i, row in df_opinions.iterrows():
        fig.add_annotation(
            x=row["Opinion"],
            y=row["Frequency"] + 0.1,
            text=str(row["Frequency"]),
            showarrow=False,
            font=dict(color="black", size=12),
            yshift=10
        )
    return fig


def plot_reviews_year_distribution(dataset):
    # We convert the reviewTime attribute to a DateTime format.
    dataset["reviewTime"] = pd.to_datetime(dataset["reviewTime"])
    dataset["year"] = dataset["reviewTime"].dt.year.astype(int)
    # We compute the number of reviews for each year.
    reviews_per_year = dataset.groupby("year").size().reset_index(name="num_reviews")
    fig = px.bar(
        reviews_per_year,
        x="year",
        y="num_reviews",
        labels={"year": "Year", "num_reviews": "Number of reviews"},
        color_discrete_sequence=["salmon"],  
    )
    fig.update_layout(
        showlegend=False,  
        xaxis=dict(tickmode="linear"),  # set tick mode to linear for better x-axis display
    )
    return fig


def plot_opinions_percentages_over_time(dataset):
    # We compute the relevant information for each year.
    # In particular we compute the mean and the standard deviation.
    dataset["reviewTime"] = pd.to_datetime(dataset["reviewTime"])
    dataset["year"] = dataset["reviewTime"].dt.year.astype(int)
    grouped_dataset = dataset.groupby("year").agg({
        "rating": ["mean", "std"],
        "reviewLength": ["mean", "std"],
        "summaryLength": ["mean", "std"]
    }).reset_index()
    grouped_dataset.columns = ["year", "rating_mean", "rating_std", 
                               "reviewLength_mean", "reviewLength_std", 
                               "summaryLength_mean", "summaryLength_std"]
    
    # We plot the percentages of opinions over the years.
    grouped_dataset = dataset.groupby(["year", "opinion"]).size().unstack(fill_value=0)
    grouped_dataset["total"] = grouped_dataset.sum(axis=1)
    grouped_dataset["negative"] = grouped_dataset["negative"] / grouped_dataset["total"] * 100
    grouped_dataset["neutral"] = grouped_dataset["neutral"] / grouped_dataset["total"] * 100
    grouped_dataset["positive"] = grouped_dataset["positive"] / grouped_dataset["total"] * 100
    fig = px.bar(
        grouped_dataset,
        x=grouped_dataset.index,
        y=["negative", "neutral", "positive"],
        labels={"value": "Opinion percentage", "variable": "Opinion"},
        color_discrete_map={"negative": "salmon", "neutral": "skyblue", "positive": "mediumseagreen"},
    )
    fig.update_layout(barmode="stack", legend_title_text="Opinion")
    return fig
    
    
def plot_top_rated_products(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(["asin", "title"]).size().reset_index(name="num_reviews")
    product_reviews_distribution = product_reviews_distribution.sort_values(by="num_reviews", ascending=False)
    reviews_counts = product_reviews_distribution["num_reviews"].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)

    # We keep only the products with a number of reviews that exceeds the third quartile.
    product_reviews_distribution = product_reviews_distribution[product_reviews_distribution["num_reviews"] >= q3]
    asins = product_reviews_distribution["asin"].values
    ratings_per_product = dataset.groupby("asin")["rating"].agg(["mean", "std"]).reset_index()
    ratings_per_product = ratings_per_product[ratings_per_product["asin"].isin(asins)]

    # We compute the mean rating for each product.
    mean_ratings = ratings_per_product["mean"].values
    mean_ratings_sorted_indices = np.argsort(mean_ratings)[::-1]
    mean_ratings = mean_ratings[mean_ratings_sorted_indices]
    
    # We sort the products by the average rating in descending order.
    ratings_per_product = ratings_per_product.sort_values(by="mean", ascending=False)
    
    # We plot the top products.
    top_products = ratings_per_product.head(k)
    fig = px.bar(
        top_products, 
        x="asin", 
        y="mean",
        orientation="v", 
        labels={"mean": "Average rating", "asin": "Product"},
        color_discrete_sequence=["salmon"])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode="array", tickvals=top_products["asin"], ticktext=top_products["asin"]),  
        showlegend=False, 
    )
    return fig


def plot_bottom_rated_products(dataset, k=20):
    # We compute the number of reviews for each product and we sort in descending order.
    product_reviews_distribution = dataset.groupby(["asin", "title"]).size().reset_index(name="num_reviews")
    product_reviews_distribution = product_reviews_distribution.sort_values(by="num_reviews", ascending=False)
    reviews_counts = product_reviews_distribution["num_reviews"].values

    # We compute the quartiles.
    q1 = np.percentile(reviews_counts, 25)
    q2 = np.percentile(reviews_counts, 50)  
    q3 = np.percentile(reviews_counts, 75)

    # We keep only the products with a number of reviews that exceeds the third quartile.
    product_reviews_distribution = product_reviews_distribution[product_reviews_distribution["num_reviews"] >= q3]
    asins = product_reviews_distribution["asin"].values
    ratings_per_product = dataset.groupby("asin")["rating"].agg(["mean", "std"]).reset_index()
    ratings_per_product = ratings_per_product[ratings_per_product["asin"].isin(asins)]

    # We compute the mean rating for each product.
    mean_ratings = ratings_per_product["mean"].values
    mean_ratings_sorted_indices = np.argsort(mean_ratings)[::-1]
    mean_ratings = mean_ratings[mean_ratings_sorted_indices]
    
    # We sort the products by the average rating in descending order.
    ratings_per_product = ratings_per_product.sort_values(by="mean", ascending=False)
    
    # We plot the bottom products.
    bottom_products = ratings_per_product.tail(k)
    bottom_products = bottom_products.sort_values(by='mean', ascending=True)
    fig = px.bar(
        bottom_products, 
        x="asin", 
        y="mean",
        orientation="v", 
        labels={"mean": "Average rating", "asin": "Product"},
        color_discrete_sequence=["salmon"])
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickmode="array", tickvals=bottom_products["asin"], ticktext=bottom_products["asin"]),  
        showlegend=False, 
    )
    return fig


def plot_distribution_products_prices(dataset):
    new_dataset = dataset.copy()
    
    # We define the price intervals to consider.
    intervals = [0, 8, 12, 18, 24, 34, 50, 200, float("inf")]
    
    # We remove duplicates.
    new_dataset = new_dataset.drop_duplicates(subset=["asin", "price"])
    
    # For each product we assign an interval.
    new_dataset["bin"] = pd.cut(new_dataset["price"], bins=intervals, right=False, 
                                labels=[f"[{intervals[i]}, {intervals[i+1]})" for i in range(len(intervals) - 1)])
    
    # We count the number of products for each interval.
    counts_per_interval = new_dataset["bin"].value_counts().sort_index()
    
    # We plot the distribution.
    fig = px.bar(
        x=counts_per_interval.index,  
        y=counts_per_interval.values,  
        labels={"x": "Price bin", "y": "Number of products"},
        color_discrete_sequence=["salmon"],  
    )
    fig.update_layout(
        xaxis_title="Price intervals",
        yaxis_title="Number of products",
        bargap=0.2,  
        showlegend=False,  
    )
    return fig


def plot_scatter_plot_ratings_prices(dataset): 
    # We plot a scatter plot between rating and price.
    scatter_plot = px.scatter(
        dataset, 
        x="price", 
        y="rating" 
    )
    scatter_plot.update_layout(
        xaxis_title="Price",
        yaxis_title="Rating",
    )
    return scatter_plot
    


    
    
    
    
    
    
    
    
    
    
    

    