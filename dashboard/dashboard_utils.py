import os
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import dashboard_exploratory_utils as deu
import dashboard_sentiment_utils as dsu

from dash import Dash, Input, Output, dcc, html, dash_table



def get_general_info_square(general_info):
    square = html.Div(
        children=[
            html.H2("General information about the dataset", className="title_square"),
            html.Br(),
            dash_table.DataTable(
                general_info.to_dict("records"), 
                [{"name": i, "id": i} for i in general_info.columns]
                ),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_ratings_distribution_square(dataset):
    square = html.Div(
        children=[
            html.H2("Distribution of ratings", className="title_square"),
            dcc.Graph(
                id="ratings_distribution",
                figure=deu.plot_ratings_distribution(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="ratings_distribution mx-auto")
            ],
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_opinions_distribution_square(dataset):
    square = html.Div(
        children=[
            html.H2("Distribution of opinions", className="title_square"),
            dcc.Graph(
                id="opinions_distribution",
                figure=deu.plot_opinions_distribution(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="opinions_distribution mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_reviews_year_distribution_square(dataset):
    square = html.Div(
        children=[
            html.H2("Distribution of reviews for each year", className="title_square"),
            dcc.Graph(
                id="reviews_year_distribution",
                figure=deu.plot_reviews_year_distribution(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="reviews_year_distribution mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square
    
    
def get_opinions_percentages_square(dataset):
    square = html.Div(
        children=[
            html.H2("Opinion percentages for each year", className="title_square"),
            dcc.Graph(
                id="opinion_percentages",
                figure=deu.plot_opinions_percentages_over_time(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="opinion_percentages mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_sentiments_distribution_square(dataset):
    square = html.Div(
        children=[
            html.H2("Distribution of sentiments", className="title_square"),
            dcc.Graph(
                id="sentiments_distribution",
                figure=dsu.plot_sentiments_distribution(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="sentiments_distribution mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sensitivity_analysis"
    )
    return square
    

def get_top_products_by_average_rating_square(dataset):
    square = html.Div(
        children=[
            html.H2("Top K products by average rating", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="top_products_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=20  # Starting value of K
            ),
            html.Div(id="top_products_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_bottom_products_by_average_rating_square(dataset):
    square = html.Div(
        children=[
            html.H2("Bottom K products by average rating", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="bottom_products_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=20  # Starting value of K
            ),
            html.Div(id="bottom_products_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square
    
    
def get_most_common_words_square(dataset):
    square = html.Div(
        children=[
            html.H2("Most common words", className="title_square"),
            html.Img(src="./assets/most_common_words_wordcloud.png", 
                     style={"display": "block", "marginLeft": "auto", "marginRight": "auto"}, 
                     className="words_wordcloud mx-auto")
            ],
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square 
    

def get_most_common_positive_words_square(dataset):
    square = html.Div(
        children=[
            html.H2("Most common words in positive reviews", className="title_square"),
            html.Img(src="./assets/most_common_positive_words_wordcloud.png", 
                     style={"display": "block", "marginLeft": "auto", "marginRight": "auto"}, 
                     className="words_wordcloud mx-auto")
            ],
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square 


def get_most_common_neutral_words_square(dataset):
    square = html.Div(
        children=[
            html.H2("Most common words in neutral reviews", className="title_square"),
            html.Img(src="./assets/most_common_neutral_words_wordcloud.png", 
                     style={"display": "block", "marginLeft": "auto", "marginRight": "auto"}, 
                     className="words_wordcloud mx-auto")
            ],
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square 


def get_most_common_negative_words_square(dataset):
    square = html.Div(
        children=[
            html.H2("Most common words in negative reviews", className="title_square"),
            html.Img(src="./assets/most_common_negative_words_wordcloud.png", 
                     style={"display": "block", "marginLeft": "auto", "marginRight": "auto"}, 
                     className="words_wordcloud mx-auto")
            ],
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square 


def get_distribution_products_prices_square(dataset):
    square = html.Div(
        children=[
            html.H2("Distribution of products by price", className="title_square"),
            dcc.Graph(
                id="products_price_distribution",
                figure=deu.plot_distribution_products_prices(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="products_price_distribution mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_scatter_plot_ratings_prices_square(dataset):
    square = html.Div(
        children=[
            html.H2("Scatter plot between ratings and prices", className="title_square"),
            dcc.Graph(
                id="scatter_plot_ratings_prices",
                figure=deu.plot_scatter_plot_ratings_prices(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="scatter_plot_ratings_prices mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square
    

def get_product_search_square(dataset):
    square = html.Div(
        children=[
            html.P("Search products by ASIN code", style={"font-weight": "bold"}, className="title_square"),
            html.Div([
                dcc.Input(
                    id="product_asin_input", 
                    type="text", 
                    placeholder="B01CGRK6O4"
                    ),
                html.Button("Search product", id="search_button", n_clicks=0),
                html.Div(id="product_info_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_exploratory_analysis"
    )
    return square


def get_top_products_by_average_sentiment_square(dataset):
    square = html.Div(
        children=[
            html.H2("Top K products by average sentiment", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="top_products_sentiment_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=30  # Starting value of K
            ),
            html.Div(id="top_products_sentiment_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_bottom_products_by_average_sentiment_square(dataset):
    square = html.Div(
        children=[
            html.H2("Bottom K products by average sentiment", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="bottom_products_sentiment_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=30  # Starting value of K
            ),
            html.Div(id="bottom_products_sentiment_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_scatter_plot_sentiments_prices_square(dataset):
    square = html.Div(
        children=[
            html.H2("Scatter plot between sentiments and prices", className="title_square"),
            html.P("Negative sentiment = -1"),
            html.P("Neutral sentiment = 0"),
            html.P("Positive sentiment = 1"),
            dcc.Graph(
                id="scatter_plot_sentiments_prices",
                figure=dsu.plot_scatter_plot_sentiments_prices(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="scatter_plot_sentiments_prices mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_sentiment_analysis_price_intervals_square(dataset):
    square = html.Div(
        children=[
            html.H2("Sentiment analysis of price intervals", className="title_square"),
            dcc.Graph(
                id="sentiment_analysis_price_intervals",
                figure=dsu.plot_sentiment_analysis_price_intervals(dataset), 
                style={"display": "block", "margin": "auto"}, 
                className="sentiment_analysis_price_intervals mx-auto")
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_product_sentiment_search_square(dataset):
    square = html.Div(
        children=[
            html.P("Search products by ASIN code", style={"font-weight": "bold"}, className="title_square"),
            html.Div([
                dcc.Input(
                    id="product_sentiment_asin_input", 
                    type="text", 
                    placeholder="B01CGRK6O4"
                    ),
                html.Button("Search product", id="search_button", n_clicks=0),
                html.Div(id="product_sentiment_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_positive_mismatch_square(dataset):
    square = html.Div(
        children=[
            html.P("Positive mismatch", 
                   style={"font-weight": "bold"}, className="title_square"),
            html.P("Get an example of a mismatch between the true positive opinion and the predicted sentiment.", 
                   className="paragraph_square"),
            html.Div([
                html.Button("Get an example", id="positive_example_button", n_clicks=0),
                html.Div(id="positive_mismatch_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_neutral_mismatch_square(dataset):
    square = html.Div(
        children=[
            html.P("Neutral mismatch", 
                   style={"font-weight": "bold"}, className="title_square"),
            html.P("Get an example of a mismatch between the true neutral opinion and the predicted sentiment.", 
                   className="paragraph_square"),
            html.Div([
                html.Button("Get an example", id="neutral_example_button", n_clicks=0),
                html.Div(id="neutral_mismatch_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_negative_mismatch_square(dataset):
    square = html.Div(
        children=[
            html.P("Negative mismatch", 
                   style={"font-weight": "bold"}, className="title_square"),
            html.P("Get an example of a mismatch between the true negative opinion and the predicted sentiment.", 
                   className="paragraph_square"),
            html.Div([
                html.Button("Get an example", id="negative_example_button", n_clicks=0),
                html.Div(id="negative_mismatch_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_sentiment_analysis"
    )
    return square


def get_most_frequent_aspects_square(dataset):
    square = html.Div(
        children=[
            html.H2("Top K most frequent aspects", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="top_frequent_aspects_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=20  # Starting value of K
            ),
            html.Div(id="top_frequent_aspects_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_absa"
    )
    return square 


def get_top_aspects_by_sentiment_square(dataset):
    square = html.Div(
        children=[
            html.H2("Top K aspects by mean sentiment", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="top_aspects_by_sentiment_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=20  # Starting value of K
            ),
            html.Div(id="top_aspects_by_sentiment_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_absa"
    )
    return square 


def get_bottom_aspects_by_sentiment_square(dataset):
    square = html.Div(
        children=[
            html.H2("Bottom K aspects by mean sentiment", className="title_square"),
            html.H6("Select the value of K with the slider:", className="slider_sentence"),
            dcc.Slider(
                id="bottom_aspects_by_sentiment_slider",
                min=1,
                max=50,
                step=1,
                marks={i: str(i) for i in range(1, 51)},
                value=20  # Starting value of K
            ),
            html.Div(id="bottom_aspects_by_sentiment_slider_output"),
            ], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_absa"
    )
    return square 


def get_aspect_sentiment_over_time_square(dataset):
    square = html.Div(
        children=[
            html.P("Plot mean aspect sentiment over time", style={"font-weight": "bold"}, className="title_square"),
            html.Div([
                dcc.Input(
                    id="aspect_sentiment_input", 
                    type="text", 
                    placeholder="tea"
                    ),
                html.Button("Insert aspect", id="aspect_sentiment_button", n_clicks=0),
                html.Div(id="aspect_sentiment_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_absa"
    )
    return square


def get_reviews_by_aspect_square(dataset):
    square = html.Div(
        children=[
            html.P("Search reviews by aspect", style={"font-weight": "bold"}, className="title_square"),
            html.Div([
                dcc.Input(
                    id="review_aspect_input", 
                    type="text", 
                    placeholder="tea"
                    ),
                html.Button("Insert aspect", id="search_review_aspect_button", n_clicks=0),
                html.Div(id="review_aspect_output")
                ])], 
        style={"width": "100%", 
               "padding": "10px", 
               "margin": "10px", 
               "border": "1px solid #ddd", 
               "border-radius": "4px", 
               "marginLeft": "auto", 
               "marginRight": "auto"},
        className="square_absa"
    )
    return square
    
    
    
    
    

    