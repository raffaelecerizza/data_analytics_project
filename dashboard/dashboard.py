import os
import ast
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import dashboard_utils as du
import dashboard_exploratory_utils as deu
import dashboard_sentiment_utils as dsu
import dashboard_absa_utils as dau

from dash import Dash, Input, Output, State, dcc, html, dash_table


# Constants.
PREPROCESSED_DATASET_PATH = "./dataset/dataset.csv"
GENERAL_INFO_PATH = "./dataset/general_info.csv"
PRODUCTS_DATASET_PATH = "./dataset/meta_Grocery_and_Gourmet_Food.json"
DATASET_WITH_SENTIMENT_PATH = "./dataset/dataset_with_sentiment.csv"
ABSA_DATASET_PATH = "./dataset/absa_dataset.csv"

# Import data.
general_info = pd.read_csv(GENERAL_INFO_PATH)
dataset = pd.read_csv(ABSA_DATASET_PATH)
dataset = dataset.sample(n=100000, random_state=42)
dataset_sentiment = dataset
dataset_absa = dataset
#dataset_sentiment = pd.read_csv(DATASET_WITH_SENTIMENT_PATH)
#dataset_absa = pd.read_csv(ABSA_DATASET_PATH)
#products_dataset = pd.read_json(PRODUCTS_DATASET_PATH, lines=True)

# We set the path for the assets folder.
# CURRENT DIRECTORY MUST BE THAT OF THE PROJECT (NOT OF THE DASHBOARD).
assets_path = os.getcwd() +"/assets"

# General style for the dashboard.
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    assets_folder=assets_path,
    suppress_callback_exceptions=True
)

# Style arguments for the sidebar. 
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Style for the content of the sidebar.
CONTENT_STYLE = {
    "margin-left": "18rem",  
    "margin-right": "2rem",  
    "padding": "2rem 1rem",
    "max-width": "100%",
    "overflow-x": "hidden"
}

# Sidebar definition.
sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Exploratory analysis", href="/exploratory_analysis", active="exact"),
                dbc.NavLink("Sentiment analysis", href="/sentiment_analysis", active="exact"),
                dbc.NavLink("Aspect-based sentiment analysis", href="/absa_analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
sidebar_content = html.Div(id="sidebar_content", style=CONTENT_STYLE)


app.layout = html.Div(children=[
    dcc.Location(id="url", refresh=False),
    html.Div(children=[
        html.Div(children=[
            dcc.Location(id="sidebar_location"),
            sidebar, 
            sidebar_content
        ], className= "sidebar_content"),
    ], className="sidebar"),
])

home_page_layout = [
    html.Div(children=[
        html.Br(),
        html.Br(),
        html.H1("Amazon Fine Food Reviews", style={"font-weight": "bold", "textAlign": "center"}, className="title_home"),
        html.Br(),
        html.Div(children=[
            html.Img(src="./assets/serving_tray.png", style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "25%"}),
            html.Br(),
            html.P("Raffaele Cerizza - 845512", style={"font-size": "small", "textAlign": "center"})
        ], style={"textAlign": "center"})
    ], className="home_page")
]
    
exploratory_analysis_layout = [
    html.Div(children=[
        html.H1("Exploratory analysis", style={"font-weight": "bold", "textAlign": "center"}, className="title_analysis"),
        du.get_general_info_square(general_info),
        dcc.Tabs(
            id="exploratory_analysis_tabs", 
            value="reviews", 
            children=[
                dcc.Tab(label="Reviews", value="reviews"),
                dcc.Tab(label="Products", value="products"),
                dcc.Tab(label="Prices", value="prices"),
                dcc.Tab(label="Words", value="words"),
                ]
            ),
        html.Div(id="exploratory_analysis_tabs_content")
    ], className="exploratory_analysis")
]

reviews_tab_layout = [
    html.Div(children=[
        du.get_ratings_distribution_square(dataset),
        du.get_opinions_distribution_square(dataset),
        du.get_reviews_year_distribution_square(dataset),
        du.get_opinions_percentages_square(dataset)
    ], className="reviews_tab_layout")
]

products_tab_layout = [
    html.Div(children=[
        du.get_product_search_square(dataset),
        du.get_top_products_by_average_rating_square(dataset),
        du.get_bottom_products_by_average_rating_square(dataset)
    ], className="products_tab_layout")
]

prices_tab_layout = [
    html.Div(children=[
        du.get_distribution_products_prices_square(dataset),
        du.get_scatter_plot_ratings_prices_square(dataset)
    ], className="prices_tab_layout")
]

words_tab_layout = [
    html.Div(children=[
        du.get_most_common_words_square(dataset),
        du.get_most_common_positive_words_square(dataset),
        du.get_most_common_neutral_words_square(dataset),
        du.get_most_common_negative_words_square(dataset)
    ], className="words_tab_layout")
]

sentiment_analysis_layout = [
    html.Div(children=[
        html.H1("Sentiment analysis", style={"font-weight": "bold", "textAlign": "center"}, className="title_analysis"),
        dcc.Tabs(
            id="sentiment_analysis_tabs", 
            value="results", 
            children=[
                dcc.Tab(label="Results", value="results"),
                dcc.Tab(label="Search", value="search"),
                dcc.Tab(label="Mismatches", value="mismatches"),
                ]
            ),
        html.Div(id="sentiment_analysis_tabs_content")
    ], className="sentiment_analysis")
]

sentiment_results_tab_layout = [
    html.Div(children=[
        du.get_sentiments_distribution_square(dataset_sentiment),
        du.get_top_products_by_average_sentiment_square(dataset_sentiment),
        du.get_bottom_products_by_average_sentiment_square(dataset_sentiment),
        du.get_scatter_plot_sentiments_prices_square(dataset_sentiment),
        #du.get_sentiment_analysis_price_intervals_square(dataset_sentiment)
    ], className="sentiment_results_tab_layout")
]

sentiment_search_tab_layout = [
    html.Div(children=[
        du.get_product_sentiment_search_square(dataset_sentiment)
    ], className="sentiment_search_tab_layout")
]

sentiment_mismatches_tab_layout = [
    html.Div(children=[
        du.get_positive_mismatch_square(dataset_sentiment),
        du.get_neutral_mismatch_square(dataset_sentiment),
        du.get_negative_mismatch_square(dataset_sentiment)
    ], className="sentiment_mismatches_tab_layout")
]

absa_layout = [
    html.Div(children=[
        html.H1("Aspect-based sentiment analysis", style={"font-weight": "bold", "textAlign": "center"}, className="title_analysis"),
        dcc.Tabs(
            id="absa_tabs", 
            value="results", 
            children=[
                dcc.Tab(label="Results", value="results"),
                dcc.Tab(label="Search", value="search")
                ]
            ),
        html.Div(id="absa_tabs_content")
    ], className="absa")
]

absa_results_tab_layout = [
    html.Div(children=[
       du.get_most_frequent_aspects_square(dataset_absa),
       du.get_top_aspects_by_sentiment_square(dataset_absa),
       du.get_bottom_aspects_by_sentiment_square(dataset_absa),
    ], className="absa_results_tab_layout")
]

absa_search_tab_layout = [
    html.Div(children=[
       du.get_aspect_sentiment_over_time_square(dataset_absa),
       du.get_reviews_by_aspect_square(dataset_absa)
    ], className="absa_search_tab_layout")
]


# Callback for the sidebar.
@app.callback(
    Output("sidebar_content", "children"), 
    [Input("sidebar_location", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return home_page_layout
    elif pathname == "/exploratory_analysis":
        return exploratory_analysis_layout
    elif pathname == "/sentiment_analysis":
        return sentiment_analysis_layout
    elif pathname == "/absa_analysis":
        return absa_layout
    
    
# Callback for the tabs in the exploratory analysis.
@app.callback(
    Output("exploratory_analysis_tabs_content", "children"),
    [Input("exploratory_analysis_tabs", "value")]
)
def render_tab_content(selected_tab):
    if selected_tab == "reviews":
        return reviews_tab_layout
    elif selected_tab == "products":
        return products_tab_layout
    elif selected_tab == "prices":
        return prices_tab_layout
    elif selected_tab == "words":
        return words_tab_layout
    
    
# Callback for changing the K value for the top rated products.
@app.callback(
    Output("top_products_slider_output", "children"),
    [Input("top_products_slider", "value")]
)
def update_top_products_figure(value):
    return dcc.Graph(
        figure=deu.plot_top_rated_products(dataset, k=value), 
        className="top_products mx-auto"
    )
    
    
# Callback for changing the K value for the bottom rated products.
@app.callback(
    Output("bottom_products_slider_output", "children"),
    [Input("bottom_products_slider", "value")]
)
def update_bottom_products_figure(value):
    return dcc.Graph(
        figure=deu.plot_bottom_rated_products(dataset, k=value), 
        className="bottom_products mx-auto"
    )
    
    
# Callback for searching products.
@app.callback(
    Output("product_info_output", "children"),
    [Input("search_button", "n_clicks")],
    [State("product_asin_input", "value")]
)
def search_product(n_clicks, product_code):
    if n_clicks > 0:
        product_row = dataset[dataset["asin"] == product_code]
        
        # If there are no products with the specified ASIN, we return an error message.
        if product_row.empty:
            return html.Div(children=[
                html.Br(),
                html.P([
                html.Span("There are no products with the specified ASIN!", style={"font-weight": "bold", "display": "inline"}),
                ], className="paragraph_square")
            ])
        
        product_row = product_row[["asin", "title", "description", "price", "imageURLHighRes"]]
        product_row = product_row.head(1)
        product_asin = product_row["asin"].values[0]
        product_title = product_row["title"].values[0]
        product_description = product_row["description"].values[0]
        product_description = eval(product_description)
        product_description = ' '.join(product_description)
        product_price = product_row["price"].values[0]
        product_image = product_row["imageURLHighRes"].values[0]
        product_image = eval(product_image)
        if len(product_image) > 0:
            product_image = product_image[0]
        else:
            product_image = ""
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold"}),
                product_asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Title: ", style={"font-weight": "bold"}),
                product_title
                ], className="paragraph_square"),
            html.P([
                html.Span("Description: ", style={"font-weight": "bold"}),
                product_description
                ], className="paragraph_square"),
            html.P([
                html.Span("Price ($): ", style={"font-weight": "bold"}),
                product_price
                ], className="paragraph_square"),
            html.Img(
                src=product_image,
                style={"display": "block", "margin": "auto"},
                className="product_image mx-auto"
            )
        ]) 
        return search_result
      
      
# Callback for the tabs in the sentiment analysis.
@app.callback(
    Output("sentiment_analysis_tabs_content", "children"),
    [Input("sentiment_analysis_tabs", "value")]
)
def render_tab_content(selected_tab):
    if selected_tab == "results":
        return sentiment_results_tab_layout
    elif selected_tab == "search":
        return sentiment_search_tab_layout
    elif selected_tab == "mismatches":
        return sentiment_mismatches_tab_layout
    
    
# Callback for changing the K value for the top products by average sentiment.
@app.callback(
    Output("top_products_sentiment_slider_output", "children"),
    [Input("top_products_sentiment_slider", "value")]
)
def update_top_products_sentiment_figure(value):
    return dcc.Graph(
        figure=dsu.plot_top_products_by_sentiment(dataset_sentiment, k=value), 
        className="top_products mx-auto"
    )
    
    
# Callback for changing the K value for the bottom products by average sentiment.
@app.callback(
    Output("bottom_products_sentiment_slider_output", "children"),
    [Input("bottom_products_sentiment_slider", "value")]
)
def update_bottom_products_sentiment_figure(value):
    return dcc.Graph(
        figure=dsu.plot_bottom_products_by_sentiment(dataset_sentiment, k=value), 
        className="bottom_products mx-auto"
    )
    
    
# Callback for searching sentiments of products.
@app.callback(
    Output("product_sentiment_output", "children"),
    [Input("search_button", "n_clicks")],
    [State("product_sentiment_asin_input", "value")]
)
def search_sentiment_product(n_clicks, product_code):
    if n_clicks > 0:
        product_reviews = dataset_sentiment[dataset_sentiment["asin"] == product_code]
        
        # If there are no products with the specified ASIN, we return an error message.
        if product_reviews.empty:
            return html.Div(children=[
                html.Br(),
                html.P([
                html.Span("There are no reviews for the specified ASIN!", style={"font-weight": "bold", "display": "inline"}),
                ], className="paragraph_square")
            ])
        
        product_reviews = product_reviews[["asin", "reviewText", "rating", "predictedSentiment"]]
        # We permute the rows in order to get different results for the head(1).
        product_reviews = product_reviews.sample(frac=1).reset_index(drop=True)
        product_reviews = product_reviews.head(1)

        product_asin = product_reviews["asin"].values[0]
        product_review = product_reviews["reviewText"].values[0]
        product_rating = product_reviews["rating"].values[0]
        product_sentiment = product_reviews["predictedSentiment"].values[0]
        
        color = "lightgreen"
        if product_sentiment == "neutral":
            color = "skyblue"
        elif product_sentiment == "negative":
            color = "salmon"
        
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold"}),
                product_asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Review: ", style={"font-weight": "bold", "display": "inline"}),
                html.P(product_review, style={"textAlign": "justify", "display": "inline"})
                ], className="paragraph_square"),
            html.P([
                html.Span("Rating: ", style={"font-weight": "bold"}),
                product_rating
                ], className="paragraph_square"),
            html.P([
                html.Span("Predicted sentiment: ", style={"font-weight": "bold"}),
                html.Span(product_sentiment, style={"color": color, "font-weight": "bold"})
                ], className="paragraph_square"),
        ]) 
        return search_result
    
    
# Callback for positive mismatches.
@app.callback(
    Output("positive_mismatch_output", "children"),
    [Input("positive_example_button", "n_clicks")]
)
def find_positive_mismatch(n_clicks):
    if n_clicks > 0:
        reviews = dataset_sentiment[(dataset_sentiment["opinion"] == "positive") & (dataset_sentiment["predictedSentiment"] != "positive")]
        # We permute the rows in order to get different results for the head(1).
        reviews = reviews.sample(frac=1).reset_index(drop=True)
        reviews = reviews.head(1)

        asin = reviews["asin"].values[0]
        review = reviews["reviewText"].values[0]
        rating = reviews["rating"].values[0]
        sentiment = reviews["predictedSentiment"].values[0]
        
        color = "salmon"
        if sentiment == "neutral":
            color = "skyblue"
        
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold"}),
                asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Review: ", style={"font-weight": "bold", "display": "inline"}),
                html.P(review, style={"textAlign": "justify", "display": "inline"})
                ], className="paragraph_square"),
            html.P([
                html.Span("Rating: ", style={"font-weight": "bold"}),
                rating
                ], className="paragraph_square"),
            html.P([
                html.Span("Predicted sentiment: ", style={"font-weight": "bold", "display": "inline"}),
                html.Span(sentiment, style={"color": color, "font-weight": "bold"})
                ], className="paragraph_square"),
        ]) 
        return search_result
    
    
# Callback for neutral mismatches.
@app.callback(
    Output("neutral_mismatch_output", "children"),
    [Input("neutral_example_button", "n_clicks")]
)
def find_neutral_mismatch(n_clicks):
    if n_clicks > 0:
        reviews = dataset_sentiment[(dataset_sentiment["opinion"] == "neutral") & (dataset_sentiment["predictedSentiment"] != "neutral")]
        # We permute the rows in order to get different results for the head(1).
        reviews = reviews.sample(frac=1).reset_index(drop=True)
        reviews = reviews.head(1)

        asin = reviews["asin"].values[0]
        review = reviews["reviewText"].values[0]
        rating = reviews["rating"].values[0]
        sentiment = reviews["predictedSentiment"].values[0]
        
        color = "lightgreen"
        if sentiment == "negative":
            color = "salmon"
        
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold"}),
                asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Review: ", style={"font-weight": "bold", "display": "inline"}),
                html.P(review, style={"textAlign": "justify", "display": "inline"})
                ], className="paragraph_square"),
            html.P([
                html.Span("Rating: ", style={"font-weight": "bold"}),
                rating
                ], className="paragraph_square"),
            html.P([
                html.Span("Predicted sentiment: ", style={"font-weight": "bold", "display": "inline"}),
                html.Span(sentiment, style={"color": color, "font-weight": "bold"})
                ], className="paragraph_square"),
        ]) 
        return search_result
    
    
# Callback for negative mismatches.
@app.callback(
    Output("negative_mismatch_output", "children"),
    [Input("negative_example_button", "n_clicks")]
)
def find_negative_mismatch(n_clicks):
    if n_clicks > 0:
        reviews = dataset_sentiment[(dataset_sentiment["opinion"] == "negative") & (dataset_sentiment["predictedSentiment"] != "negative")]
        # We permute the rows in order to get different results for the head(1).
        reviews = reviews.sample(frac=1).reset_index(drop=True)
        reviews = reviews.head(1)

        asin = reviews["asin"].values[0]
        review = reviews["reviewText"].values[0]
        rating = reviews["rating"].values[0]
        sentiment = reviews["predictedSentiment"].values[0]
        
        color = "lightgreen"
        if sentiment == "neutral":
            color = "skyblue"
        
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold", "display": "inline"}),
                asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Review: ", style={"font-weight": "bold", "display": "inline"}),
                html.P(review, style={"textAlign": "justify", "display": "inline"})
                ], className="paragraph_square"),
            html.P([
                html.Span("Rating: ", style={"font-weight": "bold"}),
                rating
                ], className="paragraph_square"),
            html.P([
                html.Span("Predicted sentiment: ", style={"font-weight": "bold", "display": "inline"}),
                html.Span(sentiment, style={"color": color, "font-weight": "bold"})
                ], className="paragraph_square"),
        ]) 
        return search_result
    
    
# Callback for the tabs in the aspect-based sentiment analysis.
@app.callback(
    Output("absa_tabs_content", "children"),
    [Input("absa_tabs", "value")]
)
def render_tab_content(selected_tab):
    if selected_tab == "results":
        return absa_results_tab_layout
    elif selected_tab == "search":
        return absa_search_tab_layout
    

# Callback for changing the K value for the most frequent aspects.
@app.callback(
    Output("top_frequent_aspects_slider_output", "children"),
    [Input("top_frequent_aspects_slider", "value")]
)
def update_top_frequent_aspects_figure(value):
    return dcc.Graph(
        figure=dau.plot_most_frequent_aspects(dataset_absa, k=value), 
        className="top_aspects mx-auto"
    )
    
    
# Callback for changing the K value for the top aspects by sentiment.
@app.callback(
    Output("top_aspects_by_sentiment_slider_output", "children"),
    [Input("top_aspects_by_sentiment_slider", "value")]
)
def update_top_aspects_by_sentiment_figure(value):
    return dcc.Graph(
        figure=dau.plot_top_aspects_by_sentiment(dataset_absa, k=value), 
        className="top_aspects mx-auto"
    )
    
    
# Callback for changing the K value for the bottom aspects by sentiment.
@app.callback(
    Output("bottom_aspects_by_sentiment_slider_output", "children"),
    [Input("bottom_aspects_by_sentiment_slider", "value")]
)
def update_bottom_aspects_by_sentiment_figure(value):
    return dcc.Graph(
        figure=dau.plot_bottom_aspects_by_sentiment(dataset_absa, k=value), 
        className="bottom_aspects mx-auto"
    )
    
    
# Callback for plotting aspect sentiment over time.
@app.callback(
    Output("aspect_sentiment_output", "children"),
    [Input("aspect_sentiment_button", "n_clicks")],
    [State("aspect_sentiment_input", "value")]
)
def plot_aspect_sentiment(n_clicks, aspect):
    if n_clicks > 0:
        # We filter the dataset in order to consider only the reviews with the specific aspect.
        reviews = dataset_absa[dataset_absa["aspects"].apply(lambda x: aspect in eval(x))]
        
        # If there are no reviews with the specified aspect, we return an error message.
        if reviews.empty:
            return html.Div(children=[
                html.Br(),
                html.P([
                html.Span("There are no reviews with the specified aspect!", style={"font-weight": "bold", "display": "inline"}),
                ], className="paragraph_square")
            ])
        
        return dcc.Graph(
            figure=dau.plot_aspect_sentiment_over_time(dataset_absa, aspect=aspect), 
            className="aspect_sentiment_over_time mx-auto"
        )
        

# Callback for searching reviews by aspect.
@app.callback(
    Output("review_aspect_output", "children"),
    [Input("search_review_aspect_button", "n_clicks")],
    [State("review_aspect_input", "value")]
)
def search_reviews_by_aspect(n_clicks, aspect):
    if n_clicks > 0:
        # We filter the dataset in order to consider only the reviews with the specific aspect.
        reviews = dataset_absa[dataset_absa["aspects"].apply(lambda x: aspect in eval(x))]  
        
        # If there are no reviews with the specified aspect, we return an error message.
        if reviews.empty:
            return html.Div(children=[
                html.Br(),
                html.P([
                html.Span("There are no reviews with the specified aspect!", style={"font-weight": "bold", "display": "inline"}),
                ], className="paragraph_square")
            ])
        
        # We permute the rows in order to get different results for the head(1).
        reviews = reviews.sample(frac=1).reset_index(drop=True)
        reviews = reviews.head(1)

        asin = reviews["asin"].values[0]
        review = reviews["reviewText"].values[0]
        rating = reviews["rating"].values[0]

        aspects = reviews["aspects"].values[0]
        aspects = eval(aspects)
        aspects_text = ', '.join(aspects)
        
        sentiments = reviews["sentiments"].values[0]
        sentiments = eval(sentiments)
        sentiments = [string.lower() for string in sentiments]

        colors = []
        for sentiment in sentiments:
            if sentiment == "positive":
                colors.append("lightgreen")
            elif sentiment == "neutral":
                colors.append("skyblue")
            else:
                colors.append("salmon")
                
        sentiments_paragraphs = []
        for i in range(len(sentiments)-1):
            sentiments_paragraphs.append(html.P(
                sentiments[i] + ", ", 
                style={"color": colors[i], "font-weight": "bold", "display": "inline"},
                className="paragraph_square"))
        sentiments_paragraphs.append(html.P(
                sentiments[len(sentiments)-1], 
                style={"color": colors[len(colors)-1], "font-weight": "bold", "display": "inline"},
                className="paragraph_square"))
        
        search_result = html.Div(children=[
            html.Br(),
            html.P([
                html.Span("ASIN: ", style={"font-weight": "bold", "display": "inline"}),
                asin
                ], className="paragraph_square"),
            html.P([
                html.Span("Review: ", style={"font-weight": "bold", "display": "inline"}),
                html.P(review, style={"textAlign": "justify", "display": "inline"})
                ], className="paragraph_square"),
            html.P([
                html.Span("Rating: ", style={"font-weight": "bold"}),
                rating
                ], className="paragraph_square"),
            html.P([
                html.Span("Aspects: ", style={"font-weight": "bold"}),
                aspects_text
                ], className="paragraph_square"),
            html.Div([
                html.Span("Sentiments: ", style={"font-weight": "bold", "display": "inline"}),
                *sentiments_paragraphs
                ], className="paragraph_square"),
        ]) 
        return search_result
        
    
    

if __name__ == "__main__":
    app.run_server(debug=True)