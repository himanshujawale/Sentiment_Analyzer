import torch
import gradio as gr
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline

def sentiment_bar_chart(df):
    """
    Takes a DataFrame with two columns: review and sentiment
    (positive / negative) and returns a Plotly bar chart object.
    """
    # Use column positions so it works whatever the headers are called
    sentiment_col = df.columns[1]

    # Count positive and negative reviews (case/space safe)
    counts = (
        df[sentiment_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .reindex(["positive", "negative"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Sentiment", "Count"]

    # Build the bar chart
    # fig = px.bar(
    #     counts,
    #     x="Sentiment",
    #     y="Count",
    #     color="Sentiment",
    #     color_discrete_map={"positive": "green", "negative": "red"},
    #     text="Count",
    #     title="Review Sentiment Counts",
    # )
    # fig.update_layout(showlegend=False)
    #
    # return fig
# Build the pie chart
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={"positive": "green", "negative": "red"},
        title="Review Sentiment Distribution",
        hole=0
    )
    fig.update_traces(textinfo="label+percent+value", textposition="inside")
    return fig


# use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# sentiment_analyzer = pipeline(
#     "sentiment-analysis",
#     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# )

analyzer =pipeline("sentiment-analysis", model=model_path)

analyzer.model.save_pretrained("../Models/sentiment-model")
analyzer.tokenizer.save_pretrained("../Models/sentiment-model")

# print(analyzer(["this production is ok", "this product is cheap"]))

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def read_reviews_and_analyze_sentiment(file_path, review_column="Reviews"):
    """
    Read reviews from an Excel file and return a DataFrame
    containing the reviews and their corresponding sentiments.

    Parameters:
        file_path (str): Path to the Excel file.
        review_column (str): Name of the column containing reviews.

    Returns:
        pd.DataFrame: DataFrame containing reviews and sentiments.
    """

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Validate the review column
    if 'Review' not in df.columns:
        raise ValueError("Excel file must contain a 'Review' column.")

    # Keep the original reviews
    # df = df[[review]].copy()

    # Apply sentiment analysis
    df["Sentiment"] = df['Review'].apply(sentiment_analyzer)
    chart_object = sentiment_bar_chart(df)

    return df,  chart_object

result = read_reviews_and_analyze_sentiment("../Files/Prod-review.xlsx")
print(result)


# Example usage


demo = gr.Interface(fn=read_reviews_and_analyze_sentiment,
                    # inputs=[gr.Textbox(label="Input your review comment",lines=4)],
                    inputs=[gr.File(file_types=[".xlsx",".csv"], label="UPLOAD YOUR REVIEW COMMENT FILE")],
                    # outputs=[gr.Textbox(label="Sentiment",lines=1)],
                    outputs=[gr.DataFrame(label="Sentiments"), gr.Plot(label="Sentiment Analysis")],
                    title="@Project 3: sentiment analyzer",
                    description="THIS APPLICATION WILL BE USED TO ANALYZE THE SENTIMENT BASED ON FILE UPLOADED.")

demo.launch()




