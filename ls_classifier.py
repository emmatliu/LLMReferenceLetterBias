# Run example: `python3 classifier.py --task formality`

import pandas as pd
from transformers import pipeline
from collections import Counter

# (label tracked, other labels)
task_label_mapping = {
    "sentiment": ("POSITIVE", "NEGATIVE"),
    # "sentiment": ("positive", "neutral", "negative"),
    "formality": ("formal", "informal"),
}

# Define a function to perform sentiment analysis on each row of the dataframe
def predict(text, classifier, task, output_type="csv", is_sentencelevel=True):
    if is_sentencelevel:
        labels = []
        scores = []
        text = text
        sentences = text.split(".")
        for sentence in sentences:
            if len(sentence) >= 800:
                continue
            result = classifier((sentence + "."))[0]
            labels.append(result["label"])
            scores.append(result["score"])
        confidence = sum(scores) / len(scores)

        if output_type == "csv":
            mapping = Counter(labels)
            label_tracked, other_label = task_label_mapping[task]
            return (
                mapping[label_tracked]
                / (mapping[label_tracked] + mapping[other_label]),
                confidence,
            )
        # Get the most common word
        return max(set(labels), key=labels.count), confidence
    result = classifier(text)[0]
    return result["label"], result["score"]

def compute_sentiment_and_formality(df,hallucination=False):
    if hallucination:
        INPUT = 'hallucination'
    else:
        INPUT = 'text'

    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you
    classifier_sentiment = pipeline("sentiment-analysis")

    # https://huggingface.co/s-nlp/xlmr_formality_classifier
    classifier_formality = pipeline(
        "text-classification", "s-nlp/roberta-base-formality-ranker"
    )
    
    # Apply the sentiment analysis function to each row of the dataframe
    sentiment_outputs = None
    formality_outputs = None
    formality_outputs = df[INPUT].apply(
        (lambda x: predict(x, classifier_formality, "formality"))
    )
    sentiment_outputs = df[INPUT].apply(
        (lambda x: predict(x, classifier_sentiment, "sentiment"))
    )
    
    if sentiment_outputs is not None:
        df["per_pos"] = [s[0] for s in sentiment_outputs]
        df["con_pos"] = [s[1] for s in sentiment_outputs]
    if formality_outputs is not None:
        df["per_for"] = [s[0] for s in formality_outputs]
        df["con_for"] = [s[1] for s in formality_outputs]
    
    return df