import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def run_inference(df, INPUT, TASK, classifier, label_mapping, rev_map, task_label_mapping, is_sentencelevel=True):
    inferences = []
    for i in tqdm(range(len(df)), ascii=True):
        if is_sentencelevel:
            labels = []
            scores = []
            sentences = df.iloc[i, :][INPUT].split(".")
            for sentence in sentences:
                if len(sentence) >= 800:
                    continue
                output = classifier((sentence + ".").lower())[0]
                labels.append(label_mapping[TASK][rev_map[output["label"]]])
                scores.append(output["score"])
            confidence = sum(scores) / len(scores)
            mapping = Counter(labels)
            label_tracked, other_label = task_label_mapping[TASK]
            inferences.append(
                (
                    mapping[label_tracked]
                    / (mapping[label_tracked] + mapping[other_label]),
                    confidence,
                )
            )
        else:
            output = classifier(df.iloc[i, :][INPUT])[0]
            inferences.append(
                (label_mapping[TASK][rev_map[output["label"]]], output["score"])
            )

    return inferences

# TODO: remove when model is fixed :/
def compute_agentic_communal(df, hallucination=False):
    df['per_ac'] = np.random.rand(len(df))
    df['con_ac'] = np.random.rand(len(df))
    return df

# Need clarification on model lol
# def compute_agentic_communal(df,hallucination=False):
#     model_path = "./checkpoints/checkpoint-48" # 
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
#     rev_map = {v: k for k, v in model.config.id2label.items()}

#     if hallucination:
#         INPUT = "hallucination"
#     else:
#         INPUT = "TEXT" # need to tell users what this should be called TODO: change this to the correct column name

#     TASK = "ac_classifier"
#     task_label_mapping = {
#         # Track percentage agentic / percentage agentic + percentage communal
#         "ac_classifier": ("agentic", "communal"),
#     }
#     label_mapping = {
#         "ac_classifier": {
#             0: "communal",
#             1: "agentic",
#         }
#     }

#     inferences = run_inference(df, INPUT, TASK, classifier, label_mapping, rev_map, task_label_mapping)
#     df["per_ac"] = [i[0] for i in inferences]
#     df["con_ac"] = [i[1] for i in inferences]

#     return df