import json
# pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import Ridge
# pip install lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from collections import defaultdict



def load_data(input_dir):
    with open(input_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [entry["prompt"] for entry in data]
    responses = [entry["response"] for entry in data]
    scores = [entry["scores"] for entry in data]

    return prompts, responses, scores

def extract_specific_scores(scores, score_name):
    return [s[score_name] for s in scores]

'''
Helper function that allows the LIME thingy to create new prompts give them
to the model and check if the models returned scores are greater or worser
with specific words swaped
'''
def predict_new_prompts_scores(new_prompts, model, vectorizer):
    vectors = vectorizer.transform(new_prompts)
    predictions = model.predict(vectors)
    result = np.zeros((len(predictions), 2))
    result[:, 1] = predictions
    return result

# Execution
input_dir = "csharp-deepseek-scores.json"
output_dir = "csharp-lime-explanations.txt"

# Parsing the data
prompts, responses, scores = load_data(input_dir)
helpfulness_scores = extract_specific_scores(scores, "helpfulness")
# correctness_scores = extract_specific_scores(scores, "correctness")
# coherence_scores = extract_specific_scores(scores, "coherence")
# complexity_scores = extract_specific_scores(scores, "complexity")
# verbosity_scores = extract_specific_scores(scores, "verbosity")

# TF-IDF vectorization
'''
TF-IDF vectorization basically does two things:
1. Term Frequency (TF) checks how often a word appears in the given prompt
2. Inverse Document Frequency (IDF) checks how often a word can appear in
   all of the data. For example the word 'the' might not be that impactful,
   since it appears really oftenly

TF-IDF vectorization combines the two above ideas and gives weights to words
in a matrix to see how impactful it is compared to other words in the given
data.
'''
vectorizer = TfidfVectorizer()
prompt_tfidf_matrix  = vectorizer.fit_transform(prompts)
helpfulness_score_targets  = np.array(helpfulness_scores)

# Training Ridge model
'''
Training the Ridge model to make connections between prompt words and the
outputed scores
'''
ridge_model = Ridge()
ridge_model.fit(prompt_tfidf_matrix, helpfulness_score_targets)

# LIME explainer
'''
LimeTextExplainer is the brain that basically can tell which words in the
prompt are the most important ones in deciding the score
'''
target_score_name = 'helpfulness'
explainer = LimeTextExplainer(class_names=[target_score_name])

# Generating explanation
'''
This part of the code takes in the given prompt, LIME does it's magic by
swapping some random word from the prompt and reevaluating how the modified
prompt did compared to the original one
'''
# with open(output_dir, "w", encoding="utf-8") as f:
#     for idx, example_prompt in enumerate(prompts):
#         # f.write(f"Explaining prompt {idx + 1}/{len(prompts)}: '{example_prompt}'\n")

#         explanation = explainer.explain_instance(
#             example_prompt,
#             lambda x: predict_new_prompts_scores(x, model=ridge_model, vectorizer=vectorizer),
#             num_features=10
#         )

#         f.write("Top features affecting prediction:\n")
#         for feature, weight in explanation.as_list():
#             f.write(f"{feature}: {weight}\n")
#         f.write("\n" + "="*60 + "\n\n")

# print(f"Explanations written to {output_dir}")

all_features = defaultdict(float)

with open(output_dir, "w", encoding="utf-8") as f:
    for idx, example_prompt in enumerate(prompts):
        explanation = explainer.explain_instance(
            example_prompt,
            lambda x: predict_new_prompts_scores(x, model=ridge_model, vectorizer=vectorizer),
            num_features=20
        )

        f.write(f"Top features affecting prediction for prompt {idx + 1}:\n")
        for feature, weight in explanation.as_list():
            f.write(f"{feature}: {weight}\n")
            # Aggregate weights for each feature
            all_features[feature] += weight
        
        f.write("\n" + "="*60 + "\n\n")

    # Sort features by their weights
    sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)

    # Separate the top 20 positive and top 20 negative features
    top_positive = sorted_features[:20]  # Top 20 features with largest positive weights
    top_negative = sorted_features[-20:]  # Top 20 features with largest negative weights

    # Prepare data for bar charts
    positive_features, positive_weights = zip(*top_positive)
    negative_features, negative_weights = zip(*top_negative)

    # Create a bar chart for the top 20 positive features
    plt.figure(figsize=(12, 8))
    plt.barh(positive_features, positive_weights, color='lightgreen')
    plt.xlabel('Feature Importance')
    plt.title('DeepSeek Features with the Largest Positive Weights for Helpfulness')
    plt.tight_layout()
    plt.savefig(f"deepseek-top_positive_bar_chart.png")
    plt.close()

    # Create a bar chart for the top 20 negative features
    plt.figure(figsize=(12, 8))
    plt.barh(negative_features, negative_weights, color='salmon')
    plt.xlabel('Feature Importance')
    plt.title('DeepSeek Features with the Largest Negative Weights for Helpfulness')
    plt.tight_layout()
    plt.savefig(f"deepseek-top_negative_bar_chart.png")
    plt.close()

print(f"Explanations written to {output_dir}")