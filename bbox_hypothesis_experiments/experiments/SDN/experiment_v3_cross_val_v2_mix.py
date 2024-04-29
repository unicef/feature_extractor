import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import logging

from tqdm import tqdm

# experiment id
experiment_id = 3
info = 'mix'

# Set up logging to a file
logging.basicConfig(filename=f'/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/experiments/metrics_{experiment_id}_{info}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

random_seed = 42
np.random.seed(random_seed)

# Set paths 
school_images_path = '/work/alex.unicef/GeoAI/satellite_imagery/school'
not_school_images_path = '/work/alex.unicef/GeoAI/satellite_imagery/not_school'

# Read data
school_dataset = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/school_data.csv')
school_embedds_original = np.load('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_original/school_embeds.npy')
school_embedds_bbox = np.load('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_bbox/school_embeds.npy')

not_school_dataset = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/not_school_data.csv')
not_school_embedds = np.load('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_original/not_school_embeds.npy')

combined_embeddings_original = np.concatenate((not_school_embedds, school_embedds_original), axis=0)
combined_embeddings_bbox = np.concatenate((not_school_embedds, school_embedds_bbox), axis=0)

# Create a DataFrame with embeddings, labels, and indices
data = {
    'label': np.concatenate([np.zeros(len(not_school_embedds)), np.ones(len(school_embedds_original))]),
    'index': np.concatenate([np.arange(len(not_school_embedds) + len(school_embedds_original))]),
    'filename': not_school_dataset['filename'].tolist() + school_dataset['filename'].tolist()
}

df = pd.DataFrame(data)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define evaluation metrics (F1 score, precision, recall)
scoring = {
    'f1_score': make_scorer(f1_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score)
}
  
N = 2000
metrics_data = {'n': [], 'f1_score': [], 'precision': [], 'recall': []}
selected_indices = []
scores = []
delta = 5
threshold = 0.9

for n in tqdm(range(10, N, delta)):
    # Select new schools among those the model is less sure about
    df_train = df.copy()
    df_train = df_train.drop(selected_indices)

    if len(scores) > 0 and len(scores[scores > 0]) > 0:      
        items_df = pd.DataFrame(data={
            'proba': sigmoid(scores),
            'score': scores}).sort_values(by='proba')
        items_df = items_df.drop(selected_indices)
        items_df = items_df[(items_df['proba'] > 0.5) & (items_df['proba'] < threshold)]
        less_sure_positive_indices = np.array(items_df.index)
        additional_indices = less_sure_positive_indices[:delta]
        selected_indices = np.concatenate([selected_indices, additional_indices])  
        df_train = df_train.drop(additional_indices)

      
    m = n - len(selected_indices)
    additional_indices = np.random.choice(df_train[df_train['label'] == 1].index, m, replace=False)  
    selected_indices = np.concatenate([selected_indices, additional_indices])
    df_train = df.copy()
    
    df_train.loc[selected_indices, 'label'] = 1
    df_train.loc[~df_train.index.isin(selected_indices), 'label'] = 0
    labels = df_train['label'].values

    # Initialize and train an SVM classifier with C=1
    svm_model = LinearSVC(C=1, max_iter=100000, class_weight='balanced', dual='auto')
    svm_model.fit(combined_embeddings_bbox, labels)
    predicted_labels = svm_model.predict(combined_embeddings_original) 
    scores = svm_model.decision_function(combined_embeddings_original)
    
    # Calculate the F1 score on the validation data
    f1 = f1_score(df['label'].values, predicted_labels)
    precision = precision_score(df['label'].values, predicted_labels)
    recall = recall_score(df['label'].values, predicted_labels)

    # Log the mean and standard deviation of evaluation metrics
    logging.info(f"n = {n:.4f}")
    logging.info(f"F1 Score = {f1:.4f}")
    logging.info(f"Precision = {precision:.4f}")
    logging.info(f"Recall = {recall:.4f}")