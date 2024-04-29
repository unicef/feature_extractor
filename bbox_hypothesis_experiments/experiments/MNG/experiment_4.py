import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import logging

# experiment id
experiment_id = 4
info = ''
# Set up logging to a file
logging.basicConfig(filename=f'/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/experiments/MNG/experiment_{experiment_id}_{info}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Loading filenames with setting paths
school_images_path = '/work/alex.unicef/raw_data/MNG/school'
not_school_images_path = '/work/alex.unicef/raw_data/MNG/not_school'
data_schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/school_data.csv')
data_not_schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/not_school_data.csv')

# Loading embeddings
not_school_embeddings_path = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/embeddings/DYNOv2_original/not_school_embeds.npy'
school_embeddings_path = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/embeddings/DYNOv2_original/school_embeds.npy'
not_school_embeddings = np.load(not_school_embeddings_path)
school_embeddings = np.load(school_embeddings_path)

print("not_school_embeddings shape: ", not_school_embeddings.shape)
print("school_embeddings shape: ", school_embeddings.shape)

# Combine embeddings and create DataFrame
combined_embeddings = np.concatenate((not_school_embeddings, school_embeddings), axis=0)
data = {
    'label': np.concatenate([np.zeros(len(not_school_embeddings)), np.ones(len(school_embeddings))]),
    'filename': data_not_schools['filename'].tolist() + data_schools['filename'].tolist()
}
df = pd.DataFrame(data)

# Define the SVM model
svm_model = LinearSVC(C=1, max_iter=100000, class_weight='balanced')

# Define evaluation metrics (F1 score, precision, recall)
scoring = {
    'f1_score': make_scorer(f1_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score)
}

# Perform cross-validation
cv_results = cross_validate(svm_model, combined_embeddings, df['label'], cv=5, scoring=scoring)

# Log the cross-validation results
for fold, f1, precision, recall in zip(range(1, 6), cv_results['test_f1_score'], cv_results['test_precision'], cv_results['test_recall']):
    logging.info(f"Fold {fold}: F1 Score = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

# Calculate mean and standard deviation of the evaluation metrics
mean_f1 = np.mean(cv_results['test_f1_score'])
mean_precision = np.mean(cv_results['test_precision'])
mean_recall = np.mean(cv_results['test_recall'])
std_f1 = np.std(cv_results['test_f1_score'])
std_precision = np.std(cv_results['test_precision'])
std_recall = np.std(cv_results['test_recall'])

# Log the mean and standard deviation of evaluation metrics
logging.info(f"Mean F1 Score = {mean_f1:.4f} +/- {std_f1:.4f}")
logging.info(f"Mean Precision = {mean_precision:.4f} +/- {std_precision:.4f}")
logging.info(f"Mean Recall = {mean_recall:.4f} +/- {std_recall:.4f}")
