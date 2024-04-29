import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
import logging
from scipy import stats

# experiment id
experiment_id = 6
info = 'cross_val'
# Set up logging to a file
logging.basicConfig(filename=f'/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/experiments/metrics_{experiment_id}_{info}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

random_seed = 42
np.random.seed(random_seed)

# Loading filenames with setting paths
school_images_path = '/work/alex.unicef/GeoAI/satellite_imagery/school'
not_school_images_path = '/work/alex.unicef/GeoAI/satellite_imagery/not_school'

data_not_schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/not_school_data.csv')
data_schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/school_data.csv')


# Loading embeddings
not_school_embeddings_path = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_original/not_school_embeds.npy'
not_school_embeddings = np.load(not_school_embeddings_path)

school_embeddings_path = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_original/school_embeds.npy'
school_embeddings = np.load(school_embeddings_path)

school_embeddings_bbox_path = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/embeddings/DYNOv2_bbox/school_embeds.npy'
school_embeddings_bbox = np.load(school_embeddings_path)


# Create a DataFrame with embeddings, labels, and indices  TRAIN
data = {
    'label': np.concatenate([np.zeros(len(not_school_embeddings)), np.ones(len(school_embeddings_bbox))]),
    'index': np.concatenate([np.arange(len(not_school_embeddings) + len(school_embeddings_bbox))]),
    'filename': data_not_schools['filename'].tolist() + data_schools['filename'].tolist()
}
df = pd.DataFrame(data)

combined_embeddings_bbox = np.concatenate((not_school_embeddings, school_embeddings_bbox), axis=0)
combined_embeddings = np.concatenate((not_school_embeddings, school_embeddings), axis=0)

def calculate_stats(measurements):
  # Calculate the mean (average) of the measurements
  mean = np.mean(measurements)

  # Define the confidence level (e.g., 95% confidence interval)
  confidence_level = 0.95

  # Calculate the standard error of the mean
  std_error = stats.sem(measurements)

  # Calculate the margin of error using the t-distribution
  margin_of_error = std_error * stats.t.ppf((1 + confidence_level) / 2, len(measurements) - 1)

  # Calculate the lower and upper bounds of the confidence interval
  lower_bound = mean - margin_of_error
  upper_bound = mean + margin_of_error

  return mean, lower_bound, upper_bound


M = 20
metrics_data = {'f1_score': [], 'precision': [], 'recall': []}
for i in range(M):

    X_train, X_test, y_train, y_test = train_test_split(df['index'], df['label'].values, test_size=0.2, shuffle=True)
    X_train_embeddings = combined_embeddings_bbox[X_train]
    x_test_embeddings = combined_embeddings[X_test]
    
    svm_model = LinearSVC(C=1, max_iter=100000, class_weight='balanced')
    svm_model.fit(X_train_embeddings, y_train)

    predicted_labels = svm_model.predict(x_test_embeddings)

    npositives = predicted_labels.sum()
    value_counts = Counter("Positive" if x > 0 else "Negative" for x in y_test)

    f1 = f1_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)

    # Log metrics to the log file
    logging.info(f"Iteration {i + 1}: F1 Score = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

    metrics_data['f1_score'].append(f1)
    metrics_data['precision'].append(precision)
    metrics_data['recall'].append(recall)

# Calculate statistics using the collected metrics
mean_f1, lower_bound_f1, upper_bound_f1 = calculate_stats(metrics_data['f1_score'])
mean_precision, lower_bound_precision, upper_bound_precision = calculate_stats(metrics_data['precision'])
mean_recall, lower_bound_recall, upper_bound_recall = calculate_stats(metrics_data['recall'])

# Log the calculated statistics to the log file
logging.info(f"Mean F1 Score = {mean_f1:.4f}, Lower Bound F1 = {lower_bound_f1:.4f}, Upper Bound F1 = {upper_bound_f1:.4f}")
logging.info(f"Mean Precision = {mean_precision:.4f}, Lower Bound Precision = {lower_bound_precision:.4f}, Upper Bound Precision = {upper_bound_precision:.4f}")
logging.info(f"Mean Recall = {mean_recall:.4f}, Lower Bound Recall = {lower_bound_recall:.4f}, Upper Bound Recall = {upper_bound_recall:.4f}")