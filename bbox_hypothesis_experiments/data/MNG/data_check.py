import numpy as np
import pandas as pd


schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/school_data.csv')
print(f'schools in csv: {len(schools)}')

# schools_bbox = np.load('/work/alex.unicef/capstone_project/capstone_project/data/embeddings/DYNOv2_bbox/school_embeds.npy')
# print(f'schools in bbox: {schools_bbox.shape}')

schools_original = np.load('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/embeddings/DYNOv2_original/school_embeds.npy')
print(f'schools in original: {schools_original.shape}')



not_schools = pd.read_csv('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/not_school_data.csv')
print(f'not schools in csv: {len(not_schools)}')

not_schools_original = np.load('/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/embeddings/DYNOv2_original/not_school_embeds.npy')
print(f'not_schools in original: {not_schools_original.shape}')

