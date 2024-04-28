import numpy as np
import pandas as pd


schools = pd.read_csv('/work/alex.unicef/capstone_project/capstone_project/data/school_data.csv')
print(f'schools in csv: {len(schools)}')

schools_bbox = np.load('/work/alex.unicef/capstone_project/capstone_project/data/embeddings/DYNOv2_bbox/school_embeds.npy')
print(f'schools in bbox: {schools_bbox.shape}')

schools_original = np.load('/work/alex.unicef/capstone_project/capstone_project/data/embeddings/DYNOv2_original/school_embeds.npy')
print(f'schools in original: {schools_original.shape}')

