import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torch.multiprocessing as mp  # Add this import

# Set the multiprocessing start method to 'spawn'
if __name__ == '__main__':
    mp.set_start_method('spawn')

# path to metadata
school_data = pd.read_csv('/work/alex.unicef/capstone_project/data_original_CLIP/school_data.csv')
# path to images
images_path = '/work/alex.unicef/GeoAI/satellite_imagery/school/'
# path to output
output_path = '/work/alex.unicef/capstone_project/capstone_project/embeddings/tmp_output/school_embeds.npy'

"""## Create custom augmentation"""
class BboxCrop(object):
    def __init__(self, ):
      pass

    def __call__(self, img):
      string_arr = img.split('/')[-1]
      cwd = '/work/alex.unicef/'
      bbox_name = os.path.join(cwd, "GeoAI/bounding_boxes",string_arr.split('.')[0] + '.txt')
      with open(bbox_name, 'r') as file:
          bbox = file.read()
          values = bbox.split()
          yolo_bbox = [float(value) for value in values]

      # Calculate cropping coordinates
      image_width, image_height = img.size
      x_min = int((yolo_bbox[1] - yolo_bbox[3] / 2) * image_width)
      y_min = int((yolo_bbox[2] - yolo_bbox[4] / 2) * image_height)
      x_max = int((yolo_bbox[1] + yolo_bbox[3] / 2) * image_width)
      y_max = int((yolo_bbox[2] + yolo_bbox[4] / 2) * image_height)

      # Crop the image to the specified region
      cropped_image = img.crop((x_min, y_min, x_max, y_max))

      return cropped_image

"""## Create dataset"""

class SudanDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_metadata = annotations_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_metadata.iloc[idx]['filename'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image

dataset = SudanDataset(school_data, images_path, transform=transforms.Compose([
    BboxCrop(),
    transforms.Resize([256, 256])
]))

"""## Create model class"""
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm

class MyEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DYNOv2
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)

        
    def collate_fn(self, xs):
        return self.processor(images=xs, return_tensors="pt").to(self.device)

    def __call__(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, collate_fn=self.collate_fn)

        feats = []
        for ims in tqdm(dataloader, desc="Computing embeddings"):
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = self.model(**ims)
                image_feats = output.last_hidden_state
                image_feats = image_feats.mean(dim=1)
                feats.append(image_feats.to(self.device))

        return feats

embedder = MyEmbedder()

feats = embedder(dataset)

# Apply cpu() and detach() to each tensor in the list
feats_cpu = [feat.cpu().detach().numpy() for feat in feats]

# Now you can concatenate the CPU tensors
feats_concatenated = np.concatenate(feats_cpu)

# Make sure to assert the shape of the concatenated tensor, not the original tensor
assert feats_concatenated.shape[0] == len(school_data)

# Save the concatenated numpy array
np.save(output_path, feats_concatenated)