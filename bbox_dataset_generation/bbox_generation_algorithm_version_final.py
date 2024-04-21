import warnings
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import itertools
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")

def download_image(image_path):
    return Image.open(image_path).convert("RGB")

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def crop_image(img: str):
    cwd = '/work/alex.unicef/'
    string_arr = img.split('/')[-1]
    bbox_name = os.path.join(cwd, "GeoAI/bounding_boxes",string_arr.split('.')[0] + '.txt')
    img = Image.open(img).convert('RGB')

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

    return cropped_image, [x_min, y_min, x_max, y_max]

class BuildingBbox():
  def __init__(self, is_school, bbox, id):
    self.is_school = is_school
    self.bbox = bbox # [x_min, y_min, x_max, y_max]
    self.id = id

def is_equal_to_school(school_bbox, bbox):
  x_min_s, y_min_s, x_max_s, y_max_s = school_bbox
  x_min, y_min, x_max, y_max = bbox

  if (x_min_s < x_min and x_max_s > x_max) and (y_max_s > y_max and y_min_s < y_min):
    return True
  return False

def generate_subsets(n):
  subsets = [[]]
  for i in range(0, n):

    new_subsets = []

    for subset in subsets:
      new_subsets.append(subset+[i])

    subsets += new_subsets

  return subsets

# let's exclude empty subset and [0, 1] which is true school bbox

def merge_bboxes(bbox_ids, dictionary):
  # take the first building
  x_min, y_min, x_max, y_max = dictionary.get(bbox_ids[0]).bbox

  for i, bbox_id in enumerate(bbox_ids):
    if i == 0:
      continue
    building_bbox = dictionary.get(bbox_ids[i])
    x_min_c, y_min_c, x_max_c, y_max_c = building_bbox.bbox
    x_min, y_min, x_max, y_max = [min(x_min, x_min_c), min(y_min, y_min_c),
                                  max(x_max, x_max_c), max(y_max, y_max_c)]

  return [x_min, y_min, x_max, y_max]

def remove_duplicated_bboxes(bboxes):
    set_of_tuples = {tuple(np.array(inner_list[0])) for inner_list in bboxes}
    unique_bboxes = [list(inner_tuple) for inner_tuple in set_of_tuples]
    return unique_bboxes

def calculate_bounding_box_area(box):
    """
    Calculate the area of a bounding box.

    Parameters:
    - min_x, min_y: Coordinates of the bottom-left corner
    - max_x, max_y: Coordinates of the top-right corner

    Returns:
    - Area of the bounding box
    """
    x_min, y_min, x_max, y_max = np.array(box)
    area = (x_max - x_min) * (y_max - y_min)
    return area

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1, box2: Tuple or list representing (x_min, y_min, x_max, y_max) of the bounding box

    Returns:
    - IoU score
    """
    x1, y1, w1, h1 = box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1]

    # Calculate coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate area of intersection and union
    area_intersection = w_intersection * h_intersection
    area_union = w1 * h1 + w2 * h2 - area_intersection

    # Calculate IoU
    iou = area_intersection / area_union if area_union > 0 else 0.0

    return iou

def plot_bboxes(image, school_bbox, unique_false_bboxes):
    # DEBUG - plot image
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    # DEBUG - plot school_bbox
    x_min_s, y_min_s, x_max_s, y_max_s = school_bbox
    box_width = x_max_s - x_min_s
    box_height = y_max_s - y_min_s
    rect = plt.Rectangle((x_min_s, y_min_s), box_width, box_height, fill=False, edgecolor='green', linewidth=2.5)
    ax.add_patch(rect)

    # plot
    for bbox in unique_false_bboxes:
      # DEBUG - plot bbox
      x_min, y_min, x_max, y_max = bbox
      box_width = x_max - x_min
      box_height = y_max - y_min
      rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='blue', linewidth=0.5)
      ax.add_patch(rect)

    # DEBUG
    plt.show()

"""detailed method"""

def plot_bounding_box(ax, box, school_bbox):
    x_min_s, y_min_s, x_max_s, y_max_s = school_bbox

    # DEBUG - plot school_bbox
    box_width = x_max_s - x_min_s
    box_height = y_max_s - y_min_s
    rect = plt.Rectangle((x_min_s, y_min_s), box_width, box_height, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect)

    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min
    # Draw bounding box
    rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

def plot_generated_boxes(image_pil, unique_false_bboxes, school_bbox):
  # Calculate the number of rows and columns in the grid
  num_boxes = len(unique_false_bboxes)
  num_cols = 2  # Number of columns per row (you can adjust this)
  num_rows = (num_boxes + num_cols - 1) // num_cols

  # Create subplots for each bounding box
  fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

  # Flatten the axs array to iterate over it
  axs_flat = axs.flatten()

  # Plot each bounding box in a separate subplot
  for i, (box, ax) in enumerate(zip(unique_false_bboxes, axs_flat)):
      ax.imshow(image_pil)
      plot_bounding_box(ax, box, school_bbox)
      ax.set_title(f'{box}')
      ax.axis('off')

  # Hide empty subplots
  for j in range(num_boxes, len(axs_flat)):
      axs_flat[j].axis('off')

  plt.tight_layout()
  plt.show()

def generate_bboxes_for_image(image_path):
  image_pil = download_image(image_path)

  # detect all buildings
  text_prompt = "building"
  masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)

  # get school bbox
  cropped_image, school_bbox = crop_image(image_path)

  # remove boxes that are like the while initial image -> eps = 0.3 NEW
  # remove boxes that have IOU > 90% with the school bounding box -> 0.9
  w, h, c = np.array(image_pil).shape
  eps = 3
  image_square = w * h

  iou_eps = 0.79

  valid_boxes = []
  for box in boxes:

    if np.abs(calculate_bounding_box_area(box) - image_square) < eps:
      continue

    #print(calculate_iou(school_bbox, box))
    if calculate_iou(school_bbox, box) > iou_eps:
      #print('drop')
      continue

    valid_boxes.append(box)

  boxes = valid_boxes

  n = len(boxes)

  # detect buildings that are parts of the school box
  buildingBboxes = []
  school_subset = []
  for i, bbox in enumerate(boxes):
    if is_equal_to_school(school_bbox, bbox):
      buildingBboxes.append(BuildingBbox(True, bbox, i))
      school_subset.append(i)
    else:
      buildingBboxes.append(BuildingBbox(False, bbox, i))

  i = len(boxes)
  # when no buildings are detected inside school bbox NEW
  if school_subset == []:
    school_subset.append(i)
    buildingBboxes.append(BuildingBbox(True, school_bbox, i))
    n += 1

  # create dict structure for faster buildings' data retrieving
  keys = range(0,n)
  values = buildingBboxes
  dictionary = dict(zip(keys, values))

  # generate all possible combinations of buildings
  subsets = generate_subsets(n)
  false_bboxes = set()
  true_bbox = school_bbox

  # create corresponding bboxes
  for subset in subsets:
    if subset == school_subset or subset == []:
      continue
    false_bboxes.add(tuple(merge_bboxes(subset, dictionary)))

  false_bboxes = [list(inner_tuple) for inner_tuple in false_bboxes]

  # remove dupllicated bboxes
  #unique_false_bboxes = remove_duplicated_bboxes(false_bboxes)

  #print(len(unique_false_bboxes))
  #print(len(false_bboxes))

  # Merge bounding boxes
  #plot_bboxes(image_pil, school_bbox, false_bboxes)

  # Resulting merged bounding boxes
  #display_image_with_boxes(image_pil, boxes, logits)

  #return unique_false_bboxes, school_bbox
  return false_bboxes, school_bbox


model = LangSAM()

image = "/content/drive/MyDrive/GeoAI/satellite_imagery/school/42901352.png"
false_bboxes, school_bbox = generate_bboxes_for_image(image)

print(false_bboxes)

#let's save dataset for all images
#format: [id, image_path, is_correct_bbox, bbox]



image_paths = []
school_images_path = '/content/drive/MyDrive/GeoAI/satellite_imagery/school/'

for filename in os.listdir(school_images_path):
    image_paths.append(filename)

print(len(image_paths))

# class - school or not_school
dataset = pd.DataFrame(data={'filename':[], 'width':[], 'height': [], 'class': [],
                             'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []})

first_chunk = image_paths[1000:1100]

# 600-700  --> 7
# trouble files -  /content/drive/MyDrive/GeoAI/satellite_imagery/school/61223303.png

trouble_list = []

for filename in tqdm(first_chunk, desc="Processing Images"):
  try:
    image_path = school_images_path + filename
    print(image_path)
    false_bboxes, school_bbox = generate_bboxes_for_image(image_path)

    for bbox in false_bboxes:
      x_min, y_min, x_max, y_max = np.asarray(bbox)

      tmp_df = pd.DataFrame(data={'filename':filename, 'width': [x_max-x_min], 'height': [y_max-y_min], 'class': [0],
                                  'xmin': [x_min], 'ymin': [y_min], 'xmax': [x_max], 'ymax': [y_max]})
      dataset = pd.concat([dataset, tmp_df])

    x_min, y_min, x_max, y_max = school_bbox
    tmp_df = pd.DataFrame(data={'filename':filename, 'width': [x_max-x_min], 'height': [y_max-y_min], 'class': [1],
                                'xmin': [x_min], 'ymin': [y_min], 'xmax': [x_max], 'ymax': [y_max]})
    dataset = pd.concat([dataset, tmp_df])
  except:
    print('TROUBLE')
    trouble_list.append(image_path)

print(trouble_list)

dataset.to_csv('/work/alex.unicef/feature_extractor/bbox_dataset_generation/'+'11.csv')

for filename in tqdm(first_chunk, desc="Processing Images"):
    image_path = school_images_path + filename
    print(image_path)
    false_bboxes, school_bbox = generate_bboxes_for_image(image_path)

    for bbox in false_bboxes:
      x_min, y_min, x_max, y_max = np.asarray(bbox)

      tmp_df = pd.DataFrame(data={'filename':filename, 'width': [x_max-x_min], 'height': [y_max-y_min], 'class': [0],
                                  'xmin': [x_min], 'ymin': [y_min], 'xmax': [x_max], 'ymax': [y_max]})
      dataset = pd.concat([dataset, tmp_df])

    x_min, y_min, x_max, y_max = school_bbox
    tmp_df = pd.DataFrame(data={'filename':filename, 'width': [x_max-x_min], 'height': [y_max-y_min], 'class': [1],
                                'xmin': [x_min], 'ymin': [y_min], 'xmax': [x_max], 'ymax': [y_max]})
    dataset = pd.concat([dataset, tmp_df])

dataset.head(3)

dataset = dataset.reset_index()

dataset.to_csv('/work/alex.unicef/feature_extractor/bbox_dataset_generation/first_chunk_adjusted_algorithm.csv', index=False)

# reasons why the algorithm's output is not ideal:

# 1. buiding are not detected al all on langSAM stage -> can be eliminated using real maps
# 2. sometimes generated boxes by langSAM are weird -> can implement check&reduce procedure
#    (epsilon with school intercection / IoU with > 90% intersection

#     and reduce zero square bboxes)

# 3. the dataset is imbalanced -> can augment

# 4. check metrics on non school dataset as well after training !!!

# CLIP + SVM check
