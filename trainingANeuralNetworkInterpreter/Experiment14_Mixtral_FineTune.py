#!/usr/bin/env python
# coding: utf-8

# Load all the libraries we need. It's important to set CUDA_VISIBLE_DEVICES before importing PyTorch in the script because PyTorch initializes its CUDA subsystem as soon as it's imported. If one changes CUDA_VISIBLE_DEVICES after PyTorch has been imported, it won't have any effect because PyTorch has already initialized its CUDA environment and detected available GPUs. 

# In[1]:

# %%

#General purpose
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
os.environ['HF_TOKEN'] = "yourToken"
os.environ['TRANSFORMERS_CACHE'] = 'yourCache'

from datetime import datetime
import json
import random
import math
from PIL import Image
import pickle # For loading the human-annotated images

# General data libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#machine learning libs
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#some special machine learnings libs for our CNN and for data efficiency
from torchvision import datasets, transforms
from datasets import load_dataset

#Large language models and Fine tuning 
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# Libs for facilitating distributed training for our large language model

from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator

# We start a second thread to monitor GPU usage during training
import threading
import subprocess
import time

# %%

print(transformers.__version__)


print("CUDA Version from nvidia-smi: \n")  # Replace with the actual output
os.system("nvidia-smi | grep 'CUDA Version'")
print(f"Number of GPUs and their usage \n ")
os.system("nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv")

print("CUDA_VISIBLE_DEVICES set to:", os.environ.get("CUDA_VISIBLE_DEVICES"))


print("PyTorch version:", torch.__version__)
print("CUDA version PyTorch is built with:", torch.version.cuda)

# %%

def generate_vertical_line_image(image, line_min_height=6, line_max_height=12):
    height, width = image.shape

    num_lines = 1
    for _ in range(num_lines):
        line_height = random.randint(line_min_height, line_max_height)
        line_start_x = random.randint(0, width - 1)
        line_start_y = random.randint(0, height - line_height)

        image[line_start_y:line_start_y + line_height, line_start_x] = 255

    return image

def generate_horizontal_line_image(image, line_min_length=6, line_max_length=12):
    height, width = image.shape

    num_lines = 1
    for _ in range(num_lines):
        line_length = random.randint(line_min_length, line_max_length)
        line_start_x = random.randint(0, width - line_length)
        line_start_y = random.randint(0, height - 1)

        image[line_start_y, line_start_x:line_start_x + line_length] = 255

    return image

def generate_contra_diagonal_line(image, line_min_length=4, line_max_length=12):
    height, width = image.shape

    # Determine line length
    line_length = random.randint(line_min_length, line_max_length)

    # Random start position for contra-diagonal line
    line_start_x = random.randint(0, width - line_length)
    line_start_y = random.randint(0, height - line_length)

    # Draw the contra-diagonal line
    for i in range(line_length):
        image[line_start_y + i, line_start_x + i] = 255

    return image

def generate_diagonal_line(image, line_min_length=4, line_max_length=12):
    height, width = image.shape

    # Determine line length
    line_length = random.randint(line_min_length, line_max_length)

    # Random start position for diagonal line
    line_start_x = random.randint(0, width - line_length)
    line_start_y = random.randint(line_length - 1, height - 1)

    # Draw the diagonal line
    for i in range(line_length):
        image[line_start_y - i, line_start_x + i] = 255

    return image



def add_crossing_to_image(image):
    def rotate_point(point, angle, origin=(0, 0)):
        """ Rotate a point around a given origin. """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def translate_point(point, dx, dy):
        """ Translate a point by dx and dy. """
        x, y = point
        return x + dx, y + dy

    def draw_line(image, start_point, end_point):
        """ Draw a straight line on the image, ensuring it stays within bounds. """
        x0, y0 = int(start_point[0]), int(start_point[1])
        x1, y1 = int(end_point[0]), int(end_point[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0]:
                image[y0, x0] = 255  # Set pixel to white
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    height, width = image.shape

    # Define the middle point and first line
    mid_point = (width // 2, height // 2)
    first_line_start = (mid_point[0], mid_point[1] - 3)
    first_line_end = (mid_point[0], mid_point[1] + 3)

    # Random rotation and translation for the first line
    rotation_angle = random.uniform(0, 2 * math.pi)  # Random rotation angle
    translation_dx = random.randint(-9, 9)  # Random translation in x
    translation_dy = random.randint(-9, 9)  # Random translation in y

    # Rotate and translate the first line
    rotated_start = rotate_point(first_line_start, rotation_angle, mid_point)
    rotated_end = rotate_point(first_line_end, rotation_angle, mid_point)
    translated_start = translate_point(rotated_start, translation_dx, translation_dy)
    translated_end = translate_point(rotated_end, translation_dx, translation_dy)

    # Draw the first line
    draw_line(image, translated_start, translated_end)

    # Define the second line
    second_line_mid_point = ((translated_start[0] + translated_end[0]) / 2, (translated_start[1] + translated_end[1]) / 2)
    second_line_length = 3  # Length of the second line

    # Calculate the second line's rotation angle with improved range
    second_line_rotation_angle = (rotation_angle + random.uniform(math.pi / 4 - math.pi / 7, math.pi / 4 + math.pi / 7)) % (2 * math.pi)

    # Calculate the second line's end points
    second_line_end_1 = (
        second_line_mid_point[0] + second_line_length * math.cos(second_line_rotation_angle),
        second_line_mid_point[1] + second_line_length * math.sin(second_line_rotation_angle)
    )
    second_line_end_2 = (
        second_line_mid_point[0] - second_line_length * math.cos(second_line_rotation_angle),
        second_line_mid_point[1] - second_line_length * math.sin(second_line_rotation_angle)
    )

    # Draw the second line
    draw_line(image, second_line_mid_point, second_line_end_1)
    draw_line(image, second_line_mid_point, second_line_end_2)

    return image


def add_checkerboard_pattern(image, grid_size=2, pattern_size=4):
    def draw_partial_checkerboard(image, grid_size, start, end):
        """ Draw a checkerboard pattern on a part of the image. """
        for y in range(start[1], end[1], grid_size):
            for x in range(start[0], end[0], grid_size):
                if (x // grid_size + y // grid_size) % 2 == 0:
                    image[y:y+grid_size, x:x+grid_size] = 255  # Set pixel to white

    def draw_random_checkerboard_5x5(image, grid_size, pattern_size):
        """ Draw a 5x5 checkerboard pattern at a random position in the image, considering the edges. """
        # Adjust the maximum size for a 5x5 checkerboard
        total_pattern_size = grid_size * pattern_size  # 5 squares of size 'grid_size' each

        # Randomly select the upper left corner within bounds
        max_x = image.shape[1] - total_pattern_size
        max_y = image.shape[0] - total_pattern_size
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)

        # Define the region for the checkerboard pattern
        start_point = (start_x, start_y)
        end_point = (start_x + total_pattern_size, start_y + total_pattern_size)

        # Draw the checkerboard pattern
        draw_partial_checkerboard(image, grid_size, start_point, end_point)

    # Draw a 5x5 checkerboard pattern on the image
    draw_random_checkerboard_5x5(image, grid_size, pattern_size)
    return image


def add_corner_to_image(image):
    def draw_perpendicular_corner(image, point, corner_type):
        """ Draw a corner consisting of two 4-pixel lines that are perpendicular to each other. """
        x, y = point

        # Size of each line in the corner
        line_size = 3

        # Draw the corner based on the specified type
        if corner_type == 1:  # Upper Left
            image[y:y+line_size, x] = 255  # Vertical line
            image[y, x:x+line_size] = 255  # Horizontal line
        elif corner_type == 2:  # Upper Right
            image[y:y+line_size, x] = 255  # Vertical line
            image[y, x-line_size:x] = 255  # Horizontal line
        elif corner_type == 3:  # Bottom Left
            image[y-line_size:y +1 , x] = 255  # Vertical line
            image[y, x:x+line_size] = 255  # Horizontal line
        elif corner_type == 4:  # Bottom Right
            image[y-line_size:y +1, x] = 255  # Vertical line
            image[y, x-line_size:x] = 255  # Horizontal line

    # Randomly sample a point within a specific area to ensure space for the corner
    random_point = (random.randint(5, 22), random.randint(5, 22))

    # Randomly select a corner type
    corner_type = random.randint(1, 4)

    # Draw the perpendicular corner
    draw_perpendicular_corner(image, random_point, corner_type)

    return image



def draw_smooth_ellipse(image, center, axes, start_angle, end_angle, step=1):
    for angle in range(start_angle, end_angle, step):
        radian = math.radians(angle)
        x = center[0] + axes[0] * math.cos(radian)
        y = center[1] + axes[1] * math.sin(radian)

        # Apply anti-aliasing-like technique by softening the pixels near the ellipse boundary
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                ix, iy = int(x + dx), int(y + dy)
                if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]:
                    distance = math.sqrt((x - ix) ** 2 + (y - iy) ** 2)
                    if distance < 1.5:  # Soften pixels close to the calculated point
                        image[iy, ix] = min(255, image[iy, ix] + int(255 * (1.5 - distance)))

def generate_partial_ellipse_image(image, num_ellipses=1):
    height, width = image.shape

    for _ in range(num_ellipses):
        center = (width // 2, height // 2)
        axes = (random.randint(8, 12), random.randint(8, 12))  # Adjust axes to avoid extremes
        angle_range = random.randint(50, 80)
        start_angle = random.randint(0, 360 - angle_range)
        end_angle = start_angle + angle_range

        draw_smooth_ellipse(image, center, axes, start_angle, end_angle)

    return image

def image_generator(num_corner = 0, num_vertical=0, num_horizontal=0, num_ellipse=0, num_diagonal=0, num_contra_diagonal= 0, num_crossing=0, num_checkerboard=0):
    # Initialize a 28x28 black image
    image = np.zeros((28, 28), dtype=np.uint8)

    # Add the specified number of vertical lines
    for _ in range(num_vertical):
        image = generate_vertical_line_image(image)

    # Add the specified number of vertical lines
    for _ in range(num_diagonal):
        image = generate_diagonal_line(image)

    # Add the specified number of vertical lines
    for _ in range(num_contra_diagonal):
        image = generate_contra_diagonal_line(image)


    # Add the specified number of horizontal lines
    for _ in range(num_horizontal):
        image = generate_horizontal_line_image(image)

    for _ in range(num_crossing):
        image = add_crossing_to_image(image)

    for _ in range(num_checkerboard):
        image = add_checkerboard_pattern(image)


    # Add the specified number of partial ellipses
    for _ in range(num_ellipse):
        image = generate_partial_ellipse_image(image)

    for _ in range(num_corner):
        image = add_corner_to_image(image)

    return image


def sample_shape_count():
    # Sample using exponential distribution
    return int(np.random.exponential(scale=0.2) + 1)

def generate_shape_description(shape_counts):
    # Generate a human-readable description of the shapes
    descriptions = []
    for shape, count in shape_counts.items():
        if count > 0:
            shape_name = shape.split('_')[1]

            # Special handling for 'contra' shape
            if shape_name == 'contra':
                shape_name = 'contra diagonal'

            if count == 1:
                descriptions.append(f"1 {shape_name}")
            else:
                # Add 's' for plural, except for special case
                if shape_name == 'contra diagonal ':
                    descriptions.append(f"{count} {shape_name}")
                else:
                    descriptions.append(f"{count} {shape_name}s")
    return ', '.join(descriptions)


def create_synthetic_dataset(num_images):
    # Create directory to save images and descriptions
    images = []
    descriptions = []

    for i in range(num_images):
        shape_counts = {
            "num_corner": 0,
            "num_vertical": 0,
            "num_horizontal": 0,
            "num_diagonal": 0,
            "num_contra_diagonal": 0,
            "num_crossing": 0,
            "num_checkerboard": 0,
            "num_ellipse": 0
        }
        probabilities = [0.4, 0.3, 0.2, 0.1]
        num_shapes_to_select = np.random.choice([1, 2, 3, 4], p=probabilities)
        selected_shapes = np.random.choice(list(shape_counts.keys()), size=num_shapes_to_select, replace=False)
        for shape in selected_shapes:
            shape_counts[shape] = int(round(sample_shape_count()))

        generated_image = image_generator(**shape_counts)
        shape_description = generate_shape_description(shape_counts)
        
        total_shapes = sum(shape_counts.values())

    
        shape_description = f"Image with {total_shapes} shape{'s' if total_shapes > 1 else ''}: {shape_description}"
        
        # Store the image and description
        images.append(generated_image)
        descriptions.append(shape_description)

    return images, descriptions

# %%

class SyntheticDataset(Dataset):
    def __init__(self, images, descriptions, transform=None):
        self.images = images
        self.descriptions = descriptions
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        description = self.descriptions[idx]

        if self.transform:
            # Convert numpy array to tensor and apply transform
            #transforms.ToTensor is a constructor that creates an instance of the ToTensor transform. 
            image = transforms.ToTensor()(image)
            image = self.transform(image)

        return image, description
# %%


class ManuallyAnnotatedDataset(Dataset):
    def __init__(self, images, descriptions, transform=None):
        self.images = images
        self.descriptions = descriptions
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        description = self.descriptions[idx]

        if self.transform:
            # Assuming the images are numpy arrays and need to be converted to tensors
            image = self.transform(image)

        return image, description

# %%


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Initialize convolutional layer 1 with He Initialization
        # Increased kernel size to 9 and number of filters remains 8
        self.conv1 = self._init_conv_layer(1, 8, 9, 1, 4, 'he')  # Padding adjusted to 4

        # Initialize convolutional layer 2 with He Initialization
        # Increased kernel size to 9 and number of filters remains 16
        self.conv2 = self._init_conv_layer(8, 16, 9, 1, 4, 'he')  # Padding adjusted to 4

        # Initialize fully connected output layer
        # Input dimension remains the same
        self.out = nn.Linear(16 * 7 * 7, 10)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')

    def _init_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, init_method):
        """Initializes a convolutional layer followed by a ReLU activation and MaxPool."""
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        if init_method == 'he':
            nn.init.kaiming_normal_(layer[0].weight, nonlinearity='relu')

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        output = self.out(x)
        return output, x


# %%

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


targetCNN = CNN()
weights = torch.load('mnistTwoLayersFilterSize9And16Filters.pth')
targetCNN.load_state_dict(weights)
# %%


def process_image(targetCNN, image, description):
    instance_data = {}
    # Process the image through the first convolutional layer
    with torch.no_grad():
        activations = targetCNN.conv1(image)

    # Flatten, and store raw pixel data and activations
    flattened_image = image.view(-1).numpy()
    instance_data['raw_pixel_data'] = flattened_image
    for j in range(8):  # Assuming 8 filters in the first conv layer
        flattened_activation = activations[0, j].flatten().numpy()
        instance_data[f'activationmap{j+1}'] = flattened_activation

    # Store image description
    instance_data['image_description'] = description
    return instance_data

#num_images defines the synthetic images.
def create_data_dict(targetCNN, num_images):
    # Assuming create_synthetic_dataset and SyntheticDataset are defined elsewhere
    images, descriptions = create_synthetic_dataset(num_images)
    transform = transforms.Normalize((0.5,), (0.5,))
    syntheticDataSet = SyntheticDataset(images, descriptions, transform=transform)
    synth_loader = DataLoader(syntheticDataSet, batch_size=1, shuffle=False)

    # File paths for manually annotated data (set within the function)
    manual_images_file = 'first_100_testset_images.pkl'
    manual_annotations_file = 'first_100_testset_manual_annotations.pkl'

    # Load manually annotated data
    with open(manual_images_file, 'rb') as file:
        loaded_images = pickle.load(file)
    with open(manual_annotations_file, 'rb') as file:
        loaded_annotations = pickle.load(file)

    # Assuming ManuallyAnnotatedDataset is defined elsewhere
    manuallyAnnotatedDataSet = ManuallyAnnotatedDataset(loaded_images, loaded_annotations, transform=transform)
    manual_loader = DataLoader(manuallyAnnotatedDataSet, batch_size=1, shuffle=False)

    data_dict = {}
    total_images = 0  # Keep track of total images processed

    # Process data from synthetic dataset
    for image, description in synth_loader:
        if total_images >= num_images:
            break
        instance_data = process_image(targetCNN, image, description)
        data_dict[f'image_{total_images + 1}'] = instance_data
        total_images += 1

    # Process data from manually annotated dataset
    for image, description in manual_loader:
        instance_data = process_image(targetCNN, image, description)
        data_dict[f'image_{total_images + 1}'] = instance_data
        total_images += 1

    return data_dict

# %%

def map_activation_to_char(activation):
    """Maps an activation value to a character based on predefined intervals."""
    if activation == 0:
        return 'z'
    elif 0 < activation <= 0.6:
        return 'l'
    elif 0.6 < activation <= 1.3:
        return 'm'
    elif 1.3 < activation <= 2.2:
        return 'h'
    else:  # activation > 2.2
        return 'v'

def generate_jsonl_data(data_dict):
    jsonl_data = []

    for image_key, image_data in data_dict.items():
        # Concatenate activation maps
        concatenated_activations = np.concatenate(
            [image_data[f'activationmap{j+1}'] for j in range(8)]
        )
        input_sequence = 'f'
        # Process and append each activation map
        for j in range(8):  # Assuming 8 filters in the first conv layer
            activation_map_key = f'activationmap{j+1}'
            mapped_activations = ''.join(map(map_activation_to_char, image_data[activation_map_key]))
            input_sequence += mapped_activations + 'f'  # Append 'f' after each activation map

        # Remove the last 'f' appended
        input_sequence = input_sequence[:-1]

        # Retrieve the image description and we need it to unpack from the tuple
        output_description = image_data['image_description']
        output_description = output_description[0]

        # Create a JSONL-like entry
        jsonl_data.append({
            "input": input_sequence,
            "output": output_description
        })
    

    return jsonl_data
# %%


def formatting_func(trainingInstance):
    text = f"""You will be presented with sequences of characters representing features of images, which could be basic geometric structures or elements similar to MNIST handwritten digits. These sequences are encoded representations of the image's activation maps, derived from the first layer of a convolutional neural network. The character 'f' marks the beginning of a new filter's activation map of the first layer. The characters 'z', 'l', 'm', 'h', and 'v', indicate varying levels of activation: zero, low, medium, high, and very high, respectively. Your task is to understand the patterns in the sequence and to interpret these sequences to determine the number and types of shapes present in the image. The desired output format should be 'output: Image with [number of shapes] shapes: [number of first shape] [name of shape] ... [number of last shape] [name of last shape]', where the description includes the types of shapes, such as corners, edges, ellipses, crossings, checkerboard pattern, horizontals, verticals, etc. ### Input: {trainingInstance['input']} \n ### Output: {trainingInstance['output']}"""
    return text

# %%


base_model_id = "mistralai/Mixtral-8x7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


max_length = 1280 # This is the appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

# %%

# Let us tokenize our data with the newly padded scheme. 

# Let us plot the tokenized sequences to confirm that they all have the same length now. 

# 
# Check that input_ids is padded on the left with the eos_token (2) and there is an eos_token 2 added to the end, and the prompt starts with a bos_token (1).

# Great, we did a lot of debugging but actually now we are shure of our data set and we can make a really big data set and train our neural network interpreter on it. We comment the generating code out because we already generated the data previously.

# In[16]:


# Create a production ready data set of 15,000 images:
#realdataDict = create_data_dict(targetCNN, num_images=15000)
#realjsonl_data = generate_jsonl_data(realdataDict)

# Step 1: Write JSONL data to a file
jsonl_file_pathReal = 'realtrainingSet.jsonl'
#with open(jsonl_file_pathReal, 'w') as f:
 #   for entry in realjsonl_data:
 #       json.dump(entry, f)
  #      f.write('\n')
# %%

real_trainingdataset = load_dataset('json', data_files=jsonl_file_pathReal, split='train')

# Manually split the dataset to ensure that the humanly annotated data remains in the dataset
validation_size = int(0.1 * len(real_trainingdataset))  # 10% for validation
train_indices = list(range(validation_size, len(real_trainingdataset)))
val_indices = list(range(validation_size))

# Use Dataset.select() to split the dataset
realtrain_dataset = real_trainingdataset.select(train_indices)
realeval_dataset = real_trainingdataset.select(val_indices)

print(f"The training set contains {len(realtrain_dataset)} instances.")
print(f"The evaluation set contains {len(realeval_dataset)} instances.")


realtokenized_train_dataset = realtrain_dataset.map(generate_and_tokenize_prompt2)
realtokenized_val_dataset = realeval_dataset.map(generate_and_tokenize_prompt2)

train_dataset_size = sys.getsizeof(realtokenized_train_dataset)
val_dataset_size = sys.getsizeof(realtokenized_val_dataset)

print(f"Training dataset memory size: {train_dataset_size} bytes")
print(f"Validation dataset memory size: {val_dataset_size} bytes")
# %%


# Define the device map
device_map = {
    'model.embed_tokens': 0,
    'model.layers.0': 0, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1,
    'model.layers.4': 1, 'model.layers.5': 2, 'model.layers.6': 2, 'model.layers.7': 2,
    'model.layers.8': 2, 'model.layers.9': 3, 'model.layers.10': 3, 'model.layers.11': 3,
    'model.layers.12': 3, 'model.layers.13': 4, 'model.layers.14': 4, 'model.layers.15': 4,
    'model.layers.16': 4, 'model.layers.17': 5, 'model.layers.18': 5, 'model.layers.19': 5,
    'model.layers.20': 5, 'model.layers.21': 6, 'model.layers.22': 6, 'model.layers.23': 6,
    'model.layers.24': 6, 'model.layers.25': 7, 'model.layers.26': 7, 'model.layers.27': 7,
    'model.layers.28': 7, 'model.layers.29': 8, 'model.layers.30': 8, 'model.layers.31': 8,
    'model.norm': 8,
    'lm_head': 8 # Moving lm_head to GPU 9
}

# %%


base_model_id = "mistralai/Mixtral-8x7B-v0.1"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
# %%

nnInterpreter = AutoModelForCausalLM.from_pretrained(
    base_model_id,    
    device_map= device_map,
    #device_map= 'auto',
    trust_remote_code= True,
    use_auth_token=True,
    quantization_config= bnb_config,
)



# Check available GPUs and their names
print(nnInterpreter)

gpu_count = torch.cuda.device_count()
print(f"Available GPUs: {gpu_count}")
for i in range(gpu_count):
    print(f"CUDA:{i} - {torch.cuda.get_device_name(i)}")

nnInterpreter.hf_device_map


nnInterpreter.gradient_checkpointing_enable()
nnInterpreter = prepare_model_for_kbit_training(nnInterpreter)

# %%

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  
    task_type="CAUSAL_LM",
)

nnInterpreter = get_peft_model(nnInterpreter, config)
print_trainable_parameters(nnInterpreter)

# %%


print(nnInterpreter)

# %%

project = "FineTune6"
base_model_name = "mixtral8x7B"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
bs = 2 #batch size
ga_steps = 1 # gradient accumulation steps
epochs = 4
steps_per_epoch=len(realtokenized_train_dataset)//(bs*ga_steps)

trainer = transformers.Trainer(
    model=nnInterpreter,
    train_dataset= realtokenized_train_dataset,
    eval_dataset= realtokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size= bs,
        gradient_accumulation_steps=ga_steps,
       # gradient_checkpointing=False,
        gradient_checkpointing=True,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        lr_scheduler_type="constant",
        fp16=True,
        num_train_epochs=epochs,
    #    ddp_find_unused_parameters=False, # instead we use accelerate
        optim="paged_adamw_8bit",
        logging_strategy="steps",
        logging_steps = 200,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy = "steps",
        evaluation_strategy = "steps",
        save_steps= 400,   # Save  every 100  
        eval_steps= 200,    # Evaluate every epoch
        do_eval=True,                # Perform evaluation at the end of training
  #     report_to="wandb",           
     #   run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# %%

print(subprocess.check_output("nvidia-smi", shell=True).decode())
# %%

nnInterpreter.config.use_cache = False  # silence the warnings.Re-enable for inference!
print("training starts now:")
trainer.train()
