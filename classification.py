import vertexai
import json

import numpy as np

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image
)

from datasets import load_dataset
from tqdm import tqdm

# Authentication

with open("config-vertexai.json") as f:
    data = f.read()

creds = json.loads(data)

vertexai.init(
    project=creds["project"],
    location=creds["location"]
)

multimodal_model = GenerativeModel("gemini-pro-vision")

# Data

painting_style_ds = load_dataset(
    "keremberke/painting-style-classification",
    name="full"
)

sample_size = 50

test_data = painting_style_ds['test'].shuffle()[0:sample_size]
test_images = test_data['image_file_path']
test_labels = test_data['labels']

# Evaluation

system_instructions = """
Instructions: Consider the following image that contains movement art images that range from \
Abstract Expressionism to Pop Art.

Each image corresponds to one of the following classes:
Abstract_Expressionism
Action_painting
Analytical_Cubism
Art_Nouveau_Modern
Baroque
Color_Field_Painting
Contemporary_Realism
Cubism
Early_Renaissance
Expressionism
Fauvism
High_Renaissance
Impressionism
Mannerism_Late_Renaissance
Minimalism
Naive_Art_Primitivism
New_Realism
Northern_Renaissance
Pointillism
Pop_Art
Post_Impressionism
Realism
Rococo
Romanticism
Symbolism
Synthetic_Cubism
Ukiyo_e
"""

task_prompt = """
Identify the class of the art depicted in the image as one of the above classes.
The class label generated should strictly belong to one of the classes above.
Your answer should only contain the class depicted. Do not explain your answer.
"""

art_classification_generation_config = GenerationConfig(
    temperature=0,
    top_p=1.0,
    max_output_tokens=16
)

dataset_labels = [
    'Realism', 'Art_Nouveau_Modern', 'Analytical_Cubism',
    'Cubism', 'Expressionism', 'Action_painting', 'Synthetic_Cubism',
    'Symbolism', 'Ukiyo_e', 'Naive_Art_Primitivism', 'Post_Impressionism',
    'Impressionism', 'Fauvism', 'Rococo', 'Minimalism',
    'Mannerism_Late_Renaissance', 'Color_Field_Painting',
    'High_Renaissance', 'Romanticism', 'Pop_Art', 'Contemporary_Realism',
    'Baroque', 'New_Realism', 'Pointillism', 'Northern_Renaissance',
    'Early_Renaissance', 'Abstract_Expressionism'
]
model_predictions, ground_truths = [], []

for test_image, test_label in tqdm(zip(test_images, test_labels)):

    test_image_input = Image.load_from_file(test_image)

    prompt = [
        system_instructions,
        test_image_input,
        task_prompt
    ]

    try:
        response = multimodal_model.generate_content(
            prompt,
            generation_config=art_classification_generation_config
        )

        model_predictions.append(response.text.strip())
        ground_truths.append(dataset_labels[test_label])
    except Exception as e:
        print(e)
        continue

accuracy = (np.array(model_predictions) == np.array(ground_truths)).mean()
print(accuracy)