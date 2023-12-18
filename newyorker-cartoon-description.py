import vertexai
import json
import io

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

captions_examples_ds = load_dataset(
    "jmhessel/newyorker_caption_contest", "explanation",
    split="validation"
)

captions_gold_examples_ds = load_dataset(
    "jmhessel/newyorker_caption_contest", "explanation",
    split="test"
)

description_prompt = """
Your task is to generate a description for the cartoon presented in the input.
Write a 2-3 sentence description focusing on:
- Where is the scene taking place?
- Who/Whats in the scene? What are they doing?
- What objects and actions are being depicted?
- Is anyone particularly happy/unhappy/mad/etc?
There is no need to be formal, but please do your best to write full, grammatical sentences. 
Here are a few examples to guide your generation process.
"""

task_prompt = """Now generate a description for the following cartoon:"""

examples_for_prompt = []

n_examples = 5 # 5-shot as in the paper
few_shot_examples = captions_examples_ds.shuffle()[0: n_examples]

for example_image, example_description in zip(
    few_shot_examples['image'], 
    few_shot_examples['image_description']):
    
    with io.BytesIO() as buffer:
        example_image.save(buffer, format='JPEG')
        example_image_bytes = buffer.getvalue()
        example_image_input = Image.from_bytes(example_image_bytes)

        examples_for_prompt.append(example_image_input)
        examples_for_prompt.append(example_description)

few_shot_prompt = [description_prompt] + examples_for_prompt + [task_prompt]

# Generation
description_generation_config = GenerationConfig(
    temperature=0.8,
    top_p=.95,
    max_output_tokens=64
)

# Evaluation

n_test_examples = 30

gold_examples = captions_gold_examples_ds.shuffle()[0: n_test_examples]

model_predictions, ground_truths = [], []

for gold_example_image, gold_example_description in zip(
    gold_examples['image'], 
    gold_examples['image_description']):
    
    with io.BytesIO() as buffer:
        gold_example_image.save(buffer, format='JPEG')
        gold_example_image_bytes = buffer.getvalue()
        gold_example_image_input = Image.from_bytes(gold_example_image_bytes)

        gold_example_prompt = few_shot_prompt + [gold_example_image_input]

        try:
            generated_description = multimodal_model.generate_content(
                gold_example_prompt,
                generation_config=description_generation_config
            )
        except Exception as e:
            print(e)
            continue

        model_predictions.append(generated_description.text.strip())
        ground_truths.append(gold_example_description)

with open('generated-descriptions.txt', 'w') as f:
    for prediction in model_predictions:
        f.write(f'{prediction}\n')

with open('gold-descriptions.txt', 'w') as f:
    for description in ground_truths:
        f.write(f'{description}\n')