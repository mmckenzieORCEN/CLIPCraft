from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import json
import gzip
import pandas as pd
import torch
import numpy

def _extractURLFromFile(input_urls):
    if isinstance(input_urls, str):  # Check if user input is a string
        input_urls = [input_urls]  # Convert it to a list for processing

    urls = []
    for file_path in input_urls:
    # Detect file type based on extension
        file_extension = file_path.rsplit('.', 1)[-1].lower()

        if file_extension == 'json':
            with open(input_urls, 'r') as datafile:
                data = json.load(datafile)
                _extractURLFromJSON(data, urls)
    else:
        with open(input_urls, 'r') as datafile:
            for line in datafile:
                url = line.strip()
                if url:  # Skip empty lines
                    urls.append(url)
    return urls

def _extractURLFromJSON(data, urls):
    if isinstance(data, list):
        for item in data:
            _extractURLFromJSON(item, urls)
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str) and value.startswith('http'):
                urls.append(value)
            elif isinstance(value, (dict, list)):
                _extractURLFromJSON(value, urls)

def _imageEmbeddings(url_list):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  #Loading pre-trained CLIP model/processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url_embeddings_list = []
    for url in url_list
        try:
            response = requests.get(url, timeout = 5)  #For URLs that are unretrievable. 5 secs for slow retrieval
            content_type = response.headers.get('content-type')
            image = Image.open(BytesIO(response.content))
            
            inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True) #Inputs for the model; Can't change text inputs or get an exception
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            embeddings = outputs.image_embeds[0].detach().numpy().tolist()
            url_embeddings_list.append((url, embeddings))

        except Exception as e:
            print(f"Skipping URL: {url}")
            print(e)
    return url_embeddings_list

#User-callable function to generate text embeddings from CLIP
def createTextEmbeddings(input_text):
    text_embedding_list = []
    if isinstance(input_text, str):  # Check if user input is a string instead of list
        input_text = [input_text]  # Convert it to a list with a single element for processing
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for text in input_text:
        inputs = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
        text_embedding = model.get_text_features(**inputs)
        embedding_as_np = text_embedding.cpu().detach().numpy()
        text_embedding_list.append(text, embedding_as_np)
    return text_embedding_list

#User-callable function to display KNN, or most similar images to given text
def KNNSearch(text_embeddings, image_embeddings):
    image_embeddings = np.array([embedding for _, embedding in image_embeddings])
    distances = np.linalg.norm(image_embeddings - text_embeddings, axis=1)
    index_10_similar = np.argsort(distances)[:10]
    
    # Retrieve the nearest URLs based on the indices
    nearest_images = []
    for i in index_10_similar
        image_url = image_embeddings[i][0]
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        nearest_images.append((image, image_url))
    
    return nearest_images