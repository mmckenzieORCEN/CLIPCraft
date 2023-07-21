from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
import requests
import json
import gzip
import os
from io import BytesIO

# Helper function to extract the URLs from a json file
def _extractURLFromJSON(data, urls):
    # Determining if the provided data is in list format
    if isinstance(data, list):
        # Looping through the items in the list
        for item in data:
            _extractURLFromJSON(item, urls)
    # If the input data is in a dictionary format
    elif isinstance(data, dict):
        # In the dict, we continue if we find a key called image_url
        if "image_url" in data:
            # Value will become the value from the key-value pair in the dict
            value = data["image_url"]
            # If the value is a string and starts with http, meaning it is a url, we append it to the list
            if isinstance(value, str) and value.startswith('https'):
                urls.append(value)
        # Recursive call to loop through the dictionary
        for value in data.values():
            if isinstance(value, (dict, list)):
                _extractURLFromJSON(value, urls)
                
# Internal function to generate our image embeddings
def _imageEmbeddings(url_list):
    # Loading pre-trained CLIP model/processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url_embeddings_list = []
    # Looping through each value in the input list
    for url in url_list:
        try:
            # Grabbing our image. Timeout for URLs that are unretrievable. 5 secs for slow retrieval.
            response = requests.get(url, timeout = 5)  
            content_type = response.headers.get('content-type')
            image = Image.open(BytesIO(response.content))
            #Inputs for the model; Can't change text inputs or get an exception
            inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True) 

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            # Getting the image embeds from the model and adding them to our list with the origin URL
            embeddings = outputs.image_embeds[0].detach().numpy().tolist()
            url_embeddings_list.append((url, embeddings))
        except Exception as e:
            print(f"Skipping URL: {url}")
            print(e)
    return url_embeddings_list

# Internal function to write the embeddings to a file when the user chooses to do so
def _writeEmbeddingsToFile(type, input_image_embeddings = None, input_text_embeddings = None):
    # If our desired embedding type is of images
    j = 0
    if type == "image":
        with gzip.open("CLIP_image_embeddings.json.gz", "wt") as datafile:
            for i in range(len(input_image_embeddings)):
                new_row = {
                    'image_url': input_image_embeddings[i][0], #Create json pairs from the list
                    'image_embedding': input_image_embeddings[i][1],
                }
                datafile.write(json.dumps(new_row) + '\n')
                j += 1
        print("Successfully dumped #" + str(i) + " embeddings to file." + '\n')
                
    # If our desired embedding type is of text
    elif type == "text":
        with gzip.open("CLIP_text_embeddings.json.gz", "wt") as datafile:
            for i in range(len(input_text_embeddings)):
                new_row = {
                    'original_text': input_text_embeddings[i][0], 
                    'text_embedding': input_text_embeddings[i][1],
                }
                datafile.write(json.dumps(new_row) + '\n')
                j += 1
        print("Successfully dumped #" + str(i) + " embeddings to file." + '\n')

# Internal function that extracts URL from user-input file
def _extractURLFromFile(input_urls):
    if isinstance(input_urls, str):  # Check if user input is a string
        input_urls = [input_urls]  # Convert it to a list for processing

    url_count = 0
    urls = []
    for file_path in input_urls: 
        # Detect file type based on extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        data = []  # Initialize an empty list for each file
        # If the file extension is json or gz, we can load the data and call our helper function
        if file_extension in ['.json', '.gz']:
            with open(file_path, 'r') as datafile:
                if file_extension == '.gz':
                    datafile = gzip.open(file_path, 'rt')
                for line in datafile:
                    data.append(json.loads(line))
        else:
            # If file is not json or gzip, it must be a regular text file
            with open(file_path, 'r') as datafile:
                for line in datafile:
                    url = line.strip()
                    if url:  # Skips empty lines
                        urls.append(url)
    _extractURLFromJSON(data, urls)
    return urls


# User-callable for users to call to create Image embeddings
def createImageEmbeddings(input_urls, output_type): 
    if output_type not in ["list", "file"]:
        raise ValueError("Invalid output type. Expected 'list' or 'file'")
            
    embeddings = []
    if isinstance(input_urls, str):
    # Read the file and process its contents
        if input_urls.startswith("http"):
            embeddings.extend(_imageEmbeddings([input_urls]))
        else:
            url_list = _extractURLFromFile(input_urls)
            embeddings.extend(_imageEmbeddings(url_list))
                             
    if isinstance(input_urls, list):
        embeddings = []
        # Multiple files provided
        for url in input_urls:
            if isinstance(url, str) and url.startswith('http'):
                # Process a URL
                embeddings.extend(_imageEmbeddings([url]))
            elif isinstance(url, str):
                # Process a file
                url_list = _extractURLFromFile(url)
                embeddings.extend(_imageEmbeddings(url_list))


    print("All image embeddings finished.")
    if output_type == "list":
        return embeddings
    elif output_type == "file":
        _writeEmbeddingsToFile("image",embeddings)
        return "Embeddings successfully written to file."
    return embeddings


                
# User-callable function to generate text embeddings from CLIP
def createTextEmbeddings(input_text, output_type):
    if output_type not in ["list", "file"]:
        raise ValueError("Invalid output type. Expected 'list' or 'file'")
            
    text_embedding_list = []
    if isinstance(input_text, str):  # Check if user input is a string instead of list
        input_text = [input_text]  # Convert it to a list with a single element for processing
        
    # Initiating the model/tokenizer from HuggingFace/CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Here we loop through the text given in the input, return the tensors, and create the vector
    for text in input_text:
        inputs = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
        text_embedding = model.get_text_features(**inputs)
        embedding_as_np = text_embedding.cpu().detach().numpy().tolist()
        text_embedding_list.append((text, embedding_as_np))

    print("Text embeddings successfully created.")
    # Determining what the user selected as the output
    if output_type == "list":
            return text_embedding_list
    elif output_type == "file":
        _writeEmbeddingsToFile("text",text_embedding_list)
        return "Embeddings successfully written to file."
    return text_embedding_list
