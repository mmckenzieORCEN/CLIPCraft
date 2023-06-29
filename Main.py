from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import json
import gzip
import pandas as pd
import torch
import numpy

#Internal function that extracts URL from user-input file
def _extractURLFromFile(input_urls):
    if isinstance(input_urls, str):  # Check if user input is a string
        input_urls = [input_urls]  # Convert it to a list for processing

    urls = []
    for file_path in input_urls: 
    # Detect file type based on extension
        file_extension = file_path.rsplit('.', 1)[-1].lower()
        #If the file extension is json, we can load the data and call our helper function
        if file_extension == 'json':
            with open(input_urls, 'r') as datafile:
                data = json.load(datafile)
                _extractURLFromJSON(data, urls)
    else:
        #If the file is not json, we read line by line and extract the data per line
        with open(input_urls, 'r') as datafile:
            for line in datafile:
                url = line.strip()
                if url:  # Skips empty lines
                    urls.append(url)
    return urls
    
#Helper function to extract the URLs from a json file
def _extractURLFromJSON(data, urls):
    #If the data is in a list, we loop through and recursively call our function
    if isinstance(data, list):
        for item in data:
            _extractURLFromJSON(item, urls)
    #If it is a dictionary, we loop through the dict and find values in URL format
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str) and value.startswith('http'):
                urls.append(value)
    #Checking if the value is a list or dict
            elif isinstance(value, (dict, list)):
                _extractURLFromJSON(value, urls)

#Internal function to generate our image embeddings
def _imageEmbeddings(url_list):
    #Loading pre-trained CLIP model/processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url_embeddings_list = []
    #Looping through each value in the input list
    for url in url_list
        try:
            #Grabbing our image. Timeout for URLs that are unretrievable. 5 secs for slow retrieval.
            response = requests.get(url, timeout = 5)  
            content_type = response.headers.get('content-type')
            image = Image.open(BytesIO(response.content))
            #Inputs for the model; Can't change text inputs or get an exception
            inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True) 

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            #Getting the image embeds from the model and adding them to our list with the origin URL
            embeddings = outputs.image_embeds[0].detach().numpy().tolist()
            url_embeddings_list.append((url, embeddings))

        except Exception as e:
            print(f"Skipping URL: {url}")
            print(e)
    return url_embeddings_list

#Internal function to write the embeddings to a file when the user chooses to do so
def _writeEmbeddingsToFile(type, input_image_embeddings, input_text_embeddings):
    #If our desired embedding type is of images
    if type == "image":
        with gzip.open("CLIP_image_embeddings.json.gz", "wt") as datafile:
            for i in range(len(input_image_embeddings)):
                new_row = {
                    'image_url': input_image_embeddings[i][0], #Create json pairs from the list
                    'image_embedding': input_image_embeddings[i][1],
                }
                datafile.write(json.dumps(new_row) + '\n')
                print("Successful dump #" + str(i + 1) + '\n')
                
    #If our desired embedding type is of text
    elif type == "text":
        with gzip.open("CLIP_text_embeddings.json.gz", "wt") as datafile:
            for i in range(len(input_text_embeddings)):
                new_row = {
                    'original_text': input_text_embeddings[i][0], 
                    'text_embedding': input_text_embeddings[i][1],
                }
                datafile.write(json.dumps(new_row) + '\n')
                print("Successful dump #" + str(i + 1) + '\n')

#Function for users to call to create Image embeddings
def createImageEmbeddings(input_urls): 
while True:
        output_type = input("Enter the desired output type (list or file): ")

        if output_type in ["list", "file"]:
            break
        else:
            print("Invalid output type. Expected 'list' or 'file'. Please try again")
    
    embeddings = []
    if isinstance(input_urls, str):
        # Read the file and process its contents
        url_list = _extractURLFromFile(input_urls)
        embeddings.append((url, _imageEmbeddings(url_list)))
                             
    if isinstance(input_urls, list):
        embeddings = []
        # Multiple URLs or file names provided
        for url in input_urls:
            if isinstance(url, str):
                # Process a file
                url_list = _extractURLFromFile(url)
                embeddings.append((url, _imageEmbeddings(url_list)))
            elif isinstance(url, str) and url.startswith('http'):
                # Process a URL
                embeddings.append((url, _imageEmbeddings(url)))

    if output_type == "list":
        return embeddings
    elif output_type == "file":
        _writeEmbeddingsToFile("image",embeddings)
        return "Embeddings successfully written to file."
            else:
                raise ValueError("Invalid item in the input list. Expected URL (string).")
    else:
        raise ValueError("Invalid input type. Expected list or file name (string).")


                
#User-callable function to generate text embeddings from CLIP
def createTextEmbeddings(input_text):
    while True:
        output_type = input("Enter the desired output type (list or file): ")

        if output_type in ["list", "file"]:
            break
        else:
            print("Invalid output type. Expected 'list' or 'file'. Please try again")
            
    text_embedding_list = []
    if isinstance(input_text, str):  # Check if user input is a string instead of list
        input_text = [input_text]  # Convert it to a list with a single element for processing
        
    #Initiating the model/tokenizer from HuggingFace/CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    #Here we loop through the text given in the input, return the tensors, and create the vector
    for text in input_text:
        inputs = tokenizer(text, return_tensors = "pt", padding=True, truncation=True)
        text_embedding = model.get_text_features(**inputs)
        embedding_as_np = text_embedding.cpu().detach().numpy()
        text_embedding_list.append(text, embedding_as_np)
        
    if output_type == "list":
            return text_embedding_list
    elif output_type == "file":
        _writeEmbeddingsToFile("text",text_embedding_list)
        return "Embeddings successfully written to file."

#User-callable function to display KNN, or most similar images to given text
def KNNSearchImage(text_embeddings, image_embeddings):
    #Since we returned the image embeddings as a list of tuples, we grab the embedding tuples
    image_embeddings_array = np.array([embedding for _, embedding in image_embeddings])
    text_embeddings_array = np.array([embedding for _, embedding in text_embeddings])
    
    result = []  # List to store the result for each text embedding

    for i in range(len(text_embeddings_array)):
        # This is our Euclidean distance to determine nearest neighbors
        distances = np.linalg.norm(image_embeddings_array - text_embeddings_array[i], axis=1)

        # Grab the indexes of the top 10 similar images
        index_10_similar = np.argsort(distances)[:10]

        # Retrieve the nearest URLs based on the indices
        nearest_images = []
        for j in index_10_similar:
            image_url = image_embeddings[j][0]
            # Here we get the actual image data and append it to a list. This way we can have a viewable image
            response = requests.get(image_url)
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            nearest_images.append((image, image_url))

        result.append((text_embeddings[i][0], nearest_images))

    return result

#User-callable function to display KNN, or most similar text given an image
def KNNSearchText(text_embeddings, image_embeddings):
    # Since we returned the image embeddings as a list of tuples, we grab the embedding tuples
    image_embeddings_array = np.array([embedding for _, embedding in image_embeddings])
    text_embeddings_array = np.array([embedding for _, embedding in text_embeddings])

    result = []  # List to store the result for each image embedding

    for i in range(len(image_embeddings_array)):
        # This is our Euclidean distance to determine nearest neighbors
        distances = np.linalg.norm(text_embeddings_array - image_embeddings_array[i], axis=1)

        # Grab the indexes of the top 3 similar texts
        index_3_similar = np.argsort(distances)[:3]

        # Retrieve the nearest texts based on the indices
        nearest_texts = []
        for j in index_3_similar:
            text = text_embeddings[j][0]
            nearest_texts.append(text)

        #Here we are grabbing the URL from the array, then grabbing the actual image to display
        image_url = image_embeddings[i][0]
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(BytesIO(image_data))

        result.append((image_url, image, nearest_texts))

    return result