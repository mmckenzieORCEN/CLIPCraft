from PIL import Image
import numpy as np
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
import requests
from io import BytesIO

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

# User-callable function to display KNN, or most similar images to given text
def KNNSearchImage(text_embeddings, image_embeddings, display_amount):
     #Since we returned the image embeddings as a list of tuples, we grab the embedding tuples
    image_embeddings_array = np.stack([embedding for _, embedding in image_embeddings])
    text_embeddings_array = np.stack([embedding for _, embedding in text_embeddings])
    
    result = []  # List to store the result for each text embedding

    # Loop through each text embedding
    for i in range(len(text_embeddings_array)):
        # This is our Euclidean distance to determine nearest neighbors
        distances = np.linalg.norm(image_embeddings_array - text_embeddings_array[i], axis=1)

        # Grab the indexes of the top 5 similar images
        index_5_similar = np.argsort(distances)[:display_amount]
        # Retrieve the nearest URLs based on the indices
        nearest_images = []
        # Loop through each index in the index list
        for j in index_5_similar:
            # Set image_url to the url from our current index
            image_url = image_embeddings[j][0]
            # Here we get the actual image data and append it to a list. This way we can have a viewable image
            response = requests.get(image_url)
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            image.thumbnail((250, 250))
            # Append the current image and image url to a list
            nearest_images.append((image, image_url))
        #Append the current text as well as the list containing our image and url
        result.append((text_embeddings[i][0], nearest_images))

    # Loop through our 2d array to output the original text, as well as the image url and the image itself
    for text_embedding, nearest_images in result:
        print("Original Text:", text_embedding)
        for image, image_url in nearest_images:
            print("Image URL:", image_url)
            display(image)
        print('Most similar image(s) displayed for text input. \n')
 
        
# User-callable function to display KNN, or most similar text given an image
def KNNSearchText(text_embeddings, image_urls):
    # Use NumPy stack method to convert the 2d array to 1d with embeddings
    text_embeddings_array = np.stack([embedding for _, embedding in text_embeddings])
    result = []
    # Determine if we have only one image url as input
    if isinstance(image_urls, str):
        image_urls = [image_urls]  # Convert single URL to a list

    # Here we create a list of image embeddings by calling internal function
    image_embeddings = _imageEmbeddings(image_urls)
    #Once again using NumPy stack method to get a 1d array of embeddings
    image_embeddings_array = np.stack([embedding for _, embedding in image_embeddings])

    # Iterating over each URL in list
    for i in range(len(image_urls)):

        # Calculate the Euclidean distances between the text and image embeddings
        text_distances = np.linalg.norm(text_embeddings_array[i] - image_embeddings_array, axis=1)

        # Find the index of the closest text embedding
        closest_index = np.argmin(text_distances)
        
        # Retrieve the nearest text based on the index
        nearest_text = text_embeddings[closest_index][0]

        # Append the result for the current image URL to the result list
        result.append((image_urls[i], nearest_text))

    return result