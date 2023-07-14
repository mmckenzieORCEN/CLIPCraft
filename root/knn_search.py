from PIL import Image
import numpy as np
import requests
from io import BytesIO

# User-callable function to display KNN, or most similar images to given text
def KNNSearchImage(text_embeddings, image_embeddings):
     #Since we returned the image embeddings as a list of tuples, we grab the embedding tuples
    image_embeddings_array = np.stack([embedding for _, embedding in image_embeddings])
    text_embeddings_array = np.stack([embedding for _, embedding in text_embeddings])
    
    result = []  # List to store the result for each text embedding

    # Loop through each text embedding
    for i in range(len(text_embeddings_array)):
        # This is our Euclidean distance to determine nearest neighbors
        distances = np.linalg.norm(image_embeddings_array - text_embeddings_array[i], axis=1)

        # Grab the indexes of the top 5 similar images
        index_5_similar = np.argsort(distances)[:5]
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
        print('5 most similar images displayed for text input. \n')
        
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