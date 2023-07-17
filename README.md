# CLIPCraft
CLIPCraft is a package written in Python for use by Data Scientists/Analysts working with image-to-text and text-to-image implementations.

More specifically, CLIPCraft contains user-friendly functions that are incredibly simple and easy to use, providing a simple and convenient interface for text and image projects.

## Features:

**Embedding Extraction**: Extract embeddings for images and texts using CLIP. <br>
**Image-to-Text**: Generate textual descriptions (captions) for given images using CLIP. Currently, you must provide your own list of captions. <br>
**Text-to-Image from Embeddings**: Generate visual representations from embeddings for given texts using CLIP. 

## Installation
```bash
pip install clipcraft
```
## Usage

CLIPCraft offers 4 functions for users to interact with; <br>
<br>
**createImageEmbeddings(input_urls, output_type)** This function creates image embeddings from a file containing URLs, where input_urls is a string of a filename or a list of strings of filenames, and output_type is the type of output desired for the embeddings; "list" or "file". It returns a list of tuples, where the 0<sup>th</sup> value is the image URL, and the 1<sup>st</sup> value is the resulting embedding.<br>
<br>

**createTextEmbeddings(input_text, output_type)** This function creates text embeddings from a user-input string of text or list of strings of text. output_type is the type of output desired for the embeddings; "list" or "file". It returns a list of tuples, where the 0<sup>th</sup> value is the raw text, and the 1<sup>st</sup> value is the resulting embedding. <br>
<br>

**KNNSearchImage(text_embeddings, image_embeddings)** This function will find the nearest 10 similar images from given text embedding(s) list using K-Nearest-Neighbors. It is designed for use by providing the list returned from createTextEmbeddings. text_embeddings should be the return value of createTextEmbeddings, while image_embeddings should be the return value for createImageEmbeddings.<br>
<br>

**KNNSearchText(text_embeddings, image_urls)** This function will find the most similar caption to a given image. It is designed for use by providing the list returned from createTextEmbeddings. text_embeddings should be the return value of createTextEmbeddings, while image_urls should be a single URL or a list of URLs. 

## Example
```python
import clipcraft as cc

file_urls = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/800px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg"

image_embeds = cc.createImageEmbeddings(file_urls, "list")

text_embed_list = ["a picture of a cat"]

text_embeds = cc.createTextEmbeddings(text_embed_list, "list")

cc.KNNSearchImage(text_embeds, image_embeds)

cc.KNNSearchText(text_embeds, ["https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/800px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg"])
```
Note the arguments taken by the KNN functions are the return values of the previous external functions. <br>

The output of KNNSearchImage will be 5 images with the closest euclidean distance in ascending order; that is, the closest related image being output first. <br>
The output of KNNSearchText will be a caption of the image based on the lowest euclidean distance between your list of input captions and the input image.

## Requirements
To run the functions from this package, you must install PyTorch, as the Hugging Face Transformers library is built on top of it. Instructions to do so can be found here: https://pytorch.org/get-started/locally/ <br>
<br>
The requirements for the package can be found in requirements.txt. However, note that these dependencies will be automatically installed when invoking the pip command. 


