# pip install transformers faiss-gpu torch Pillow
import faiss # only unix based OS support 
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
import time

#load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Device detected: {device}')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

#Populate the images variable with all the images in the dataset folder
images = []
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)

#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    index.add(vector)

#Create Faiss index using FlatL2 type with 384 dimensions as this
#is the number of dimensions of the features
index = faiss.IndexFlatL2(384)

t0 = time.time()
for image_path in images:
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    add_vector_to_index( features.mean(dim=1), index)

#Store the index locally
faiss.write_index(index,"vector.index")

print('Extraction done in :', time.time()-t0)