import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os

#load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Device detected: {device}')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

#Populate the images variable with all the images in the dataset folder
images = []
image_filenames = []  # This list will store the filenames of the images
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
      image_path = os.path.join(root, file)
      images.append(image_path)
      image_filenames.append(file)  # Add the filename to the list

#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

#Create Faiss index using FlatL2 type with 384 dimensions as this
#is the number of dimensions of the features
index = faiss.IndexFlatL2(384)

import time
t0 = time.time()
for image_path in images:
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    add_vector_to_index( features.mean(dim=1), index)

print('Extraction done in :', time.time()-t0)

#Store the index locally
faiss.write_index(index,"vector.index")

# Also save the filenames to a file
with open('filenames.txt', 'w') as f:
    for filename in image_filenames:
        f.write(filename + '\n')
