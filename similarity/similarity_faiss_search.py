import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os

#input image
image = Image.open('./dataset/frames/frame0.jpg')

#Load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

images = []
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
        images.append(root  + '/'+ file)
        
#Extract the features
with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    #Normalize the features before search
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    #Read the index file and perform search of top-10 images
    index = faiss.read_index("vector.index")
    _d,indices = index.search(vector,10)
    
    # print('distances:', d, 'indexes:', i)
    
    for i, index in enumerate(indices[0]):
        print()
        print(f"Image {i}: {images[index]}")