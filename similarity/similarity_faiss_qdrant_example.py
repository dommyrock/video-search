# pip install transformers faiss-gpu torch Pillow qdrant_client
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import faiss
from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)

#load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Device detected: {device}')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

#Define a function that normalizes embeddings and add them to the Qdrant vector database
def add_vector_to_index(embedding, filename):
    #convert embedding to numpy
    embedding = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    embedding = np.float32(embedding)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(embedding)
    #Add to Qdrant vector database
    
    
    ###########TODO################## fix this [look api docs]
# assert len(kwargs) == 0, f"Unknown arguments: {list(kwargs.keys())}"
# AssertionError: Unknown arguments: ['ids', 'labels', 'payload']
    client.upsert(
        collection_name='frames_collection',
        points=np.array([embedding]),
        ids=[filename],
        labels=[filename],
        payload={}
    )

#Create a function that processes an image and adds it to the Qdrant vector database
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    # Move the tensor to CPU before passing it to the add_vector_to_index function
    features = features.mean(dim=1).cpu()
    add_vector_to_index(features, os.path.basename(image_path))

#Get images in the dataset folder
images = []
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
        image_path = os.path.join(root, file)
        images.append(image_path)

#Process all images
for image_path in images:
    process_image(image_path)