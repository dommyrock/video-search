import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
data_index = faiss.read_index("data.bin")

# Prep images and index
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img
  
# Prep to search the index
def search_index(index: faiss.IndexFlatL2, embeddings: list, k: int = 10) -> list:
    """
    Search the index for the images that are most similar to the provided image.
    """
    D, I = index.search(np.array(embeddings[0].reshape(1, -1)), k)

    return I[0]
  
# Search the Index
search_file = "./dataset/frames/frame216.jpg"
img = cv2.resize(cv2.imread(search_file), (416, 416)) 

print("Input image:") # 1st one with 100 similarity is the one being searched on

images = []
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
        images.append(root  + '/'+ file)
            
with torch.no_grad():
    embedding = dinov2_vits14(load_image(search_file).to(device))
    indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

    for i, index in enumerate(indices):
        print()
        print(f"Image {i}: {images[index]}")