import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import json 
import os
from matplotlib import pyplot as plt
# requires https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# sudo apt-get update && sudo apt-get install -y python3-opencv

# load model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

# Prep images and index
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img
     
     
def create_index(files: list) -> faiss.IndexFlatL2:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    index = faiss.IndexFlatL2(384)

    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(files):
        print(f"Processing img {i+1} of {len(files)}")
        
        embeddings = dinov2_vits14(load_image(file).to(device))
        embedding = embeddings[0].cpu().numpy()
        all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
        index.add(np.array(embedding).reshape(1, -1))

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    faiss.write_index(index, "data.bin")

    return index, all_embeddings

# Prep to search the index
def search_index(index: faiss.IndexFlatL2, embeddings: list, k: int = 10) -> list:
    """
    Search the index for the images that are most similar to the provided image.
    """
    D, I = index.search(np.array(embeddings[0].reshape(1, -1)), k)

    return I[0]
  
#Get images in the dataset folder
images = []
for root, dirs, files in os.walk('./dataset/frames'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)
            
data_index, all_embeddings = create_index(images)
  
# Search the Index
search_file = "./dataset/frames/frame216.jpg"
img = cv2.resize(cv2.imread(search_file), (416, 416)) 
print("Input image:")

with torch.no_grad():
    embedding = dinov2_vits14(load_image(search_file).to(device))

    indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

    for i, index in enumerate(indices):
        print()
        print(f"Image {i}: {images[index]}")
        # show 416x416 previews of results images (doesnt work in console)
        # img = cv2.resize(cv2.imread(files[index]), (416, 416)) 
        # plt.figure(figsize=(16, 16))
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()