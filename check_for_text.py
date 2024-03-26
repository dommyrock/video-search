# Extracts all the frames with the any amount of 'text' in them 
import os
import cv2
import pytesseract
from prettytable import PrettyTable
import concurrent.futures

def process_image(filename):
    filepath = os.path.join(dir_path, filename)

    img = cv2.imread(filepath)

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img)

    # If any text is detected in the image, return the filename
    if text.strip():
        return filename

dir_path = './frames'
image_files = [f for f in os.listdir(dir_path)]

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Use the executor to map the process_image function to the image files
    filenames_with_text = list(executor.map(process_image, image_files))

# Remove None values from the list
filenames_with_text = [f for f in filenames_with_text if f is not None]

# Create a pretty table
table = PrettyTable()
table.field_names = ["Images with txt"]
for filename in filenames_with_text:
    table.add_row([filename])

# Print the table
print(table)
