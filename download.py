'''
src = http://tacodataset.org/
This script downloads TACO's images from Flickr given an annotation json file
Code written by Pedro F. Proenza, 2019

Modifications made by Max Wroe:
    - Alter script to work with new annotation data (4 classifications) for COMP3330
    - No longer sort files into batch subdirectories, place in data folder with name matching image ID
'''

import os.path
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

annotations_path = './data/annotations_comp3330.json'
dataset_dir = "./data/images/unofficial"

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

# Load annotations
with open(annotations_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):

        image = annotations['images'][i]

        file_name = image['file_name']
        image_id = str(image['id'])
        #image_id should have leading 0's to make it 6 characters long
        original_length = len(image_id)
        if(original_length < 6):
            leading_zero_string = "0" * (6 - original_length)
        new_name = leading_zero_string + image_id #create the new name
        image_name = image_id + "." + file_name.split('.')[-1] #The name of the file (ID plus extension)
        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_path = os.path.join(dataset_dir, image_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            # Load and Save Image
            response = requests.get(url_original)
            img = Image.open(BytesIO(response.content))
            if img._getexif():
                img.save(file_path, exif=img.info["exif"])
            else:
                img.save(file_path)

        # Show loading bar
        bar_size = 30
        x = int(bar_size * i / nr_images)
        print('Loaded')
        i+=1

    print('Finished\n')
