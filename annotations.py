'''
Provides new annotations for TACO dataset images, alters classifications  from the 60 original classes to be inline
with COMP3330 requirements (Plastic Cups, Plastic Bags, Other Plastic Waste, No Plastic Waste)
This script has been modified from :
https://github.com/wimlds-trojmiasto/detect-waste/blob/7852bce50405b9797e9b2c5b09b4ac033aa52edf/annotations_preprocessing.py
For the purposes of COMP3330 Assignment 2 Part 2


'''

#Run this script to preprocess images from the original imported taco dataset, place them into corresponding folders




import os
import json
from collections import defaultdict, Counter
import funcy
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



NEW_CATEGORY_DIR = 'data/annotations_comp3330.json'
TRAIN_DEST = 'train_annotations.json'
TEST_DEST = 'test_annotations.json'
SOURCE_ANNOTATIONS_DIR = './data/annotations_unofficial.json'


#Save dictionary in coco dataset style
def save_coco(dest, info,
              images, annotations, categories):
    data_dict = {'info': info,
                 'images': images,
                 'annotations': annotations,
                 'categories': categories}
    with open(dest, 'w') as f:
        json.dump(data_dict,
                  f, indent=2, sort_keys=True)
    return data_dict

# filter_annotations and save_coco on akarazniewicz/cocosplit
def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda im: int(im['id']), images)
    return funcy.lfilter(lambda ann:
                         int(ann['image_id']) in image_ids, annotations)

# function based on https://github.com/trent-b/iterative-stratification''', shuffles annotations
#into train or test folders based on a random split
def MultiStratifiedShuffleSplit(images,
                                annotations,
                                test_size):
    # count categories per image
    categories_per_image = defaultdict(Counter)
    max_id = 0
    for ann in annotations:
        categories_per_image[ann['image_id']][ann['category_id']] += 1
        if ann['category_id'] > max_id:
            max_id = ann['category_id']

    # prepare list with count of cateory objects per image
    all_categories = []
    for cat in categories_per_image.values():
        pair = []
        for i in range(1, max_id + 1):
            pair.append(cat[i])
        all_categories.append(pair)

    # multilabel-stratified-split
    strat_split = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                   test_size=test_size,
                                                   random_state=2020)

    for train_index, test_index in strat_split.split(images,
                                                     all_categories):
        x = [images[i] for i in train_index]
        y = [images[i] for i in test_index]
    print('Train:', len(x), 'images, test:', len(y))
    return x, y

# Converts TACO labels to COMP3330 classes (Plastic Bags, Plastic bottles, other, no plastic)
def taco_to_comp3330(label):
    plastic_bags = [
        "degraded_plasticbag", "trash_plastic", "Single-use carrier bag", "Polypropylene bag", "Drink carton","Plastified paper bag"
    ]
    plastic_bottles = [
        "bottleTops", "bottleLabel", "fizzyDrinkBottle", "milk_bottle", "degraded_plasticbottle",
        "Clear plastic bottle",
        "Other plastic bottle", "Plastic bottle cap",
    ]
    other_plastic_waste = [
        "Styrofoam piece", "Garbage bag", "Other plastic wrapper", "microplastics", "smoking_plastic",
        "plasticAlcoholPackaging",
        "alcohol_plastic_cups", "macroplastics", "plastic_cups", "plasticCutlery",
        "plastic_cup_tops", "mediumplastics", "plasticFoodPackaging", "metals_and_plastic", "plastic", "Plastic straw",
        "Other plastic", "Plastic film",
        "Other plastic container", "Plastic glooves", "Plastic utensils", "Tupperware", "Disposable food container",
        "Plastic Film", "Six pack rings", "Spread tub",
        "Disposable plastic cup", "Other plastic cup", "Plastic lid", "Metal lid"

    ]
    # I think this might just confuse the network, these all look pretty similiar to plastic waste....
    no_plastic_waste = [
        "rubbish", "Rubbish", "litter", "Unlabeled litter", "trash_etc", "unknown", "Food waste", "trash_wood", "wood",
        "bio",
        "Corrugated carton", "Egg carton", "Toilet tube", "Other carton", "Normal paper", "Paper bag", "trash_paper",
        "paper",
        "Aluminium blister pack", "Carded blister pack", "Meal carton", "Pizza box", "Cigarette", "Paper cup",
        "Meal carton", "Foam cup",
        "Glass cup", "Wrapping paper", "Magazine paper",
        "Foam food container", "Rope", "Shoe", "Squeezable tube", "Paper straw", "Rope & strings", "Tissues",
        "trash_fabric", "cloth", "non_recyclable", "Battery", "trash_fishing_gear", "other", "Glass bottle",
        "Broken glass", "Glass jar",
        "Glass", "glass", "beerBottle", "wineBottle", "juice_bottles", "waterBottle", "glass_jar", "ice_tea_bottles",
        "spiritBottle",
        "glass_jar_lid", "crisp_large", "crisp_small", "aluminium_foil", "ice_tea_can", "energy_can", "beerCan",
        "tinCan",
        "metal", "trash_rubber", "rubber", "trash_metal", "HDPEM", "PET", "AluCan", "Crisp packet", "Food Can",
        "Aluminium foil",
        "Metal bottle cap", "Aerosol", "Drink can", "Food can", "Pop tab", "Scrap metal",
    ]

    if (label in no_plastic_waste):
        label = "no_plastic_waste"
    elif (label in other_plastic_waste):
        label = "other_plastic_waste"
    elif (label in plastic_bottles):
        label = "plastic_bottles"
    elif (label in plastic_bags):
        label = "plastic_bags"
    else:
        label = "unknown"
    return label

#Now split the dataset into training and test files respectively
def split_coco_dataset(dataset_directory,
                       test_size=0.2):

    with open(dataset_directory, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    info = dataset['info']
    annotations = dataset['annotations']
    images = dataset['images']

    images_with_annotations = funcy.lmap(lambda ann:
                                         int(ann['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in
                                     images_with_annotations, images)

    #If only one category than standard random shuffle
    if len(dataset['categories']) == 1:
        np.random.shuffle(images)
        x = images[int(len(images) * test_size):]
        y = images[0:int(len(images) * test_size)]
        print('Train:', len(x), 'images, valid:', len(y))
    #Otherwise use multi stratified shuffle split
    else:
        x, y = MultiStratifiedShuffleSplit(images, annotations, test_size)


    train = save_coco(TRAIN_DEST, info,
                      x, filter_annotations(annotations, x), categories)
    test = save_coco(TEST_DEST , info, y,
                     filter_annotations(annotations, y), categories)

    print('Finished stratified shuffle split. Results saved in:',
          TRAIN_DEST, TEST_DEST)
    return train, test


#The main script, correctly categorises dataset
if __name__ == '__main__':
    # create directory to store all annotations
    if not os.path.exists(SOURCE_ANNOTATIONS_DIR):
        os.mkdir(SOURCE_ANNOTATIONS_DIR)

    #Move the category ids from the TACO dataset (60 categories) to COMP annotation style (4 Categories)
    #Read in annotations.json
    with open(SOURCE_ANNOTATIONS_DIR, 'r') as f:
        dataset = json.loads(f.read())
    categories = dataset['categories']
    annotations = dataset['annotations']
    info = dataset['info']

    #Change categories to comp3330
    comp3330_categories = dataset['categories']
    for annotation in annotations:
        cat_id = annotation['category_id']
        cat_taco = categories[cat_id - 1]['name']
        comp3330_categories[cat_id - 1]['supercategory'] = taco_to_comp3330(cat_taco)


    comp3330_ids = {}
    comp3330_category_names = []
    category_id = 1
    for category in comp3330_categories:
        if category['supercategory'] not in comp3330_ids:
            comp3330_category_names.append(category['supercategory'])
            comp3330_ids[category['supercategory']] = category_id
            category_id += 1

    #Update the IDs of annotations
    taco_to_comp3330_ids = {}
    for i, category in enumerate(comp3330_categories):
        taco_to_comp3330_ids[category['id']] = comp3330_ids[category['supercategory']]

    annotations_temp = annotations.copy()
    annotations_comp3330 = annotations

    for i, ann in enumerate(annotations):
        annotations_comp3330[i]['category_id'] = \
            taco_to_comp3330_ids[ann['category_id']]
        annotations_comp3330[i].pop('segmentation', None)

    for ann in annotations_temp:
        cat_id = ann['category_id']
        try:
            comp3330_categories[cat_id]['category'] = \
                comp3330_categories[cat_id]['supercategory']
        except:
            continue
        try:
            comp3330_categories[cat_id]['name'] = \
                comp3330_categories[cat_id]['supercategory']
        except:
            continue

    annotations = annotations_comp3330


    #Now create the updated annotations json file
    for cat, items in zip(dataset['categories'], comp3330_ids.items()):
        dataset['categories'] = [cat for cat in dataset['categories']
                                 if cat['id'] < len(comp3330_ids)]
        category, id = items
        cat['name'] = category
        cat['supercategory'] = category
        cat['category'] = category
        cat['id'] = id

    with open(NEW_CATEGORY_DIR, 'w') as f:
        json.dump(dataset, f)

    #Split the data into train test splits, default 80-20 train-test split
    # split_coco_dataset(NEW_CATEGORY_DIR, 0.2)
