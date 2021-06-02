#Run this script to preprocess images from the original imported taco dataset, place them into corresponding folders
import os
import json
DEST_NAME_DIR = 'annotations.json'
SOURCE_ANNOTATIONS_DIR = './data/annotations_unofficial.json'


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

    with open(DEST_NAME_DIR, 'w') as f:
        json.dump(dataset, f)
