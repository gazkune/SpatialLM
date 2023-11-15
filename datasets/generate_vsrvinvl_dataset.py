"""
A script to generate a Visual Spatial Reasoning (VSR) VINVL dataset and store the correspondent files
"""
from pathlib import Path
from collections import defaultdict
from collections import Counter

import os, csv, json, sys, time, random
import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split_type", type=str, default="random", choices=["random", "zero-shot"],
        help="Split type (random or zero-shot) of the VSR dataset to analyse."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "dev", "test"],
        help="Split of the VSR dataset to analyse for the given type."
    )
    parser.add_argument(
        "--vsrroot", type=str, default="/home/gorka/datasets/visual-spatial-reasoning", help="Path to the VSR dataset root."
    )
    parser.add_argument(
        "--vinvlroot", type=str, default="./datasets/vinvl-predictions", help="Path to the VinVL annotations."
    )    
    parser.add_argument(
        "--grid_size", type=int, default=32, help="The size of the grid (grid_size x grid_size) to generate location tokens."
    )
    parser.add_argument(
        "--outputfolder", type=str, default="./datasets/vsr_seq2seq_files", help="Path to the file to store fine results."
    )    
    parser.add_argument(
        "--ignore_locations", action="store_true", help="Whether we have to use location tokens in image descriptions."
    )    
    parser.add_argument(
        "--ignore_attributes", action="store_true", help="Whether we have to use object attributes in image descriptions."
    )

    args = parser.parse_args()
    return args

def bb2grid(bbox, grid_size):
    # This function calculates the grid coordinates of a given bounding box
    # Input:
    # - bbox: a list of 4 normalized coordinates: [x0, y0, x1, y1] (top-left x and y: bottom-right x and y)
    # - grid_size: int number with the size of the grid (eg: 32 means a grid of 32x32)
    # Output:
    # - grid_x0, grid_y0, grid_x1, grid_y1

    x0, y0, x1, y1 = bbox

    grid_x0 = min(int(x0*grid_size), grid_size-1)
    grid_y0 = min(int(y0*grid_size), grid_size-1) 

    grid_x1 = min(int(x1*grid_size), grid_size-1)
    grid_y1 = min(int(y1*grid_size), grid_size-1)

    return grid_x0, grid_y0, grid_x1, grid_y1

def main():
    # Monitor the needed time to load the dataset
    start = time.time()    

    print("Parsing args...")
    args = parse_args()
    print(f'Split type: {args.split_type}')
    print(f'Split: {args.split}')
    print(f'Ignore locations: {args.ignore_locations}')
    print(f'Ignore attributes: {args.ignore_attributes}')

    # Handle all file stuff here (input and output)

    vsr_annotations = os.path.join(args.vsrroot, args.split_type, f'{args.split}.jsonl')
    
    cocotrain_predictions_file = os.path.join(args.vinvlroot, 'cocotrain2014', 'clean_predictions.tsv')  
    cocoval_predictions_file = os.path.join(args.vinvlroot, 'cocoval2014', 'clean_predictions.tsv')

    # We need two output files: 1) train.source 2) train.target 
    # Source file
    source_file = f'{args.split}_grid{args.grid_size}'
    target_file = f'{args.split}_grid{args.grid_size}'
    
    if args.ignore_locations == True:
        source_file += '_nolocation'
        target_file += '_nolocation'
    if args.ignore_attributes == True:
        source_file += '_noattr'
        target_file += '_noattr'
            
    source_file += '.source'
    target_file += '.target'    
                           
    source_filename = os.path.join(args.outputfolder, args.split_type, source_file)
    target_filename = os.path.join(args.outputfolder, args.split_type, target_file)
    
    print(f'files to store: {source_filename} and {target_filename}')    

    # First of all, read VINVL prediction files (train and val) and store in a convenient data strcuture
    # Process the images and the predictions
    # Build a dictionary to store image_id:descriptions
    # NOTE: it is faster to process all images than to filter them using valid_image_ids
    imageid_to_description = {}    

    predictions = []
    with open(cocotrain_predictions_file) as trainfile:
        predictions = list(csv.reader(trainfile, delimiter="\t"))        
    with open(cocoval_predictions_file) as valfile:
        predictions = predictions + list(csv.reader(valfile, delimiter="\t"))    

    for prediction in predictions:
        # Each prediction contains:
        # element 0: the image ID (COCO_val2014_000000...)
        # element 1: "objects": a list of dicts with keys: "rect", "bbox_id", "class", "conf", "attributes", "attr_scores"
        image_filename = prediction[0] # NOTE: .jpg extension not included in the name
        image_id = int(prediction[0].split('_')[-1]) # NOTE: this is the number only (without COCO_val2014 or similars)            
            
        image_predictions = json.loads(prediction[1]) # NOTE: This is very important! Convert the string to a dictionary
        H, W = image_predictions["image_h"], image_predictions["image_w"]

        img_description = ''
        
        for obj in image_predictions["objects"]:
            # Exract the relevant info from an object
            object_name = obj["class"]
            attributes = obj["attributes"]

            # Generate the image description
            descr = ' '.join(attributes)

            if args.ignore_locations == True:
                if args.ignore_attributes == True:
                    img_description += str(f'{object_name} ')
                else:
                    img_description += str(f'{object_name} {descr} ')
            else:
                rect = obj["rect"]

                # First normalize the bbox
                # Retrieve the object bbox
                x0, y0, x1, y1 = rect
                # Normalize the bbox using image height and width            
                x0 = x0 / W
                y0 = y0 / H
                x1 = x1 / W
                y1 = y1 / H
                    
                # Obtain the grid coordinates of x0, y0, x1, y1 (top-left and bottom_right points)
                grid_x0, grid_y0, grid_x1, grid_y1 = bb2grid([x0, y0, x1, y1], args.grid_size)
                if args.ignore_attributes == True:
                    img_description += str(f'{grid_x0} {grid_y0} {grid_x1} {grid_y1} {object_name} ')    
                else:
                    img_description += str(f'{grid_x0} {grid_y0} {grid_x1} {grid_y1} {object_name} {descr} ')

            imageid_to_description[image_id] = img_description
    
    # Check whether everything is well processed
    print(f'Number of images from VINVL train+val: {len(imageid_to_description.keys())}')
    
    # Process VSR annotations
    with open(vsr_annotations, 'r') as json_file:
        vsr_annotation_list = list(json_file)

    with open(source_filename, 'w') as source, open(target_filename, 'w') as target:
        for i, json_str in enumerate(vsr_annotation_list):
            annotation = json.loads(json_str)        
            image = annotation["image"]
            image_id = int(image.split('.')[0])
            caption = annotation["caption"]
            img_description = imageid_to_description[image_id]
            label = annotation["label"]            

            # Write a line in all files
            source.write(f'caption: {caption} context: {img_description}\n')
            target.write('True\n') if label == 1 else target.write('False\n')
            
    
    print(f'Time to generate the dataset: {time.time() - start}s')


if __name__ == "__main__":
    main()