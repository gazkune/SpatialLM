"""
A custom COCO dataset for spatial pretraining of LMs.
"""
from pathlib import Path
from collections import defaultdict
from collections import Counter

import torch
import torch.utils.data
import numpy as np

import os, json, sys, time, random, math, csv
from string import digits

class CocoForSpatialPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, root, source, split, target_model, location_encoding, grid_size, distractors, attributes, avoid_contamination = None, val_file = None):
        """
        A PyTorch Dataset for loading Coco instances annotations and converting
        them to a spatial pretraining dataset for LMs.
        Inputs:
        - root: path to the root Coco directory or VinVL predictions.
        - source: string for the annotation source ('vinvl' or 'coco')
        - split: string for the dataset split ('train' or 'val')    
        - target_model: for what model to create the pretraining dataset ('bert' or 't5')
        - location_encoding: how location has to be encoded ('none', 'token', 'grid' or 'rect')
        - grid_size: the size of the grid; ignored for 'rect' location_encoding (int)
        - distractors: int to specify how many additional objects we should add to the image description (0: None, -1: all)
        - attributes: bool to specify whether to use attributes in image descriptions (only for Vinvl)
        - avoid_contamination: a list of image_ids to avoid contaminations ('vsr_random' or 'vsr_zero-shot')
        - val_file: if split == 'val', val_file allows to define a json file with validation data. If None, fixed random ops are performed
        """
        # Monitor the needed time to load the dataset
        start = time.time()

        self.root = root
        self.source = source
        self.split = split
        self.target_model = target_model
        self.location_encoding = location_encoding
        self.grid_size = grid_size
        self.distractors = distractors
        self.attributes = attributes
        self.avoid_contamination = avoid_contamination
        self.val_file = val_file

        if self.split == 'val' and self.val_file != None:
            with open(self.val_file, 'r') as f:
                self.val_dataset = json.load(f)
                self.filtered_image_ids = list(self.val_dataset.keys())
            
        if self.avoid_contamination != None:
            # NOTE: this is only prepared for VSR dataset
            if 'vsr' in self.avoid_contamination: 
                vsrroot = '/home/gorka/datasets/visual-spatial-reasoning'
                variant = self.avoid_contamination.split('_')[1]
                vsr_devlist = os.path.join(vsrroot, variant, 'vsr_dev_imageids.txt')
                vsr_testlist = os.path.join(vsrroot, variant, 'vsr_test_imageids.txt')
                with open(vsr_devlist, 'r') as devfile, open(vsr_testlist, 'r') as testfile:
                    devids = devfile.readlines()
                    testids = testfile.readlines()

                self.forbidden_imageids = list(set(devids + testids))
                print(f'Forbidden image ids count: {len(self.forbidden_imageids)}')
        
        if self.val_file == None or self.split == 'train':
            if self.source == 'coco':
                self.init_from_coco()                
            elif self.source == 'vinvl':
                self.init_from_vinvl()            

            spatial_words = ['top left', 'bottom left', 'top right', 'bottom right', 'center', 'left', 'right', 'top', 'bottom', 'wider', 'narrower', 'taller', 'shorter', 'larger', 'smaller', 'surrounding', 'inside', 'left of', 'above', 'right of', 'below', 'overlapping', 'separated']
            self.spatial_word_stats = dict.fromkeys(spatial_words, 0)
            self.answer_stats = dict.fromkeys(['yes', 'no'], 0)

            if self.split == 'val':
                random.seed(1)      
        
        print(f'CocoForSpatialPretrainingDataset: time to load the dataset: {time.time() - start}s')

    def init_from_coco(self):
        instances_file = os.path.join(self.root, 'annotations', f'instances_{self.split}2014.json') # NOTE: We use the 2014 version

        # Build category_to_objname and objname_to_category dictionaries
        # label_to_objname = {id (int): object_name (str)}
        # objname_to_label = {object_name (str): id (int)}
        with open(instances_file, 'r') as f:
            instances = json.load(f)

        self.label_to_objname = {}
        self.objname_to_label = {}
        for category in instances['categories']:
            id = category['id']
            obj = category['name']
            self.label_to_objname[id] = obj
            self.objname_to_label[obj] = id

        # Process image data and store it
        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
            
        for image_data in instances['images']:
            image_id = image_data['id']            
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']            
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = [width, height]
            
        # Add object data from instances
        self.filtered_image_ids = [] # NOTE: We will only use images which have at least one object
        self.image_id_to_objects = defaultdict(list)
        self.image_id_to_labels = defaultdict(list) 
        for object_data in instances['annotations']:
            image_id = object_data['image_id']            
            self.filtered_image_ids.append(image_id) # NOTE: We will have duplicate IDs. Remove them outside the loop
                
            # Retrieve category id of the object
            category = object_data['category_id']
            self.image_id_to_labels[image_id].append(category)

            # Retrieve the object bbox
            x, y, w, h = object_data['bbox']
            # Normalize the bbox using image height and width
            W, H = self.image_id_to_size[image_id]
            x0 = x / W
            y0 = y / H
            x1 = (x + w) / W
            y1 = (y + h) / H
            # We append the normalized bbox
            self.image_id_to_objects[image_id].append(torch.Tensor([x0, y0, x1, y1]))

        self.filtered_image_ids = list(set(self.filtered_image_ids)) # Remove duplicates
        # NOTE: remove also forbidden_image_ids
        if self.avoid_contamination != None:
            print(f'Image ids before filtering with forbidden ids: {len(self.filtered_image_ids)}')
            self.filtered_image_ids = [x for x in self.filtered_image_ids if x not in self.forbidden_imageids]
            print(f'Image ids after filtering with forbidden ids: {len(self.filtered_image_ids)}')

    def init_from_vinvl(self):
        self.split = 'coco' + self.split + '2014'
        predictions_file = os.path.join(self.root, self.split, 'clean_predictions.tsv')

        self.filtered_image_ids = []
        self.image_id_to_objects = defaultdict(list)
        self.image_id_to_labels = defaultdict(list) # NOTE: here we will store objects and attributes as strings
        with open(predictions_file) as file:
            predictions = list(csv.reader(file, delimiter="\t"))
            for prediction in predictions:
                # Each prediction contains:
                # element 0: the image ID (COCO_val2014_000000...)
                # element 1: "objects": a list of dicts with keys: "rect", "bbox_id", "class", "conf", "attributes", "attr_scores"
                image_filename = prediction[0] # NOTE: .jpg extension not included in the name
                image_id = int(prediction[0].split('_')[-1]) # NOTE: this is the number only (without COCO_val2014 or similars)

                self.filtered_image_ids.append(image_id)
                image_predictions = json.loads(prediction[1]) # NOTE: This is very important! Convert the string to a dictionary
                H, W = image_predictions["image_h"], image_predictions["image_w"]

                for obj in image_predictions["objects"]:
                    # Exract the relevant info from an object
                    object_name = obj["class"]
                    attributes = obj["attributes"]
                    attributes = ' '.join(attributes)

                    # Add the new object-attributes string
                    if self.attributes == True:
                        self.image_id_to_labels[image_id].append(f'{object_name}, {attributes}') # NOTE: we add a comma here to separate obj name and attributes
                    else:
                        self.image_id_to_labels[image_id].append(f'{object_name}')
                    rect = obj["rect"]

                    # First normalize the bbox
                    # Retrieve the object bbox
                    x0, y0, x1, y1 = rect
                    # Normalize the bbox using image height and width            
                    x0 = x0 / W
                    y0 = y0 / H
                    x1 = x1 / W
                    y1 = y1 / H

                    # We append the normalized bbox
                    self.image_id_to_objects[image_id].append(torch.Tensor([x0, y0, x1, y1]))
        
        # NOTE: remove forbidden_image_ids
        if self.avoid_contamination != None:
            print(f'Image ids before filtering with forbidden ids: {len(self.filtered_image_ids)}')
            image_ids_noforbidden = [x for x in self.filtered_image_ids if x not in self.forbidden_imageids]
            print(f'Image ids after filtering with forbidden ids: {len(image_ids_noforbidden)}')

    def __len__(self):
        return len(self.filtered_image_ids)

    def __getitem__(self, idx):
        # IDEA (if self.split == 'train'): 
        # 1) Use the index to retrieve an image from self.image_ids list
        # 2) Decide between one-, two- or three-object questions
        # 3.1) One-object questions:
        # - Sample one object and decide between position- or size-related questions.
        # - Position-related question: is obj1 in the top left region? Yes/No
        # - Size-related question: is obj1 taller than wider? Yes/No
        # 3.2) Two-object questions:
        # - Sample randomly two objects from self.image_id_to_labels
        # - Position-related questions: Is obj1 <relation> Obj2? Yes/No
        # - Size-related questions: Is obj1 <relation> than Obj2? Yes/No
        # 3.3) Three-object questions:
        # - Position-related question: Is obj1 between obj2 and obj3? Yes/No (NOTE: how to define between?)
        # - Size-related questions: Is obj1 the tallest? Yes/No
        # 4) Generate the description for the objects depending on self.location_encoding                
        
        # Retrieve the image_id
        image_id = self.filtered_image_ids[idx]        
        if self.split == 'val' and self.val_file != None:            
            return image_id, self.val_dataset[image_id]['img_description'], self.val_dataset[image_id]['question'], self.val_dataset[image_id]['answer']
                
        # Retrieve all the object labels in the image
        obj_labels = self.image_id_to_labels[image_id] # NOTE: if self.source == 'vinvl' and self.attributes, objectname + attributes
        # Retrieve object BBs for image_id
        obj_bbs = self.image_id_to_objects[image_id]
        # Decide btw one- or two-object questions        
        if len(obj_labels) < 2:
            objects_for_questions = 1
        else: # We have two or more objects
            if random.random() < 0.7: # NOTE: this probability is ad-hoc
                objects_for_questions = 2
            else:
                objects_for_questions = 1

        # Choose suitable number of object labels and corresponding BBs randomly
        indices = random.sample(range(len(obj_labels)), k=objects_for_questions)
        #print(f'selected indices: {indices}')
        sel_labels = [obj_labels[x] for x in indices] #obj_labels[indices]
        sel_bbs = [obj_bbs[x] for x in indices] #obj_bbs[indices]
        # Convert object labels to object names if self.source = 'coco'
        if self.source == 'coco':
            sel_obj_names = [self.label_to_objname[x] for x in sel_labels]
            obj_names = [self.label_to_objname[x] for x in obj_labels]
        elif self.source == 'vinvl':
            sel_obj_names = sel_labels
            obj_names = obj_labels
        
        # Generate the image description based on the two objects
        # NOTE: only token locations
        img_description = self.generate_img_description(sel_obj_names, sel_bbs, obj_names, obj_bbs, self.distractors)

        # Randomly choose between a positive question (answer 'yes') or a negative one (answer 'no')
        if random.random() >= 0.5:
            answer = 'yes'
            self.answer_stats['yes'] += 1
        else:
            answer = 'no'
            self.answer_stats['no'] += 1
        
        if objects_for_questions == 1: # One-object question            
            question = self.generate_one_object_qa(sel_bbs[0], sel_obj_names[0], answer)
        if objects_for_questions == 2: # Two-object question
            # Generate a size- or position-related question
            if random.random() > 0.5:
                # Generate a size related question for those two objects
                question = self.generate_size_qa(sel_bbs, sel_obj_names, answer)
            else:
                # Generate a position related question for those two objects
                question = self.generate_pos_qa(sel_bbs, sel_obj_names, answer)

        #return image_id, sel_obj_names, sel_bbs, img_description, question, answer
        return image_id, img_description, question, answer
    
    def generate_img_description(self, sel_obj_names, sel_bbs, obj_names, obj_bbs, distractors):
        # This function generates an image description using the input objects and their bounding boxes
        # NOTE: distractor objects can be added (from obj_names and obj_bbs). 
        # We do not include any object with the same name as the selected ones    
        
        # NOTE: for vinvl, sel_obj_names is a list of 'cat, white black small' like strings
        # We need to extract only the object name to properly handle distractor objects
        sel_obj_names_vinvl = []
        if self.source == 'vinvl':
            for objattr in sel_obj_names:
                if self.attributes == True:
                    objname = objattr.split(',')[0]
                else:
                    objname = objattr # NOTE: in this case, we do not have attributes

                sel_obj_names_vinvl.append(objname)
        
        # Select distractor objects
        if distractors != 0:
            distractor_obj_names = []
            distractor_obj_bbs = []
            for bbox, objname in zip(obj_bbs, obj_names):
                # See whether we have to stop
                if len(distractor_obj_names) == distractors:
                    break

                if self.source == 'coco':
                    obj = objname
                elif self.source == 'vinvl':
                    if self.attributes == True:
                        obj = objname.split(',')[0]
                        objname = objname.replace(',', '') # We remove the comma
                    else:
                        obj = objname # NOTE: no attributes here
                
                # Check whether objname is already in sel_obj_names to add it
                # NOTE: for vinvl, we need to check in sel_obj_names_vinvl
                if self.source == 'coco':
                    if not obj in sel_obj_names:
                        distractor_obj_names.append(objname) # NOTE: this way, for coco we only add the object name
                        distractor_obj_bbs.append(bbox)
                elif self.source == 'vinvl':
                    if not obj in sel_obj_names_vinvl:
                        distractor_obj_names.append(objname) # NOTE: this way, for vinvl we add obj+attributes
                        distractor_obj_bbs.append(bbox)
            
            # Concatenate selected objects and distractors
            sel_obj_names += distractor_obj_names
            sel_bbs += distractor_obj_bbs
            # Shuffle them to avoid any bias
            aux_list = list(zip(sel_obj_names, sel_bbs))
            random.shuffle(aux_list)
            sel_obj_names, sel_bbs = zip(*aux_list)
            sel_obj_names = list(sel_obj_names)
            sel_bbs = list(sel_bbs)


        description = ''

        for bbox, objname in zip(sel_bbs, sel_obj_names):
            if self.location_encoding == 'token':
                grid_x0, grid_y0, grid_x1, grid_y1 = self.bb2grid(bbox, self.grid_size)
                description += f'{grid_x0} {grid_y0} {grid_x1} {grid_y1} {objname} '
            elif self.location_encoding == 'none':
                description += f'{objname} '

        # NOTE: remove the commas from original vinvl sel_obj_names
        return description.replace(',', '')

    def bb2grid(self, bbox, grid_size):
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

    def generate_one_object_qa(self, bbox, objname, answer):
        top_left = [0.0, 0.0, 0.5, 0.5]
        bottom_left = [0.0, 0.5, 0.5, 1.0]
        top_right = [0.5, 0.0, 1.0, 0.5]
        bottom_right = [0.5, 0.5, 1.0, 1.0]        

        # Check the relation with different image regions
        region = ''
        if bbox[0] >= top_left[0] and bbox[2] <= top_left[2]: # Left part of the image
            if bbox[1] >= top_left[1] and bbox[3] <= top_left[3]: # top-left region
                region = 'top left' if answer == 'yes' else random.choice(['bottom left', 'top right', 'bottom right', 'center'])
                self.spatial_word_stats[region] += 1
            if bbox[1] >= bottom_left[1] and bbox[3] <= bottom_left[3]: # bottom-left region
                region = 'bottom left' if answer == 'yes' else random.choice(['top left', 'top right', 'bottom right', 'center'])
                self.spatial_word_stats[region] += 1
            else: # only left part
                region = 'left' if answer == 'yes' else random.choice(['right', 'top right', 'bottom right'])
                self.spatial_word_stats[region] += 1
        if bbox[0] >= top_right[0] and bbox[2] <= top_right[2]: # Right part of the image
            if bbox[1] >= top_right[1] and bbox[3] <= top_right[3]: # top-right region
                region = 'top right' if answer == 'yes' else random.choice(['top left', 'bottom left', 'bottom right', 'center'])
                self.spatial_word_stats[region] += 1
            if bbox[1] >= bottom_right[1] and bbox[3] <= bottom_right[3]: # bottom-right region
                region = 'bottom right' if answer == 'yes' else random.choice(['top left', 'bottom left', 'top right', 'center'])
                self.spatial_word_stats[region] += 1
            else: # only right part
                region = 'right' if answer == 'yes' else random.choice(['left', 'top left', 'bottom left'])
                self.spatial_word_stats[region] += 1
        
        if region == '': 
            # Check top and bottom
            if bbox[3] < 0.5: # Top part
                region = 'top' if answer == 'yes' else random.choice(['bottom', 'bottom left', 'bottom right'])
                self.spatial_word_stats[region] += 1
            if bbox[1] > 0.5: # Bottom part
                region = 'bottom' if answer == 'yes' else random.choice(['top', 'top right', 'top left'])
                self.spatial_word_stats[region] += 1

        if region == '':
            region = 'center' if answer == 'yes' else random.choice(['top left', 'bottom left', 'top right', 'bottom right'])
            self.spatial_word_stats[region] += 1

        if self.source == 'coco':
            objname = objname
        elif self.source == 'vinvl':
            objname = objname.split(',')[0]

        question = f'is {objname} in the {region} region?'
        return question
    
    def generate_size_qa(self, boxes, obj_names, answer):
        box_s = boxes[0] # BB of the subject
        box_o = boxes[1] # BB of the object
        sx0, sy0, sx1, sy1 = box_s
        ox0, oy0, ox1, oy1 = box_o
        size_s = abs(sx1 - sx0) * abs(sy1 - sy0)
        size_o = abs(ox1 - ox0) * abs(oy1 - oy0)        
        
        # Randomly choose between wider/narrower (0), taller/shorter (1) and larger/smaller (2)
        relation_type = random.randint(0, 2)
        rel = '' 
        if relation_type == 0: # wider/narrower
            if sx1 - sx0 > ox1 - ox0:
                rel = 'wider' if answer == 'yes' else 'narrower'
                self.spatial_word_stats[rel] += 1
            elif sx1 - sx0 < ox1 - ox0:
                rel = 'narrower' if answer == 'yes' else 'wider'
                self.spatial_word_stats[rel] += 1
        elif relation_type == 1: # taller/shorter
            if sy1 - sy0 > oy1 - oy0:
                rel = 'taller' if answer == 'yes' else 'shorter'
                self.spatial_word_stats[rel] += 1
            elif sy1 - sy0 < oy1 - oy0:
                rel = 'shorter' if answer == 'yes' else 'taller'
                self.spatial_word_stats[rel] += 1
        elif relation_type == 2: # larger/smaller
            if size_s > size_o:
                rel = 'larger' if answer == 'yes' else 'smaller'
                self.spatial_word_stats[rel] += 1
            elif size_o > size_s:
                rel = 'smaller' if answer == 'yes' else 'larger'
                self.spatial_word_stats[rel] += 1

        if self.source == 'coco':            
            obj_names = obj_names
        elif self.source == 'vinvl':
            aux = []
            for obj in obj_names:
                aux.append(obj.split(',')[0])

            obj_names = aux

        question = f'is {obj_names[0]} {rel} than {obj_names[1]}?'
        return question
    
    def generate_pos_qa(self, boxes, obj_names, answer):
        box_s = boxes[0] # BB of the subject
        box_o = boxes[1] # BB of the object
        sx0, sy0, sx1, sy1 = box_s
        ox0, oy0, ox1, oy1 = box_o
        size_s = abs(sx1 - sx0) * abs(sy1 - sy0)
        size_o = abs(ox1 - ox0) * abs(oy1 - oy0)
        xc_s = sx0 + (sx1 - sx0) / 2
        yc_s = sy0 + (sy1 - sy0) / 2
        xc_o = ox0 + (ox1 - ox0) / 2
        yc_o = oy0 + (oy1 - oy0) / 2
        d = np.array([xc_s, yc_s]) - np.array([xc_o, yc_o])
        theta = math.atan2(d[1], d[0])        
        
        if self.source == 'coco':            
            obj_names = obj_names
        elif self.source == 'vinvl':
            aux = []
            for obj in obj_names:
                aux.append(obj.split(',')[0])

            obj_names = aux
        
        # Randomly choose between surrounding/inside/lef of/right of/below/above (0) and overlapping/separated (1)
        relation_type = random.randint(0, 1)
        rel = '' 
        if relation_type == 0:
            threshold = 0.02 #0.05 # This is for surrounding/inside only
            if sx0 - threshold <= ox0 and sx1 + threshold >= ox1 and sy0 - threshold <= oy0 and sy1 + threshold >= oy1:
                rel = 'surrounding' if answer == 'yes' else random.choice(['inside', 'left of', 'above', 'right of', 'below'])
                self.spatial_word_stats[rel] += 1
            elif sx0 + threshold >= ox0 and sx1 - threshold <= ox1 and sy0 + threshold >= oy0 and sy1 - threshold <= oy1:
                rel = 'inside' if answer == 'yes' else random.choice(['surrounding', 'left of', 'above', 'right of', 'below'])
                self.spatial_word_stats[rel] += 1
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                rel = 'left of' if answer == 'yes' else random.choice(['inside', 'surrounding', 'above', 'right of', 'below'])
                self.spatial_word_stats[rel] += 1
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                rel = 'above' if answer == 'yes' else random.choice(['inside', 'left of', 'surrounding', 'right of', 'below'])
                self.spatial_word_stats[rel] += 1
            elif -math.pi / 4 <= theta < math.pi / 4:
                rel = 'right of' if answer == 'yes' else random.choice(['inside', 'left of', 'above', 'surrounding', 'below'])
                self.spatial_word_stats[rel] += 1
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                rel = 'below' if answer == 'yes' else random.choice(['inside', 'left of', 'above', 'right of', 'surrounding'])
                self.spatial_word_stats[rel] += 1
            
            question = f'is {obj_names[0]} {rel} {obj_names[1]}?'            
        else: # Overlapping/separated
            iou = self.calculate_iou(box_s, box_o)
            if iou > 0.0: # there is an overlap
                rel = 'overlapping' if answer == 'yes' else 'separated'
                question = f'is {obj_names[0]} overlapping {obj_names[1]}?' if answer == 'yes' else f'are {obj_names[0]} and {obj_names[1]} separated?'
            else:
                rel = 'separated' if answer == 'yes' else 'overlapping'
                question = f'are {obj_names[0]} and {obj_names[1]} separated?' if answer == 'yes' else f'is {obj_names[0]} overlapping {obj_names[1]}?'
            self.spatial_word_stats[rel] += 1
            
        return question
        

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        box1 : [x1, y1, x2, y2]
        box2 : [x1, y1, x2, y2]

        Returns
        -------
        float
            in [0, 1]
        """
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2
        assert box1_x1 <= box1_x2 
        assert box1_y1 <= box1_y2
        assert box2_x1 <= box2_x2
        assert box2_y1 <= box2_y2

        # determine the coordinates of the intersection rectangle
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        bb2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
        

    def collate_fn(self, batch):
        """
        NOTE: __getitem__ returns: image_id, objects, bbs, img_description, question, answer
        Collate function to be used when wrapping CocoForSpatialPretrainingDataset dataset in a
        DataLoader. Returns a tuple of the following: 
        - all_image_ids: a list (batch_size) containing the ids of questions        
        - all_img_descriptions: a list (batch_size) of strings
        - all_questions: a list (batch_size) of strings
        - all_answers: a list (batch_size) of strings
        """
        # image_id, img_description, question, answer
        all_image_ids, all_image_descriptions, all_questions, all_answers = [], [], [], []
        for image_id, image_description, question, answer in batch:
            all_image_ids.append(image_id)            
            all_image_descriptions.append(image_description)
            all_questions.append(question)
            all_answers.append(answer)

        return all_image_ids, all_image_descriptions, all_questions, all_answers


# Function to store the dataset (designed for the val split) and obtain some statistics
def store_analyze_dataset(ds):    
    
    dataset = {}    
    for i in range(len(ds)):        
        image_id, img_description, question, answer = ds[i]
        dataset[image_id] = {'img_description': img_description, 'question': question, 'answer': answer}
    
    with open("validation.json", "w") as write_file:
        json.dump(dataset, write_file)

# Function to store the dataset in a seq2seq style file
def store_dataset_seq2seq(ds, rounds, filename):
    source_filename = filename + '.source'
    target_filename = filename + '.target'
    ans_filename = filename + '.answers'

    with open(source_filename, 'w') as source, open(target_filename, 'w') as target, open(ans_filename, 'w') as ansfile:
        for round in range(rounds):
            for i in range(len(ds)):
                image_id, img_description, question, answer = ds[i]

                # Process answers
                # NOTE: we need three lists: correct_ans, part_correct2_ans and part_correct1_ans                
                answers = [answer]*10
                correct_ans, part_correct2_ans, part_correct1_ans = [], [], []
                correct_ans.append(answer) # NOTE: we only have one answers here                
                
                # Write a line in all files
                source.write(f'question: {question} context: {img_description}\n')
                
                target.write( "\t".join(correct_ans))
                target.write('[SEP]')
                
                target.write( "\t".join(part_correct2_ans))
                target.write('[SEP]')
                
                target.write( "\t".join(part_correct1_ans))
                target.write('[SEP]')
                
                target.write('\n')

                ansfile.write("\t".join(answers))
                ansfile.write('\n')


# Function to load a validation set with locations and remove them (for token locations)
def remove_location_tokens(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        val_dataset = json.load(f)

    for key in val_dataset:
        img_descr = val_dataset[key]["img_description"]

        # Remove the numbers
        remove_digits = str.maketrans('', '', digits)
        res = img_descr.translate(remove_digits)
        res = res.replace('  ', '') # remove double spaces

        # Insert the new image description
        val_dataset[key]["img_description"] = res

    # Save the new dictionary
    with open(output_filename, 'w') as f:
        json.dump(val_dataset, f)

# Code to test the dataset
def main():
    #remove_location_tokens('validation-vinvl-alldistractors-noattr.json', 'validation-vinvl-alldistractors-noattr-nolocation.json')
    #sys.exit()

    #root = '/home/gorka/datasets/coco' # NOTE: this is for COCO
    root = './datasets/vinvl-predictions' # NOTE: this is for VinVL
    split = 'val'
    source = 'vinvl'
    ds = CocoForSpatialPretrainingDataset(root=root, source=source, split=split, target_model='bert', location_encoding='token', distractors=-1, attributes=False, grid_size=32, avoid_contamination='vsr_random')#, val_file='validation.json')
    print(f'Length of the dataset: {len(ds)}')

    #store_analyze_dataset(ds)
    #store_dataset_seq2seq(ds, rounds=5, filename=f'{split}_spatialsynthetic_grid32')
    #sys.exit()
    
    image_id, img_description, question, answer = ds[0] #ds[458] #ds[1035] #ds[2563] #
    print(f'image id: {image_id}')    
    print(f'image description: {img_description}')
    print(f'question: {question} answer: {answer}')    
    sys.exit()
    # Use a dataloader
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=4, collate_fn=ds.collate_fn, shuffle=True)

    image_ids, img_descriptions, questions, answers = next(iter(dl))

    print(f'image ids: {image_ids}')    
    print(f'image descriptions: {img_descriptions}')
    print(f'questions: {questions}') 
    print(f'answers: {answers}')

if __name__ == "__main__":
    main()
