"""
A simple baseline for VSR using the spatial rules implemented for SSTD
"""
from pathlib import Path
from collections import defaultdict
from collections import Counter

import torch
import torch.utils.data
import numpy as np

import os, json, sys, time, random, math, csv, re
from string import digits

from datasets import vsr_dataset_fromfile

sstd_relations = ['surrounding', 'inside', 'left of', 'above', 'right of', 'below', 'overlapping', 'separated']

vsr2sstd_correspondence = {"at the right side of": "right of",
                           "at the left side of": "left of",
                           "around": "surrounding",
                           "into": "inside",
                           "on top of": "above",
                           "beneath": "below",
                           "left of": "left of",
                           "right of": "right of",
                           "under": "below",
                           "below": "below",
                           "above": "above",
                           "over": "above",
                           "contains": "surrounding",
                           "within": "inside",
                           "surrounding": "surrounding",
                           "inside": "inside",
                           "outside": "separated"
                           }

def extract_vsr_relations(vsr_dataset_root, variant, split):
    vsr_annotations = os.path.join(vsr_dataset_root, variant, split + '.jsonl')
    # Read the files and store in proper data structures (list of strings)
    # Process VSR annotations
    with open(vsr_annotations, 'r') as json_file:
        vsr_annotation_list = list(json_file)
    
    vsr_relation_list = []
    for i, json_str in enumerate(vsr_annotation_list):
        annotation = json.loads(json_str)                
        relation = annotation["relation"]
        vsr_relation_list.append(relation)
        
    return vsr_relation_list

def extract_subject_object(caption, relation):
    # NOTE: we take advantage of the template-based sentences in VSR to find the subject and object, given the relation
    pre_relation, post_relation = caption.split(' ' + relation + ' ')

    # NOTE: pre_relation will have this structure "The subject is" ("The person is")
    subject = pre_relation.replace('The', '').replace('is', '').strip() 

    # NOTE: post_relation will have this structure "the object" ("the tv")
    object = post_relation.replace('the', '').replace('.', '').strip()

    return subject, object

def extract_subject_object_from_context(subject, object, context):
    # NOTE: given the subject, the object and the context, we have to find the subject and the object in the context and return
    # their locations

    # NOTE: tv is a special case; VinVL uses television, but VSR tv
    if subject == 'tv':
        subject = 'television'
    if object == 'tv':
        object = 'television'

    # NOTE: person is another special case, since VinVL uses frequently some synonyms (man, woman...)
    person_synonyms = ["person", "man", "woman", "boy", "girl"]
    subj_index = -1
    obj_index = -1
    if subject == "person":
        i = 0
        while subj_index == -1 and i < len(person_synonyms):
            subj_index = context.find(' ' + person_synonyms[i] + ' ')
            i += 1
    else:
        subj_index = context.find(' ' + subject + ' ')

    if object == "person":
        i = 0
        while obj_index == -1 and i < len(person_synonyms):
            obj_index = context.find(' ' + person_synonyms[i] + ' ')
            i += 1
    else:
        obj_index = context.find(' ' + object + ' ')

    #print(f'subj_index {subj_index} and obj_index {obj_index}')
    if subj_index == -1 or obj_index == -1:        
        return None, None
    else:
        # NOTE: we found both, subject and object. Now extract location tokens
        subj_loc_region = context[max(subj_index-12, 0):subj_index]
        subj_location_tokens = re.findall(r'\d+', subj_loc_region)
        subj_location_tokens = list(map(int, subj_location_tokens))        

        obj_loc_region = context[max(obj_index-12, 0):obj_index]
        obj_location_tokens = re.findall(r'\d+', obj_loc_region)
        obj_location_tokens = list(map(int, obj_location_tokens))        

        return subj_location_tokens, obj_location_tokens

def predict_with_rules(subj_loc, obj_loc, relation):
    # NOTE: we will predict the answer (True or False) for subject and object location and the spatial relation
    sstd_relation = vsr2sstd_correspondence[relation]
    print(f'sstd relation for {relation} is {sstd_relation}')

    ans = False
    sx0, sy0, sx1, sy1 = subj_loc
    ox0, oy0, ox1, oy1 = obj_loc
    if sstd_relation == "right of":        
        # NOTE: this inplementation gets lower accuracy
        """
        sxmid = sx1 - sx0
        oxmid = ox1 - ox0
        if sxmid > oxmid:
            ans = True
        else:
            ans = False
        """
        # NOTE: first approach        
        if sx0 >= ox1:            
            ans = True
        else:
            ans = False
        
    elif sstd_relation == "left of":
        # NOTE: this inplementation gets lower accuracy
        """
        sxmid = sx1 - sx0
        oxmid = ox1 - ox0
        if sxmid < oxmid:
            ans = True
        else:
            ans = False
        """
        # NOTE: first approach        
        if sx1 <= ox0:
            ans = True
        else:
            ans = False        
    elif sstd_relation == "surrounding":
        if sx0 <= ox0 and sx1 >= ox1 and sy0 <= oy0 and sy1 >= oy1:
            ans = True
        else:
            ans = False
    elif sstd_relation == "inside":
        if sx0 >= ox0 and sx1 <= ox1 and sy0 >= oy0 and sy1 <= oy1:
            ans = True
        else:
            ans = False
    elif sstd_relation == "above":
        # NOTE: this inplementation get higher accuracy
        symid = sy1 - sy0
        oymid = oy1 - oy0
        if symid < oymid:
            ans = True
        else:
            ans = False
        # NOTE: first approach
        """
        if sy1 <= oy0:
            ans = True
        else:
            ans = False
        """
    elif sstd_relation == "below":
        # NOTE: this inplementation get higher accuracy
        symid = sy1 - sy0
        oymid = oy1 - oy0
        if symid > oymid:
            ans = True
        else:
            ans = False
        # NOTE: first approach
        """
        if sy0 >= oy1:
            ans = True
        else:
            ans = False
        """
    elif sstd_relation == "overlapping":
        iou = calculate_iou(subj_loc, obj_loc)
        if iou > 0.0:
            ans = True
        else:
            ans = False
    elif sstd_relation == "separated":
        iou = calculate_iou(subj_loc, obj_loc)
        if iou == 0.0:
            ans = True
        else:
            ans = False        
    
    return ans

def calculate_iou(box1, box2):
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


def main():
    vsr_dataset_root = '/home/gorka/datasets/visual-spatial-reasoning/'
    # NOTE: local variables to load the VSR dataset
    root = './datasets/vsr_seq2seq_files'
    variant = 'random'
    split = 'test'
    grid_size = 32

    print(f'VSR relations in SSTD: {len(vsr2sstd_correspondence.keys())}')
    ds = vsr_dataset_fromfile.VSRDatasetFromfile(root=root, variant=variant, split=split, grid_size=grid_size, locations=True, attributes=False)
    print(f'Dataset length: {len(ds)}')

    vsr_relation_list = extract_vsr_relations(vsr_dataset_root, variant, split)
    print(f'Length of VSR relation list: {len(vsr_relation_list)}')

    counter = 0 # TODO: remove this (it's just for tests)
    solvable_with_rules = 0
    solved_with_rules = 0
    solved_randomly = 0
    hits_overall = 0
    hits_with_rules = 0
    hits_random = 0
    subj_obj_notfound = 0
    rule_solved_relations_stats = {}
    for i in range(len(ds)):
        _, caption, context, label = ds[i]
        relation = vsr_relation_list[i]

        subject, object = extract_subject_object(caption, relation)        

        random_solution = False
        if relation in vsr2sstd_correspondence.keys():
            solvable_with_rules += 1
            subj_loc, obj_loc = extract_subject_object_from_context(subject, object, context)
            if subj_loc == None:
                subj_obj_notfound += 1
                random_solution = True
            else:
                solved_with_rules += 1 
                print(f'caption: {caption}')
                print(f'subject: {subject} | relation: {relation} | object: {object}')                
                print(f'context: {context}')
                print(f'{subject} location tokens: {subj_loc}')
                print(f'{object} location tokens: {obj_loc}')
                ans = predict_with_rules(subj_loc, obj_loc, relation)
                print(f'Predicted answer with rules: {ans} | Real answer: {label}')
                print('---------------------------------')
                if relation not in rule_solved_relations_stats.keys():
                    rule_solved_relations_stats[relation] = {}
                    rule_solved_relations_stats[relation]["hit"] = 0
                    rule_solved_relations_stats[relation]["total"] = 0
                    rule_solved_relations_stats[relation]["acc"] = 0

                rule_solved_relations_stats[relation]["total"] += 1
                if ans == label:
                    rule_solved_relations_stats[relation]["hit"] += 1
                    hits_overall += 1
                    hits_with_rules += 1
        else:
            random_solution = True
            
        if random_solution == True:
            solved_randomly += 1
            ans = random.randint(0, 1)
            if ans == label:
                hits_overall += 1
                hits_random += 1
        
        #if counter > 5:
            #break

        #counter += 1
    
    print('-------------------------------')
    print(f'Dataset length: {len(ds)}')
    print(f'Solvable with rules: {solvable_with_rules} (proportion = {float(solvable_with_rules/len(ds))})')
    print(f'Solved with rules: {solved_with_rules} (proportion = {float(solved_with_rules/len(ds))})')
    print(f'Subject or object not found: {subj_obj_notfound} (proportion over solvables = {float(subj_obj_notfound/solvable_with_rules)})')
    print(f'Solved randomly: {solved_randomly} (proportion = {float(solved_randomly/len(ds))})')
    
    print(f'Hits overall: {hits_overall} (acc = {float(hits_overall/len(ds))})')
    print(f'Hits with rules: {hits_with_rules} (acc over rules = {float(hits_with_rules/solved_with_rules)})')
    print(f'Hits random: {hits_random} (acc = {float(hits_random/solved_randomly)})')
    print(f'Stats for rule-based:')
    for key in rule_solved_relations_stats:
        rule_solved_relations_stats[key]["acc"] = float(rule_solved_relations_stats[key]["hit"] / rule_solved_relations_stats[key]["total"])
    
    print(rule_solved_relations_stats)

    # NOTE: Given that 774 instances could be solved using rules and the rest (1515-265) randomly, we expect a max accuracy of 0.69
    # NOTE: max accuracy is estimated assuming that random guess will get 50% right and rule-based cases will get 100% right
    # NOTE: given that around 60% of the cases solved by rules are correct, a better estimation would be:
    # accuracy for randomly solved instances: 0.5
    # accuracy for rule-based instances: 304/509 = 0.6
    # estimated max accuracy for the system (given subject and object always match): ((1515-265)*0.5 + 774*0.6) / 2024 = 0.54


if __name__ == "__main__":
    main()