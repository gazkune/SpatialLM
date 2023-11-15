"""
A custom dataset with VinVL predictions for the VSR dataset, loaded from prepared files.
"""

from pathlib import Path
from collections import defaultdict
from collections import Counter

import torch
import torch.utils.data

import os, csv, json, sys, time, random, math
import numpy as np

class VSRDatasetFromfile(torch.utils.data.Dataset):
    def __init__(self, root, variant, split, grid_size, locations, attributes):
        """
        A PyTorch Dataset for loading VSR questions and answers with VinVL annotations for the images directly from files.
        Inputs:
        - root: path to the root directory of files
        - variant: dataset variant ('random', 'zero-shot')
        - split: string for the dataset split ('train', 'dev' or 'test')
        - grid_size: the size of the grid used for the positional embeddings
        - locations: whether location tokens have to be used (True or False) 
        - attributes: whether attributes have to be used (True or False)                
        """
        # Monitor the needed time to load the dataset
        start = time.time()
        print("VSRDatasetFromfile loading...")

        self.root, self.variant, self.split, self.grid_size, self.locations, self.attributes = root, variant, split, grid_size, locations, attributes

        rootfilename = f'{self.split}_grid{self.grid_size}'        
        if self.locations == False:
            rootfilename += '_nolocation'

        if self.attributes == False:
            rootfilename += '_noattr'       
        
        self.sourcefile = os.path.join(self.root, self.variant, rootfilename + '.source')
        self.targetfile = os.path.join(self.root, self.variant, rootfilename + '.target')

        print(f'Source file: {self.sourcefile}')
        print(f'Target file: {self.targetfile}')

        # Read the files and store in proper data structures (list of strings)
        with open(self.sourcefile) as sf:
            self.caption_contexts = sf.readlines()
        with open(self.targetfile) as tf:
            self.all_labels = tf.readlines()        

        print(f'Number of questions: {len(self.caption_contexts)} | Number of answers: {len(self.all_labels)}')

    def __len__(self):
        return len(self.caption_contexts)

    def __getitem__(self, idx):
        # Return a caption_context string and a label        
        caption_context = self.caption_contexts[idx]
        # Separate caption and context for Bert-like models
        caption = caption_context.split('context: ')[0].split('caption: ')[1]
        context = caption_context.split('context: ')[1]
        label = self.all_labels[idx].rstrip('\n')
        label = True if label == 'True' else False

        return caption_context.rstrip('\n'), caption.rstrip('\n'), context.rstrip('\n'), label

    def collate_fn(self, batch):
        all_caption_contexts, all_captions, all_contexts, all_labels = [], [], [], []
        for caption_context, caption, context, label in batch:            
            all_caption_contexts.append(caption_context)
            all_captions.append(caption)
            all_contexts.append(context)            
            all_labels.append(label)

        return all_caption_contexts, all_captions, all_contexts, all_labels

def main():
    # NOTE: Some code to test the class here    
    root = './datasets/vsr_seq2seq_files'
    variant = 'random'
    split = 'train'
    grid_size = 32

    ds = VSRDatasetFromfile(root=root, variant=variant, split=split, grid_size=grid_size, locations=True, attributes=False)
    print(f'Dataset length: {len(ds)}')

    caption_context, caption, context, label = ds[3]
    print(caption_context)
    print(f'caption: {caption}')
    print(f'context: {context}')
    print(f'label: {label}')    

    # Test the dataloader    
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=4, collate_fn=ds.collate_fn, shuffle=False)

    caption_contexts, captions, contexts, labels = next(iter(dl))
    
    print(caption_contexts)
    print(captions)
    print(contexts)
    print(labels) 

if __name__ == "__main__":
    main()
