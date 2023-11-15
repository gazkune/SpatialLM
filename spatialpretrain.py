from copy import deepcopy

import argparse
import json
from typing import Any, Dict
import numpy as np
import os, sys
import math

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch import optim
from torch.utils import data

import transformers
from transformers import BertTokenizer, BertForSequenceClassification


from datasets import coco_for_spatial_pretraining
from models import models


def accuracy_score(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))

    
# TODO: move this to the suitable place
# TODO: can I implement it more efficiently
def build_target_tensor(valid_answers, ans_ids):
    """
    Given valid answers for each instance of the batch, it returns the target tensor
    :param valid_answers: Valid answers of this batch, a list of answers per instance. (bs, 10)
    :param ans_ids: Id of each answer in our answer vocabulary. Dictionary with key (answer string) and value (answer id in the vocab)
    :return: Tensor of shape (bs, n_class)
    """
    target = torch.zeros((len(valid_answers), len(ans_ids.keys())))

    for i, answers in enumerate(valid_answers): # Iterate through the batch
        n = 0
        for ans in answers:
            try:
                target[i, ans_ids[ans]] = 1                
            except KeyError:
                n += 1                
        if n == len(valid_answers):
            target[i, ans_ids["UNK"]] = 1
    return target


# TODO: move this to the suitable place
def generate_file_name(model, spatial_embedding, captions, spatial_embedding_dim, output_path):
    filerootname = ''
    run_number = 0
    if spatial_embedding == 'grid':
        filerootname = model + '_' + spatial_embedding + '_' + str(spatial_embedding_dim)
    elif spatial_embedding == 'rect':
        filerootname = model + '_' + spatial_embedding
    elif spatial_embedding == 'none':
        filerootname = model + '_' + captions

    while True:
        filename = filerootname + '_' + str(run_number) + '.pth'
        if os.path.exists(os.path.join(output_path, filename)):
            run_number +=1
        else:
            return filename, filerootname
    

class LitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        # MOVE: Load task labels
        self.dataset = args.dataset # NOTE: 'spatialcoco'

        if self.dataset == 'spatialcoco':
            self.labels = ['yes', 'no']
            self.num_labels = 1 # NOTE: it's binary classification 
            print(f'Number of labels: {self.num_labels}')
        
        # Load model, tokenizer and loss function
        self.model_name = args.model 
        if self.model_name == 'spatialbert':
            self.model = models.SpatialBertForSequenceClassification(
                "bert-base-uncased",
                num_labels=self.num_labels,
                spatial_embedding_dim=self.spatial_embedding_dim,
                input_for_classifier='cls'
            )
        elif self.model_name == 'bert':
            #self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
            self.model = models.BertForVQA('bert-base-uncased', self.num_labels)
        elif self.model_name == 'bert-large':
            self.model = models.BertForVQA('bert-large-uncased', self.num_labels)
        else:
            raise NotImplementedError
            

        self.tokenizer = None
        if self.model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif self.model_name == 'bert-large':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
            
        self.loss = torch.nn.BCEWithLogitsLoss()

        # Define other hyperparameters
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.opt_eps = args.opt_eps
        self.opt_wd = args.opt_wd
        self.scheduler_off = args.scheduler_off

        self.pretrained_on = None
        self.prev_num_labels = 0       
        

    def forward(self, batch):
        
        # question_ids, questions, question_tokens, image_ids, image_tokens, captions, spatial_embeddings, answers
        image_ids, img_descriptions, questions, answers = batch
        
        # Forward pass
        if self.model_name == 'spatialbert':
            logits = self.model(
                question_tokens=question_tokens, # [bs, q_seq_len]            
                image_tokens=image_tokens, # [bs, i_seq_len]
                spatial_embeddings=spatial_embeddings # [bs, i_seq_len, spatial_embeddings_dim]            
            )
        elif self.model_name == 'bert' or self.model_name == 'bert-large':
            inputs = self.tokenizer(questions, img_descriptions, return_tensors='pt', padding=True, truncation='only_second') # NOTE: the best might be truncation='only_second'

            input_ids = inputs['input_ids'].to(self.model.device)
            token_type_ids = inputs['token_type_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        return logits

    def configure_optimizers(self):
        # Define optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        if self.scheduler_off:
            return [optimizer]            
        else:
            scheduler = {
                "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
                "interval": "step"
            }
            return [optimizer], [scheduler]
            
    
    def general_step(self, batch, split="train"):
        
        # Model forward pass
        logits = self(batch)

        # MOVE: Load target data 
        answers = batch[-1] # a list of strings containing 'yes' or 'no'
        # Convert the yes/no answers to 1 or 0 (assuming yes = 1 and no = 0)
        target = [1.0 if ans == 'yes' else 0.0 for ans in answers]
        target = torch.unsqueeze(torch.tensor(target), 1).to(self.model.device)

        #print(f'general_step: logits shape: {logits.size()} | target shape: {target.size()}')
        loss = self.loss(logits, target)

        # Compute Accuracy
        sigmoid = torch.nn.Sigmoid()
        predictions = torch.where(sigmoid(logits) >= 0.5, 1.0, 0.0) # We use a threshold on the sigmoid activation of the logits
        accuracy = (predictions == target).sum() / predictions.size(0)
        
        # Save metrics
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, split="test")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.prev_num_labels = checkpoint['state_dict']['model.classifier.1.weight'].size(0)
        
        if self.prev_num_labels == 2253:
            self.pretrained_on = "okvqa_v1.0"
        elif self.prev_num_labels == 2250:
            self.pretrained_on = "okvqa_v1.1"
        elif self.prev_num_labels == 3129:
            self.pretrained_on = "vqa_v2"
        else:
            self.pretrained_on = "other"
        
        # Remove classifier layer's state dict if it was pretrained in another vqa task
        if self.pretrained_on != self.task_name:
            del checkpoint['state_dict']['model.classifier.1.weight']
            del checkpoint['state_dict']['model.classifier.1.bias']
        
        return super().on_load_checkpoint(checkpoint)


class SpatialCocoDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()            

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.tiny = args.tiny # NOTE: not implemented yet
        self.root = args.root
        self.source = args.source
        self.target_model = args.target_model
        self.location_encoding = args.location_encoding
        self.grid_size = args.grid_size
        self.distractors = args.distractors
        self.attributes = args.attributes
        self.spatial_val_file = args.spatial_val_file


    def train_dataloader(self):
        split = 'train'

        dataset = coco_for_spatial_pretraining.CocoForSpatialPretrainingDataset(root=self.root, source=self.source, split=split, target_model=self.target_model, location_encoding=self.location_encoding, distractors=self.distractors, attributes=self.attributes, grid_size=self.grid_size)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }
        return data.DataLoader(dataset, **params)
    
    # TODO: REMOVE VAL_DATALOADER
    def val_dataloader(self):
        split = 'val'

        dataset = coco_for_spatial_pretraining.CocoForSpatialPretrainingDataset(root=self.root, source=self.source, split=split, target_model=self.target_model, location_encoding=self.location_encoding, distractors=self.distractors, attributes=self.attributes, grid_size=self.grid_size, avoid_contamination=None, val_file=self.spatial_val_file)

        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        split = 'val'

        dataset = coco_for_spatial_pretraining.CocoForSpatialPretrainingDataset(root=self.root, source=self.source, split=split, target_model=self.target_model, location_encoding=self.location_encoding, distractors=self.distractors, attributes=self.attributes, grid_size=self.grid_size, avoid_contamination=None, val_file=self.spatial_val_file)

        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }
        return data.DataLoader(dataset, **params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="bert", choices=["spatialbert", "bert", "bert-large"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )    
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps. (1 == do not use gradient accumulation)"
    )
    parser.add_argument(
        "--scheduler_off", action="store_true", help="Do not use any scheduler"
    )
    parser.add_argument(
        "--val_check_interval", type=float, default=1.0, help="How often within a training epoch to check the val set. (1.0 == every epoch)"
    )    
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--precision", type=int, default=32, choices=[16, 32, 64], help="Precision for the GPUs." 
    )
    parser.add_argument(
        "--opt_eps", type=float, default=1e-8, help="Epsilon value for AdamW optimizer."
    )
    parser.add_argument(
        "--opt_wd", type=float, default=0.0, help="Weight decay value for AdamW optimizer."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=88000, help="Steps to be done during training."
    )
    parser.add_argument(
        "--dataset", type=str, default="spatialcoco", choices=["spatialcoco"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--root", type=str, default="/gscratch3/users/gazkune/datasets/vinvl_vqa/vinvl-predictions", help="Path to the Coco or VinVL prediction files."
    )
    # For coco: "/ikerlariak/asalaberria009/datasets/mscoco"
    parser.add_argument(
        "--source", type=str, default="vinvl", choices=["coco", "vinvl"], help="Source of the object annotations."
    )
    parser.add_argument(
        "--target_model", type=str, default="bert", choices= ["bert", "t5"], help="Generate inputs and outputs for a specific LM."
    )
    parser.add_argument(
        "--location_encoding", type=str, default="token", choices= ["none", "token", "grid", "rect", "none"], help="What kind of spatial representation to use."
    )
    parser.add_argument(
        "--distractors", type=int, default=-1, help="How many objects we should use as distractors (-1: all available)."
    )
    parser.add_argument(
        "--attributes", action="store_true", help="Use VinVL attributes for image descriptions."
    )
    parser.add_argument(
        "--spatial_val_file", type=str, default="/gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-noattr.json", help="Use an already prepared spatial validation file; if None, it will be generated on the fly."
    )
    # Use /gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-nolocation.json to ignore locations
    # Use /gscratch3/users/gazkune/datasets/vinvl_vqa/validation-vinvl-alldistractors-noattr-nolocation.json to ignore attributes and locations
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny version of the dataset for development."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader."
    )
    parser.add_argument(
        "--grid_size", type=int, default=32, help="The size of the grid for the location encoding."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Test model after fine-tuning."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )
    parser.add_argument(
        "--output_path", type=str, default="/gscratch3/users/gazkune/trained_models/vinvl_vqa/", help="Output directory for plots and models."
    )

    args = parser.parse_args()
    return args


def main():
    
    print("Parsing args...")
    args = parse_args()    
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model
    print("Loading model...")
    if args.ckpt is None:
        model = LitModel(args)
    else: 
        model = LitModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=False)
    
    print("Model loaded!")
    
    # Load data
    print("Loading dataset...")
    if args.dataset == 'spatialcoco':
        datamodule = SpatialCocoDataModule(args)
    else:
        raise NotImplementedError
    
    print("Dataset loaded!")
    
    # Define checkpoint filename and tensorboard run name
    # TODO: change this generate_filename method
    # model, spatial_embedding, spatial_embedding_dim, output_path):
    # NOTE: test another approach. Define the name for the checkpoint and TB as an argument: arg.run_name
    if args.run_name == None:
        print('A run name has to be provided')
        sys.exit()
        
    #ckpt_filename = args.run_name + '.pth'
    tb_run_name = args.run_name
    print(f'Run name: {tb_run_name}')

    # Use ModelCheckPoint to store best validation model
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_path, monitor='val_accuracy', mode='max', filename=args.run_name, save_weights_only=True, save_top_k=1)

    # Define trainer
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=args.gpus, fast_dev_run=False, logger=logger, max_steps=args.max_steps, accumulate_grad_batches=args.accumulate_grad_batches, val_check_interval=args.val_check_interval, precision=args.precision)
    # NOTE: accumulate_grad_batches=4 . Ex: to have a batch size of 56, I have to use 14 (56/4)
    # NOTE: val_check_interval -> if float (percentage of epoch); if int, number of steps to run validation

    # Train model
    print("Training starts!")
    trainer.fit(model, datamodule)
    print("Training finished!")
    #trainer.save_checkpoint(os.path.join(args.output_path, ckpt_filename), weights_only=True)

    # Evaluate model
    if args.evaluate:
        print(f'Loading {checkpoint_callback.best_model_path} with val accuracy of {checkpoint_callback.best_model_score} to test')
        print('Testing starts!')
        trainer.test(ckpt_path = 'best')
        print('Testing finished!')


if __name__ == "__main__":    
    main()
