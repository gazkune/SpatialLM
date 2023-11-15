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
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification # NOTE: to be tested


from datasets import vqavinvl_dataset_fromfile, clevr_dataset_fromfile, vsr_dataset_fromfile
from models import models


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


class LitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        # MOVE: Load task labels
        self.task_name = args.dataset # NOTE: 'vsr'

        if self.task_name == 'vsr':
            self.num_labels = 1
        else:
            raise NotImplementedError


        print(f'Number of labels: {self.num_labels}')

        
        # Load model, tokenizer and loss function
        self.model_name = args.model # NOTE: in my case the name is bert

        if self.model_name == 'bert':
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
            
        if self.task_name == 'vsr':
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

        # Store test predictions of the model in a list (to write them to a file)
        self.predictions = [] 
        

    def forward(self, batch):        
        
        if self.task_name == 'vsr':
            _, questions, contexts, labels = batch #NOTE: it's not 'questions', but 'captions'. However, to keep code simple, we will use 'questions' here   


        inputs = self.tokenizer(questions, contexts, return_tensors='pt', padding=True, truncation='only_second')
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
        answers = batch[-1]

        if self.task_name == 'vsr':
            # NOTE: answers is a list of booleans
            target = torch.unsqueeze(torch.tensor(answers).float(), dim=1).to(self.model.device) # NOTE: we are converting True to 1.0 and False to 0.0

        loss = self.loss(logits, target)

        # Compute Accuracy
        if self.task_name == 'vsr':
            sigmoid = torch.nn.Sigmoid()
            predictions = torch.where(sigmoid(logits) >= 0.5, 1.0, 0.0) # We use a threshold on the sigmoid activation of the logits
            accuracy = (predictions == target).sum() / predictions.size(0)        

        # Save metrics
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True)
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True)

        # Store the predictions
        if split == 'test':
            self.predictions += predictions.detach().cpu().tolist() # NOTE: at this moment, vqa_v2 predictions are a list, while clevr and vsr are torch.Tensor (fix this)

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, split="test")           
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.prev_num_labels = checkpoint['state_dict']['model.classifier.weight'].size(0)        
        
        return super().on_load_checkpoint(checkpoint)


class VSRDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()            

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.root = args.root
        self.variant = args.vsr_variant
        self.grid_size = args.grid_size
        self.locations = args.locations
        self.attributes = args.attributes

    def train_dataloader(self):
        split = 'train'

        dataset = vsr_dataset_fromfile.VSRDatasetFromfile(root=self.root, variant=self.variant, split=split, grid_size=self.grid_size, locations=self.locations, attributes=self.attributes)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }
        return data.DataLoader(dataset, **params)
 
    def val_dataloader(self):
        split = 'dev'

        dataset = vsr_dataset_fromfile.VSRDatasetFromfile(root=self.root, variant=self.variant, split=split, grid_size=self.grid_size, locations=self.locations, attributes=self.attributes)
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        split = 'test'

        dataset = vsr_dataset_fromfile.VSRDatasetFromfile(root=self.root, variant=self.variant, split=split, grid_size=self.grid_size, locations=self.locations, attributes=self.attributes)
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
        "--dataset", type=str, default="vsr", choices=["vsr"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--root", type=str, default="/gscratch3/users/gazkune/datasets/vinvl_vqa/seq2seq_files", help="Path to the seq2seq files."
    )
    # NOTE: if vsr -> /gscratch3/users/gazkune/datasets/vsr/vsr_seq2seq_files
    parser.add_argument(
        "--vsr_variant", type=str, default="random", choices=["random", "zero-shot"], help="Variant of the VSR dataset."
    )
    parser.add_argument(
        "--grid_size", type=int, default=32, help="Size of the grid for the location token calculation."
    )
    parser.add_argument(
        "--locations", action="store_true", help="Use location tokens in the dataset."
    )
    parser.add_argument(
        "--attributes", action="store_true", help="Use attributes for image description (only for vsr)."
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny version of the dataset for development."
    )
    parser.add_argument(
        "--from_file", type=str, default=None, help="The filename used for the training split (for now). It overrides other previous options."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader."
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
    parser.add_argument(
        "--predictions_filename", type=str, default=None, help="Filename to store the predictions of the model in the test set (only if evaluate is True)."
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

    if args.dataset == 'vsr':
        datamodule = VSRDataModule(args)
    else:
        raise NotImplementedError
    
    print("Dataset loaded!")

    #ckpt_filename = args.run_name + '.pth'
    tb_run_name = args.run_name
    print(f'Run name: {tb_run_name}')
    
    # Use ModelCheckPoint to store best validation model
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_path, monitor='val_accuracy', mode='max', filename=args.run_name, save_weights_only=True, save_top_k=1)

    # Define trainer
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=args.gpus, fast_dev_run=False, logger=logger, max_steps=args.max_steps, accumulate_grad_batches=args.accumulate_grad_batches, val_check_interval=args.val_check_interval, precision=args.precision) # NOTE: we use this for ModelCheckpoint. Not tested!

    # NOTE: accumulate_grad_batches=4 . Ex: to have a batch size of 56, I have to use 14 (56/4)
    # NOTE: val_check_interval -> if float (percentage of epoch); if int, number of steps to run validation

    # Train model
    print(f'Training starts! Max steps: {args.max_steps}')
    trainer.fit(model, datamodule)
    print("Training finished!")

    # Evaluate model
    if args.evaluate:
        print(f'Loading {checkpoint_callback.best_model_path} with val accuracy of {checkpoint_callback.best_model_score} to test')
        print('Testing starts!')
        trainer.test(ckpt_path = 'best')
        print('Testing finished!')

        # Store the results, if requested
        if args.predictions_filename != None:
            print(f'Storing predictions in file {args.predictions_filename}')
            with open(args.predictions_filename, 'w') as fd:
                for pred in model.predictions:
                    # NOTE: pred is a list of one float -> [1.0]
                    fd.write(f'{int(pred[0])}\n')

            print('Predictions stored')

if __name__ == "__main__":    
    main()
