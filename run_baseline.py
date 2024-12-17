import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, metavar='TASK', choices=['all', 'caption', 'vqa', 'mixed'], default='mixed', help='Task type')
parser.add_argument('--model', type=str, metavar='MODEL', choices=['minigpt4', 'blip2'], default='blip2', help='Model type')
parser.add_argument('--continuous_sample', type=int, default=9, help='CONTINUOUS SAMPLE')
parser.add_argument('--gpu_id', type=str, default="4")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
from typing import *
import torch
import types
from statistics import mean
from dataclasses import asdict
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, CapVQADataset
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, UniKEHyperParams
from easyeditor.util.hparams import HyperParams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import pickle
from utils import *
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
    
MANUAL_CONTINUOUS_SAMPLE = args.continuous_sample

def train_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_MiniGPT4_Caption"
    )
    
    trainer.run()    


def train_MEND_MiniGPT4_Caption_eval():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4-cap-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_MiniGPT4_Caption"
    )
    
    trainer.run()    



def train_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_MiniGPT4_VQA"
    )
    
    trainer.run() 

def train_MEND_MiniGPT4_VQA_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams, size=100)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams, size=100)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
  
       
def train_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_Caption"
    )
    
    trainer.run()

     
def train_MEND_Blip2OPT_Caption_eval():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2-cap-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_Caption"
    )
    
    trainer.run()


   
def train_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_VQA"
    )
    
    trainer.run()   
   
def train_MEND_Blip2OPT_VQA_eval():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2-vqa-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_VQA"
    )
    
    trainer.run()   

  
def train_MEND_Blip2OPT_VQA_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
    
def train_MEND_Blip2OPT_VQA_Vision_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()  
      
def train_MEND_Blip2OPT_VQA_Vision():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_VQA_Vision"
    )
    
    trainer.run()    
    
def test_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalHparams.from_hparams('./hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams, size=100)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def train_MEND_MiniGPT4_VQA_eval():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4-vqa-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_MEND_MiniGPT4_VQA"
    )
    
    trainer.run()    


def edit_MEND_Blip2OPT_Caption():
    hparam_path = './hparams/MEND/blip2-mend-caption.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/caption_eval_edit/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
    
    
def edit_MEND_Blip2OPT_VQA():
    hparam_path = './hparams/MEND/blip2-mend-vqa.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/vqa_eval/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


def edit_MEND_MiniGPT4_Caption():
    hparam_path = './hparams/MEND/minigpt4-mend-caption.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/caption_eval_edit/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
    
    
def edit_MEND_MiniGPT4_VQA():
    hparam_path = './hparams/MEND/minigpt4-mend-vqa.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/vqa_eval/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
 

def edit_MEND_MiniGPT4_Mixed():
    hparam_path = './hparams/MEND/minigpt4-mend-mixed.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='mixed'
    )
    parse_result(metrics, f'./logs/baseline/mixed/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


def edit_MEND_Blip2OPT_Mixed():
    hparam_path = './hparams/MEND/blip2-mend-mixed.yaml'
    hparams = MENDMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='mixed'
    )
    parse_result(metrics, f'./logs/baseline/mixed/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


def train_MEND_MiniGPT4_Mixed():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4-mixed.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_MEND_MiniGPT4_Mixed"
    )
    
    trainer.run()


def train_MEND_Blip2OPT_Mixed():
    hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2-mixed.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_MEND_Blip2OPT_Mixed"
    )
    
    trainer.run()
    


if __name__ == "__main__":
    # edit_MEND_Blip2OPT_Caption()
    # edit_SERAC_Blip2OPT_VQA()
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Model: {args.model}")
    LOG.info(f"Continuous Sample: {args.continuous_sample}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.task == 'caption':
        if args.model == 'minigpt4':
            train_MEND_MiniGPT4_Caption_eval()
        elif args.model == 'blip2':
            train_MEND_Blip2OPT_Caption_eval()
    elif args.task == 'vqa':
        if args.model == 'minigpt4':
            train_MEND_MiniGPT4_VQA_eval()
        elif args.model == 'blip2':
            train_MEND_Blip2OPT_VQA_eval()
    elif args.task == 'all':
        edit_MEND_MiniGPT4_Caption()
        edit_MEND_MiniGPT4_VQA()
    elif args.task == 'mixed':
        if args.model == 'minigpt4':
            train_MEND_MiniGPT4_Mixed()
        elif args.model == 'blip2':
            train_MEND_Blip2OPT_Mixed()
