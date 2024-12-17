import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, metavar='TASK', choices=['all', 'caption', 'vqa', 'mixed'], default='vqa', help='Task type')
parser.add_argument('--continuous_sample', type=int, default=3, help='CONTINUOUS SAMPLE')
parser.add_argument('--model', type=str, metavar='MODEL', choices=['minigpt4', 'blip2'], default='blip2', help='Model type')
parser.add_argument('--gpu_id', type=str, default="4")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
from typing import *
from statistics import mean
from dataclasses import asdict
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, CapVQADataset
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, UniKEHyperParams
from easyeditor.util.hparams import HyperParams
import pickle
from utils import *
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

MANUAL_CONTINUOUS_SAMPLE = args.continuous_sample
test_size = 21


def edit_SERAC_Blip2OPT_Caption():
    hparam_path = './hparams/SERAC/blip2-serac-caption.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/caption_eval_edit/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
    
    
def train_SERAC_Blip2OPT_Caption_eval():
    hparams = SERACMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/blip2-cap-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_SERAC_Blip2OPT_Caption_eval"
    )
    
    trainer.run()


def edit_SERAC_Blip2OPT_VQA():
    hparam_path = './hparams/SERAC/blip2-serac-vqa.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/vqa_eval/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)

def train_SERAC_Blip2OPT_VQA_eval():
    hparams = SERACMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/blip2-vqa-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_SERAC_Blip2OPT_VQA"
    )
    
    trainer.run()

def edit_SERAC_MiniGPT4_Caption():
    hparam_path = './hparams/SERAC/minigpt4-serac-caption.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/caption_eval_edit/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


def train_SERAC_MiniGPT4_Caption_eval():
    hparams = SERACMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/minigpt4-cap-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_SERAC_MiniGPT4_Caption_eval"
    )
    
    trainer.run()


    
def edit_SERAC_MiniGPT4_VQA():
    hparam_path = './hparams/SERAC/minigpt4-serac-vqa.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
    )
    parse_result(metrics, f'./logs/baseline/vqa_eval/{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
 

def train_SERAC_MiniGPT4_VQA_eval():
    hparams = SERACMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/minigpt4-vqa-eval.yaml')
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    eval_ds = VQADataset('./data/vqa/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds,
        train_func="train_SERAC_MiniGPT4_VQA_eval"
    )
    
    trainer.run()


def edit_SERAC_MiniGPT4_Mixed():
    hparam_path = './hparams/SERAC/minigpt4-serac-mixed.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = 9
    # hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
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


def edit_SERAC_Blip2OPT_Mixed():
    hparam_path = './hparams/SERAC/blip2-serac-mixed.yaml'
    hparams = SERACMultimodalHparams.from_hparams(hparam_path)
    hparams.continuous_sample = 9
    # hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
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
        
if __name__ == "__main__":
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Model: {args.model}")
    LOG.info(f"Continuous Sample: {args.continuous_sample}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.task == 'caption':
        if args.model == 'minigpt4':
            edit_SERAC_MiniGPT4_Caption()
        elif args.model == 'blip2':
            train_SERAC_Blip2OPT_Caption_eval()
    elif args.task == 'vqa':
        if args.model == 'minigpt4':
            edit_SERAC_MiniGPT4_VQA()
        elif args.model == 'blip2':
            train_SERAC_Blip2OPT_VQA_eval()
    elif args.task == 'all':
        if args.model == 'minigpt4':
            edit_SERAC_MiniGPT4_Caption()
            edit_SERAC_MiniGPT4_VQA()
        elif args.model == 'blip2':
            edit_SERAC_Blip2OPT_Caption()
            edit_SERAC_Blip2OPT_VQA()
    elif args.task == 'mixed':
        if args.model == 'minigpt4':
            edit_SERAC_MiniGPT4_Mixed()
        elif args.model == 'blip2':
            edit_SERAC_Blip2OPT_Mixed()
