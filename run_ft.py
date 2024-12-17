import os
import argparse
parser = argparse.ArgumentParser(description='train args')
parser.add_argument('--task', type=str, metavar='TASK', choices=['all', 'caption', 'vqa', 'mixed'], default='mixed', help='Task type')
parser.add_argument('--model', type=str, metavar='MODEL', choices=['minigpt4', 'blip2'], default='minigpt4', help='Model type')
parser.add_argument('--continuous_sample', type=int, default=1, help='CONTINUOUS SAMPLE')
parser.add_argument('--gpu_id', type=str, default="7")
parser.add_argument('--ablation', type=bool, default=False)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--lr', type=float, default=7e-4)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
minigpt4_config = './hparams/FT/minigpt4.yaml'
blip2_config = './hparams/FT/blip2.yaml'
from typing import *
import torch
from torch.utils.data import Dataset, ConcatDataset
import types
from itertools import chain
from dataclasses import asdict
from statistics import mean
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, CapVQADataset
from easyeditor.dataset.processor.base_dataset import BaseDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, UniKEHyperParams, FTHyperParams
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


USE_SAMPLE: bool = True

MANUAL_CONTINUOUS_SAMPLE = args.continuous_sample


def edit_FT_Caption(hparam_path):
    hparams = FTHyperParams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    hparams.num_steps = args.num_steps
    if args.lr is not None:
        hparams.lr = args.lr
    if USE_SAMPLE:
        file_path = './data/caption/caption_eval_edit_sample.json'
    else:
        file_path = './data/caption/caption_eval_edit.json'
    # train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    # eval_ds = CaptionDataset('./data/caption/caption_eval_edit.json', config=hparams)
    eval_ds = CaptionDataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='caption'
    )
    # parse_result(metrics, f'./logs/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)

def edit_FT_VQA(hparam_path):
    hparams = FTHyperParams.from_hparams(hparam_path)
    hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    hparams.num_steps = args.num_steps
    if args.lr is not None:
        hparams.lr = args.lr
    if USE_SAMPLE:
        file_path = './data/vqa/vqa_eval_sample.json'
    else:
        file_path = './data/vqa/vqa_eval.json'
    eval_ds = VQADataset(file_path, config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True,
        task='vqa' 
    )
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)



def edit_FT_Mixed(hparam_path):
    hparams = FTHyperParams.from_hparams(hparam_path)
    hparams.continuous_sample = 9
    # hparams.continuous_sample = MANUAL_CONTINUOUS_SAMPLE
    hparams.num_steps = args.num_steps
    if args.lr is not None:
        hparams.lr = args.lr
    eval_ds = CapVQADataset(cap_dir='./data/caption/caption_eval_edit.json', 
                            vqa_dir='./data/vqa/vqa_eval.json', config=hparams) 
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    parse_result(metrics, f'./logs/cross/{get_filename(hparam_path)}_{get_date()}.json', config=hparams, ablation=args.ablation)
    

    
if __name__ == "__main__":
    if args.model == 'minigpt4':
        hparams_dir = minigpt4_config
    elif args.model == 'blip2':
        hparams_dir = blip2_config
    else:
        raise ValueError(f"Model {args.model} not found")
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Hyperparams: {hparams_dir}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    LOG.info(f"Continuous Sample: {args.continuous_sample}")

    if args.task == 'mixed':
        edit_FT_Mixed(hparams_dir)
    elif args.task == 'caption':
        edit_FT_Caption(hparams_dir)
    elif args.task == 'vqa':
        edit_FT_VQA(hparams_dir)
    elif args.task == 'all':
        edit_FT_Caption(hparams_dir)
        edit_FT_VQA(hparams_dir)