from typing import *
import os
import argparse
parser = argparse.ArgumentParser(description='train args')
parser.add_argument('--task', type=str, metavar='TASK', choices=['all', 'caption', 'vqa', 'mixed'], default='vqa', help='Task type')
parser.add_argument('--hparams_dir', type=str, default='./hparams/IKE/minigpt4.yaml')
parser.add_argument('--model', type=str, metavar='MODEL', choices=['minigpt4', 'blip2'], default='blip2', help='Model type')
parser.add_argument('--continuous_sample', type=int, default=1, help='CONTINUOUS SAMPLE')
parser.add_argument('--k', type=int, default=32, help='num K in retrieval')
parser.add_argument('--gpu_id', type=str, default="2")
parser.add_argument('--prob_use_result', type=float, default=0.3)
parser.add_argument('--use_sample', type=bool, default=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

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

USE_SAMPLE: bool = args.use_sample


def load_cached_dataset(cached_dir: bool, path: str, dsClass: BaseDataset, config: HyperParams) -> BaseDataset:
    # if the path is a file, then load it
    if os.path.isfile(cached_dir):
        with open(cached_dir, 'rb') as f:
            return pickle.load(f)
    # if the path is dir, raise error
    elif os.path.isdir(cached_dir):
        raise ValueError(f'{cached_dir} is a directory, not a file')
    # if the path is not exist, load using default method and save it
    else:
        ds = dsClass(path, config=config)
        with open(cached_dir, 'wb') as f:
            pickle.dump(ds, f)
        return ds


def Generate_Embedding_for_IKE(hparams='./hparams/IKE/blip2.yaml', ds='vqa'):
    
    hparams = IKEMultimodalHyperParams.from_hparams(hparams)
    hparams.task_name = ds
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    if 'vqa' == ds:
        train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    elif 'caption' == ds:
        train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    elif 'all' == ds:
        hparams.task_name = 'vqa'
        train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
        encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
        
        hparams.task_name = 'caption'
        train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
        encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
        return
        
    ## Generate embedding files for IKE
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
    

 
def test_IKE_Blip2OPT_Caption():
    hparam_path = './hparams/IKE/blip2.yaml'
    hparams = IKEMultimodalHyperParams.from_hparams('./hparams/IKE/blip2.yaml')
    hparams.task_name = args.task
    hparams.k = args.k
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams)
    if USE_SAMPLE:
        file_path = './data/caption/caption_eval_edit_sample.json'
    else:
        file_path = './data/caption/caption_eval_edit.json'
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams, size=100)
    eval_ds = CaptionDataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True,
        task='caption'
    )
    
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


def test_IKE_Blip2OPT_VQA():
    hparam_path = './hparams/IKE/blip2.yaml'
    hparams = IKEMultimodalHyperParams.from_hparams(hparam_path)
    hparams.task_name = args.task
    hparams.k = args.k
    hparams.prob_use_result = args.prob_use_result
    editor = MultimodalEditor.from_hparams(hparams)
    if USE_SAMPLE:
        file_path = './data/vqa/vqa_eval_sample.json'
    else:
        file_path = './data/vqa/vqa_eval.json'
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True,
        task='vqa'
    )
    
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


    
def test_IKE_MiniGPT4_Caption():
    hparam_path = './hparams/IKE/minigpt4.yaml'
    hparams = IKEMultimodalHyperParams.from_hparams(hparam_path)
    hparams.k = args.k
    hparams.task_name = args.task
    hparams.prob_use_result = args.prob_use_result
    editor = MultimodalEditor.from_hparams(hparams)
    if USE_SAMPLE:
        file_path = './data/caption/caption_eval_edit_sample.json'
    else:
        file_path = './data/caption/caption_eval_edit.json'
    train_ds = CaptionDataset('./data/caption/caption_train_edit.json', config=hparams, size=100)
    eval_ds = CaptionDataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True,
        task='caption' 
    )
    
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)

    
    
def test_IKE_MiniGPT4_VQA():
    hparam_path = './hparams/IKE/minigpt4.yaml'
    hparams = IKEMultimodalHyperParams.from_hparams(hparam_path)
    hparams.k = args.k
    hparams.task_name = args.task
    hparams.prob_use_result = args.prob_use_result
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams, size=5)
    if USE_SAMPLE:
        file_path = './data/vqa/vqa_eval_sample.json'
    else:
        file_path = './data/vqa/vqa_eval.json'
    train_ds = VQADataset('./data/vqa/vqa_train.json', config=hparams)
    eval_ds = VQADataset(file_path, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True,
        task='vqa'
    )
    
    parse_result(metrics, f'./logs/{get_filename(file_path)}/{get_filename(file_path)}_{get_filename(hparam_path)}_{get_date()}.json', config=hparams)


        
if __name__ == "__main__":
    LOG.info(f"Task: {args.task}")
    LOG.info(f"Model: {args.model}")
    LOG.info(f"Continuous Sample: {args.continuous_sample}")
    LOG.info(f"Running on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # Generate_Embedding_for_IKE(args.hparams_dir, args.task)
    # exit(0)
    if args.task == 'caption':
        if args.model == 'minigpt4':
            test_IKE_MiniGPT4_Caption()
        elif args.model == 'blip2':
            test_IKE_Blip2OPT_Caption()
    elif args.task == 'vqa':
        if args.model == 'minigpt4':
            test_IKE_MiniGPT4_VQA()
        elif args.model == 'blip2':
            test_IKE_Blip2OPT_VQA()
    elif args.task == 'all':
        if args.model == 'minigpt4':
            test_IKE_MiniGPT4_Caption()
            test_IKE_MiniGPT4_VQA()
        elif args.model == 'blip2':
            test_IKE_Blip2OPT_Caption()
            test_IKE_Blip2OPT_VQA()
