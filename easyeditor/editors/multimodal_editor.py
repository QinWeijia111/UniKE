from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from PIL import Image
import gc
import copy
import pickle
from datetime import datetime
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

def load_object(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_object(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
def check_nan(model):
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter: {name}")
            has_nan = True
    return has_nan

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor:
    """Multimodal editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT
                
                model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint
                )  
            elif hparams.model_name == "minigpt4":
                from ..trainer.blip2_models import MiniGPT4
                
                model = MiniGPT4(
                    vit_model="eva_clip_g",
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    llama_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                )                
            self.model = model

                
            # Get tokenizer and vis_processor
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)

            # from PIL import Image
            # raw_image = Image.open("../val2014/COCO_val2014_000000235522.jpg").convert("RGB")
            # image = vis_processor(raw_image).unsqueeze(0).to('cuda')
            # model.to('cuda')
            # model.generate({"image": image})

            self.vis_tok = vis_processor
            if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                tok_name = (
                    hparams.tokenizer_name
                    if hparams.tokenizer_name is not None
                    else hparams.name
                )
                tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                    tok_name
                )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token    
                self.tok = tokenizer                         
        else:
            self.model, self.tok = self.model_name
        # device_map = {
        #     0: [_ for _ in range(0, 16)],
        #     1: [_ for _ in range(16, 32)],
        #     2: [_ for _ in range(32, 48)]
        # }
        # self.model.parallelize(device_map=device_map)
        self.model.to(f'cuda:{hparams.device}')
        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image
        if self.alg_name == 'TPATCHER':
            from ..models.unike.src import Editor
            self.editor = Editor(
                            model=model,
                            max_add_neuron_num=hparams.max_add_neuron_num,
                            freeze_model=hparams.freeze_model, freeze_k=hparams.freeze_k, freeze_a=hparams.freeze_a,
                            memory_size=hparams.memory_size, memory_loss=hparams.memory_loss,
                            amplify_v=hparams.amplify_v, activate_loss=hparams.activate_loss,
                            act_margin_val=hparams.act_margin_val, margin_val1=hparams.margin_val1,
                            margin_val2=hparams.margin_val2, device=model.device,
                            hparams=hparams,
                        )
            self.editor.set_latent_ike('./results/l-ike')
            
            

    def edit(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            keep_original_weight=False,
            verbose=True,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")
        all_metrics = []
        for i, request in enumerate(tqdm(requests)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = metrics['pre']['locality_output'].to(torch.float32)
                    post_logits = metrics['post']['locality_output'].to(torch.float32)
                    if post_logits.shape[1] > base_logits.shape[1]:
                        post_logits = post_logits[:, -base_logits.shape[1]:, :]
                    else:
                        base_logits = base_logits[:, -post_logits.shape[1]:, :]

                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                    post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                    if post_image_logits.shape[1] > base_image_logits.shape[1]:
                        post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                    else:
                        base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
            else:
                if self.alg_name == 'TPATCHER':
                    self.editor.restore_edit()
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=True,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    editor=self.editor
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                if self.alg_name == 'KN':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                metrics["pre"] = compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)

            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        return all_metrics, edited_model, weights_copy
    
    # edit_demo will return the logits after/before editing
    def edit_demo(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            keep_original_weight=False,
            verbose=True,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')


        all_metrics = []
        for i, request in enumerate(requests):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = metrics['pre']['locality_output'].to(torch.float32)
                    post_logits = metrics['post']['locality_output'].to(torch.float32)
                    if post_logits.shape[1] > base_logits.shape[1]:
                        post_logits = post_logits[:, -base_logits.shape[1]:, :]
                    else:
                        base_logits = base_logits[:, -post_logits.shape[1]:, :]

                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                    post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                    if post_image_logits.shape[1] > base_image_logits.shape[1]:
                        post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                    else:
                        base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                post, post_logits = compute_multimodal_edit_results_demo(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": post
                }
                if self.alg_name == 'KN':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                pre, pre_logits = compute_multimodal_edit_results_demo(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
                metrics["pre"] = pre
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)

            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        return all_metrics, edited_model, weights_copy, post_logits, pre_logits 

    
    def backup_layers(self):
        if self.hparams.alg_name == 'TPATCHER':
            pass
        elif self.hparams.alg_name == 'FT_MULTI':
            weights = {
                n: p
                for n, p in self.model.named_parameters()
                for layer in self.hparams.layers
                if self.hparams.layer_module_tmp.format(layer) in n
            }
        elif self.hparams.alg_name == 'MEND':
            pass
        elif self.hparams.alg_name == 'SERAC_MULTI':
            pass
        
        return {k: v.detach().clone() for k, v in weights.items()}
    
    
    def check_editors_nan(self):
        for i in self.editor.editors:
            if check_nan(i['original_module']):
                print(f"{i['name']} has nan")
                
    def edit_dataset(self,
                     ds: Union[CaptionDataset, VQADataset, CapVQADataset],
                     keep_original_weight=True,
                     verbose=True,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')
        task=kwargs.get('task', None)
        num_edits = 1
        self.model_backup = copy.deepcopy(self.model.cpu())
        self.model.cuda()
        all_metrics = []
        reload_weights = True
        local_counter = 0
        load_metrics_path = kwargs.get('load_metrics_path', None)
        if load_metrics_path is not None:
            all_metrics = load_object(load_metrics_path)
            local_counter = len(all_metrics)
            LOG.info(f"Loaded metrics from {load_metrics_path}")
        
        # compute the pre-edit results
        pres = []
        cached_path = f'./results/cache/{self.hparams.model_name}_{task}_{len(ds)}.pkl' # model-dataset-specific
        if os.path.exists(cached_path):
            pres = load_object(cached_path)
            LOG.info(f"Load pre results from cached path: {cached_path}")
        else:
            for i, request in tqdm(enumerate(ds), desc='Results before editing', total=len(ds)):
                pre, pre_logits = compute_multimodal_edit_results_demo(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
                pres.append(pre)
            save_object(pres, cached_path)
            
        self.model.zero_grad() # clear the gradient to ensure
        for i, request in tqdm(enumerate(ds), desc='Editing dataset', total=len(ds)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = metrics['pre']['locality_output'].to(torch.float32)
                    post_logits = metrics['post']['locality_output'].to(torch.float32)
                    if post_logits.shape[1] > base_logits.shape[1]:
                        post_logits = post_logits[:, -base_logits.shape[1]:, :]
                    else:
                        base_logits = base_logits[:, -post_logits.shape[1]:, :]

                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    metrics['post'].pop('locality_output_ids')
                    metrics['pre'].pop('locality_output_ids')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                    post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                    if post_image_logits.shape[1] > base_image_logits.shape[1]:
                        post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                    else:
                        base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                    metrics['post'].pop('multimodal_locality_output_ids')
                    metrics['pre'].pop('multimodal_locality_output_ids')

                LOG.info(f"Evaluation took {time() - start}")
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
            else:
                torch.cuda.empty_cache()
                self.model.to(f'cuda:{self.hparams.device}')
                pre = pres[i]
                inner_res = {}
                torch.cuda.empty_cache()
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    editor=self.editor if self.alg_name == 'TPATCHER' else None,
                    collate_fn=ds.collate_fn,
                    pre=pre,
                    inner_res=inner_res,
                    global_iter=i,
                    task=task,
                    reload_weights=reload_weights
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                # self.model = edited_model
                start = time()
                if self.alg_name == 'TPATCHER' and self.hparams.ike == True:
                    ike_method = ALG_MULTIMODAL_DICT['IKE']
                    icl_examples = ike_method(
                        self.model,
                        self.tok,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds']
                    )
                    exec_time = time() - start
                    LOG.info(f"Execution {i} editing took {exec_time}")
                    start = time()
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request, self.hparams.device),
                    }
                else:
                    post, post_logits = compute_multimodal_edit_results_demo(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
                    metrics = {
                        'case_id': i,
                        "time": exec_time,
                        "post": post,
                    }
                if i == 0:
                    self.weights_copy = weights_copy
                
                # if do not use continuous edit, restore the edit layers
                local_counter += 1
                if local_counter % self.hparams.continuous_sample == 0:
                    local_counter = 0 # restore the counter
                    reload_weights = True
                else:
                    reload_weights = False
                torch.cuda.empty_cache()
                        
                if self.alg_name == 'TPATCHER':
                    if reload_weights:
                        self.editor.clear_editors()
                        self.editor.clean_cache()
                    # add additional metrics
                    metrics["add_neuron_num"] = self.editor.add_neuron_num
                    metrics["inner_res"] = inner_res["res"]
                elif self.alg_name in ['KN']:
                    with torch.no_grad():
                        if reload_weights:
                            # weights_copy() # unpatch_fn
                            self.model.load_state_dict(self.model_backup.state_dict())
                            self.model.cuda()
                        else:
                            self.model.load_state_dict(edited_model.state_dict())
                            edited_model = edited_model.cpu()
                            del edited_model
                            self.model.cuda()
                    torch.cuda.empty_cache()
                else:
                    with torch.no_grad():
                        if reload_weights:
                            for k, v in self.weights_copy.items():
                                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                        else:
                            if self.hparams.alg_name == 'FT_MULTI':
                                for k, v in self.weights_copy.items():
                                    # copy the old weights to new model
                                    nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model, k).to(f"cuda:{self.hparams.device}")
                            else:
                                for k, v in self.weights_copy.items():
                                    # copy the old weights to new model
                                    nethook.get_parameter(self.model, k)[...] = nethook.get_parameter(edited_model.model, k).to(f"cuda:{self.hparams.device}")
                            torch.cuda.empty_cache()
                metrics["pre"] = pre
                # calculate the locality accuracy
                if self.alg_name == 'TPATCHER':
                    if 'locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['locality_output']) == \
                                len(metrics['pre']['locality_output'])
                        metrics['inner_res']['locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['locality_output'],
                                                metrics['pre']['locality_output']))
                        metrics['inner_res'].pop('locality_output')
                        
                    if 'multimodal_locality_output' in metrics['inner_res'].keys():
                        assert len(metrics['inner_res']['multimodal_locality_output']) == \
                                len(metrics['pre']['multimodal_locality_output'])
                        metrics['inner_res']['multimodal_locality_acc'] = \
                            np.mean(np.equal(metrics['inner_res']['multimodal_locality_output'],
                                                metrics['pre']['multimodal_locality_output']))
                        metrics['inner_res'].pop('multimodal_locality_output')
                if self.alg_name == 'TPATCHER' and self.hparams.ike == True:
                    metrics['post']['locality_output'] = metrics['post']['locality_output_ids']
                    metrics['post']['multimodal_locality_output'] = metrics['post']['multimodal_locality_output_ids']
                    metrics['post'].pop('locality_output_ids')
                    metrics['post'].pop('multimodal_locality_output_ids')

                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    metrics['post']['locality_acc'] = \
                        np.mean(np.equal(metrics['post']['locality_output'],
                                            metrics['pre']['locality_output']))
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    metrics['post']['multimodal_locality_acc'] = \
                        np.mean(np.equal(metrics['post']['multimodal_locality_output'],
                                            metrics['pre']['multimodal_locality_output']))
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')
                    
                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        return all_metrics, None, weights_copy

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) for image_ in image]
        image = [Image.open(ip).convert("RGB") for ip in image_path]
        image = [self.vis_tok(i).to(self.hparams.device) for i in image]
        
        requests = [{
            'prompt': prompt,
            'target': target,
            'image': image_,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]
        
        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_inputs['text']['prompt']) == len(locality_inputs['text']['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(locality_inputs['vision']['prompt']) == len(locality_inputs['vision']['ground_truth']) \
                == len(locality_inputs['vision']['image']) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            rephrase_image = [self.vis_tok(i).to(self.hparams.device) for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            
            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
            locality_image = [self.vis_tok(i).to(self.hparams.device) for i in locality_image]
             
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
            
        return requests


# if __name__ == "__main__":
#
#     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
#
#     editor.edit(
#         prompts='What university did Watts Humphrey attend?',
#         ground_truth='Illinois Institute of Technology',
#         target_new='University of Michigan'
#     )
#
#     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')


