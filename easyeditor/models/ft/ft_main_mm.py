from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import gc
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams
from ...evaluate import prepare_multimodal_edit, compute_multimodal_edit_results_demo
from ...trainer.utils import dict_to, cu_del
from ...trainer import kl_loc_loss
import gc


def construct_mm_samples(
    model,
    model_name,
    hparams: FTHyperParams,
    tok: AutoTokenizer,
    record: Dict,
    device
) -> Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    device = model.device
    # edit_inner = dict_to(edit_inner, device)
    ret['rewrite_sample'] = edit_inner
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        # edit_outer = dict_to(edit_outer, device)
        ret['rephrase_sample'] = edit_outer
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        # edit_image_outer = dict_to(edit_image_outer, device)
        ret['image_rephrase_sample'] = edit_image_outer

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        # locality = dict_to(locality, device)
        ret['locality_output_sample'] = locality
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        # m_locality = dict_to(m_locality, device)
        ret['multimodal_locality_output_sample'] = m_locality

    torch.cuda.empty_cache()
    gc.collect()
    
    return ret


def apply_ft_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    weights_copy = execute_ft(model, tok, requests, hparams)

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits

def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:

    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    from ...trainer.losses import masked_log_probs
    
    # construct 5 samples for multimodal edit
    samples = construct_mm_samples(model, hparams.model_name, hparams, tok, requests[0], model.device)
    batch_samples = dict_to(samples, model.device) # send to device
    
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.layer_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    # print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.SGD(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    batch = batch_samples["rewrite_sample"]
    # print("successfully create optimizer and ready to start training")
    for e in range(hparams.num_steps):
        outputs = _logits(model(batch))
        loss = masked_log_probs(hparams, outputs, batch["labels"], shift=True)["nll"]
        # print(f"FT Epoch [{e+1}/{hparams.num_steps}], Loss: {loss.item()}")
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Restore state of original model
    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k]

    cu_del(batch)
    del batch
    torch.cuda.empty_cache()
    return weights_copy

