from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model, apply_ft_to_multimodal_model
from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
from ..dataset import CaptionDataset, VQADataset, CapVQADataset
from ..models.ike import IKEHyperParams, apply_ike_to_model, apply_ike_to_multimodal_model, apply_ike_to_per_model
from ..models.unike import UniKEHyperParams, apply_tp_to_model_mm

ALG_DICT = {
    "FT": apply_ft_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
    'TPATCHER': apply_tp_to_model_mm,
}

ALG_MULTIMODAL_DICT = {
    'MEND': MendMultimodalRewriteExecutor().apply_to_model,
    'SERAC': SeracMultimodalRewriteExecutor().apply_to_model,
    'SERAC_MULTI': SeracMultimodalRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_multimodal_model,
    'TPATCHER': apply_tp_to_model_mm,
    'FT_MULTI': apply_ft_to_multimodal_model,
}


PER_ALG_DICT = {
    "IKE": apply_ike_to_per_model,
}

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
    "mt": CapVQADataset,
}

