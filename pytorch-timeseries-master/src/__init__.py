from .trainer import BaseTrainer
from .ucr import UCRTrainer, load_ucr_trainer
from .har import HARTrainer, load_har_trainer
from .pamap2 import PAMAPTrainer, load_pamap_trainer
from .opportunity import OPPTrainer, load_opp_trainer
from .mhealth import MHEALTHTrainer, load_mhealth_trainer
from .wisdm import WISDMTrainer, load_wisdm_trainer
from .wHAR import wHARTrainer, load_whar_trainer
__all__ = ['BaseTrainer', 'UCRTrainer', 'load_ucr_trainer', 'HARTrainer', 'load_har_trainer', 'OPPTrainer', 'load_opp_trainer', 'PAMAPTrainer', 'load_pamap_trainer',
           'MHEALTHTrainer', 'load_mhealth_trainer', 'WISDMTrainer', 'load_wisdm_trainer', 'wHARTrainer', 'load_whar_trainer']
