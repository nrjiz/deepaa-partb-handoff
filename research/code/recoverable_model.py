"""Prospective Route B sampling; protected model source remains unchanged."""
from .model_adapter import load_baseline_module
from .constants import IMAGE_SIZE,LOSS_WEIGHTS
VARIANT='k7_philox_recovery_candidate_v1'
def build_model(repository,k):
    if k!=7:raise ValueError('Route B candidate is K7 only')
    import tensorflow as tf
    base=load_baseline_module(repository)
    class RecoverableDeepAA(base.DeepAA):
        def __init__(self,cfg):
            super().__init__(cfg)
            self.sampling_rng=tf.random.Generator.from_seed(113,alg='philox')
        def reparam(self,mu,sigma,training):
            if not training:return mu
            return mu+sigma*self.sampling_rng.normal(tf.shape(mu),dtype=mu.dtype)
    cfg=base.DAAConfig(img_size=IMAGE_SIZE,latent_dim=6,num_classes=7,recon_w=LOSS_WEIGHTS['reconstruction'],class_w=LOSS_WEIGHTS['classification'],kl_w=LOSS_WEIGHTS['kl'],at_w=LOSS_WEIGHTS['archetype_simplex'])
    return base,RecoverableDeepAA(cfg),cfg
