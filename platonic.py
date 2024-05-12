"""
desired use case

import platonic

platonic_metric = platonic.Alignment(
                    dataset="wit", 
                    subset="prh1024", 
                    )   

# (something that most likely wont change during training)
texts = platonic_metric.load(modality="text")


# >>>> inside training loop <<<< #

inputs = tokenizer(texts)
features = model(inputs, extract_features=True)

# compare against vision
score = platonic.score(features, metric="cknna", topk=5, model="dinov2")



# what if i want to use custom alignmnet?
platonic_metric = platonic.Alignment(dataset="custom")

"""

"""
[ ] create dataset class in huggingface
[ ] upload features into huggingface
"""


import os 
import torch

from datasets import load_dataset
from measure_alignment import compute_score, prepare_features


SUPPORTED_DATASETS = {
    "wit_1024": {
        "dinov2": "./results/features/minhuh/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-none.pt",
        "clip": "./results/features/minhuh/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-none.pt",
        }
    }


class Alignment():

    
    def __init__(self, dataset, subset, models=[], device="cuda", dtype=torch.bfloat16):
        """ loads the dataset from subset """
        
        if dataset != "minhuh/prh":
            # TODO: support external datasets in the future
            raise ValueError(f"dataset {dataset} not supported")
            
        if subset not in SUPPORTED_DATASETS:
            raise ValueError(f"subset {subset} not supported for dataset {dataset}")
        
        self.models = models
        self.device = device
        self.dtype = dtype

        # loads the features from path if it does not exist it will download
        self.features = {}
        for m in models:
            feat_path = SUPPORTED_DATASETS[subset][m]
            if not os.path.exists(feat_path):
                raise ValueError(f"feature path {feat_path} does not exist for {m} in {dataset}/{subset}")

            self.features[m] = self.load_features(feat_path)
            
        # download dataset from huggingface        
        self.dataset = load_dataset(dataset, revision=subset, split='train')
        return
    

    def load_features(self, feat_path):
        """ loads features for a model """
        return torch.load(feat_path, map_location=self.device)["feats"].to(dtype=self.dtype)

    
    def get_data(self, modality):
        """ loads text data """
        if modality == "text": # list of strings
            return [x['text'][0] for x in self.dataset]
        elif modality == "image": # list of PIL images
            return [x['image'] for x in self.dataset]

    
    def score(self, features, metric, *args, **kwargs):
        """ 
        Scores the features 
        """
        scores = {}
        for m in self.models:
            scores[m] = compute_score(
                prepare_features(features.to(device=self.device, dtype=self.dtype)), 
                prepare_features(self.features[m].to(device=self.device, dtype=self.dtype)),
                metric, 
                *args, 
                **kwargs
            )
        return scores        
    
    