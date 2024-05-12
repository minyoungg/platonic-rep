import os
import numpy as np
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

fd = '/vision-nfs/isola/projects/minhuh/akc/data/facebook_pmd/wit/'
texts = natural_sort([f"{fd}/{f}" for f in os.listdir(fd) if 'npy' in f])
orig_texts = [np.load(f, allow_pickle=True).item()["text"] for f in texts]


import platonic

platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is subset 
                    subset="wit_1024",  # <---- this sub-sub set
                    models=["dinov2", "clip"],
                    )   

texts = platonic_metric.get_data(modality="text")

from IPython import embed; embed()

for t1, t2 in zip(texts, orig_texts):
    print(t1)
    print(t2)
    input('next')