import platonic
from tqdm.auto import trange
import torch 
from pprint import pprint

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor


# setup platonic metric
platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is the dataset 
                    subset="wit_1024",    # <--- this is the subset
                    models=["openllama_7b", "llama_65b"], 
                    ) # you can also pass in device and dtype as arguments

# load images
images = platonic_metric.get_data(modality="image")

# your model (e.g. we will use dinov2 as an example)
model_name = "vit_giant_patch14_dinov2.lvd142m"
vision_model = timm.create_model(model_name, pretrained=True).cuda().eval()

transform = create_transform(
    **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
)

# extract features
return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

lvm_feats = []
batch_size = 32


for i in trange(0, len(images), batch_size):
    ims = torch.stack([transform(images[j]) for j in range(i,i+batch_size)]).cuda()

    with torch.no_grad():
        lvm_output = vision_model(ims)

    feats = torch.stack([v[:, 0, :] for v in lvm_output.values()]).permute(1, 0, 2)
    lvm_feats.append(feats)
    
# compute score 
lvm_feats = torch.cat(lvm_feats)
score = platonic_metric.score(lvm_feats, metric="mutual_knn", topk=10, normalize=True)
pprint(score) # it will print the score and the index of the layer the maximal alignment happened