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
    dataset="minhuh/prh",
    subset="wit_1024",
    models=["openllama_7b", "llama_65b"],
)

# load images
images = platonic_metric.get_data(modality="image")

# model setup
model_name = "convnext_tiny"
vision_model = timm.create_model(model_name, pretrained=True).cuda().eval()

transform = create_transform(
    **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
)

# select feature extraction points (e.g., after each stage)
return_nodes = {
    "stages.0.blocks.1": "stage1",
    "stages.1.blocks.1": "stage2",
    "stages.2.blocks.1": "stage3",
    "stages.3.blocks.1": "stage4"
}
vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

# lvm_feats will collect features per batch; each batch contains a list of layer-wise features
lvm_feats = []
batch_size = 32

for i in trange(0, len(images), batch_size):
    ims = torch.stack([transform(images[j]) for j in range(i, i+batch_size)]).cuda()

    with torch.no_grad():
        lvm_output = vision_model(ims)

    # Each tensor has shape (batch, C, H, W) -> we apply Global Average Pooling to get (batch, C)
    # This avoids shape mismatches across layers that may have different spatial sizes (H, W)
    feats = [torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1).cpu() for f in lvm_output.values()]
    lvm_feats.append(feats)

# Transpose list of lists to group features by layer across batches
# Before zip: lvm_feats[batch][layer] -> After zip: lvm_feats[layer][batch]
lvm_feats = list(zip(*lvm_feats))

# Concatenate batch features for each layer
# Each tensor now has shape (total_images, C) for a given layer
lvm_feats = [torch.cat(layer_feats, dim=0) for layer_feats in lvm_feats]

# compute score 
score = platonic_metric.score(lvm_feats, metric="mutual_knn", topk=10, normalize=True)
pprint(score)
