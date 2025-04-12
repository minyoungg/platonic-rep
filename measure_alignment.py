import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
import utils
from pprint import pprint



def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats.cuda()
    elif isinstance(feats, list):
        return [metrics.remove_outliers(f.float(), q=q, exact=exact).cuda() for f in feats]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    best_alignment_indices = None
    best_alignment_score = 0

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk

            score = metrics.AlignmentMetrics.measure(metric, x_aligned, y_aligned, **kwargs)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    return best_alignment_score, best_alignment_indices

    
def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
        if isinstance(raw_x, torch.Tensor):
            x_feats = prepare_features(raw_x.float(), exact=precise)
        else:
            x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
        
        # x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           

            raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
            if isinstance(raw_y, torch.Tensor):
                y_feats = prepare_features(raw_y.float(), exact=precise)
            else:
                y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]
            best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk)
            
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)

    parser.add_argument("--input_dir",      type=str, default="./results/features")
    parser.add_argument("--output_dir",     type=str, default="./results/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.topk
    )
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    models_x = llm_models if args.modality_x == "language" else lvm_models
    models_y = llm_models if args.modality_y == "language" else lvm_models
    
    models_x_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_x, args.prompt_x) for m in models_x]
    models_y_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_y, args.prompt_y) for m in models_y]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{args.topk}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    alignment_scores, alignment_indices = compute_alignment(models_x_paths, models_y_paths, args.metric, args.topk, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices})
    print(f"saved to {save_path}")
    