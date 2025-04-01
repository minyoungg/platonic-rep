# NOTE: there are models we did not list, feel free to add more or your custom models
# here is the full list of precomputed features http://vision14.csail.mit.edu/prh/wit_1024/

SUPPORTED_DATASETS = {
    "wit_1024": {
        "i21k_t": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_tiny_patch16_224.augreg_in21k-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_tiny_patch16_224.augreg_in21k-cls.pt"
        },
        "i21k_s": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_small_patch16_224.augreg_in21k-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_small_patch16_224.augreg_in21k-cls.pt"
        },
        "i21k_b": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch16_224.augreg_in21k-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch16_224.augreg_in21k-cls.pt"
        },
        "i21k_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch16_224.augreg_in21k-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch16_224.augreg_in21k-cls.pt"
        },
        "dinov2_s": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-cls.pt"
        },
        "dinov2_m": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-cls.pt"
        },
        "dinov2_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-cls.pt"
        },
        "dinov2_g": {
             "path": "./results/features/minhuh/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-cls.pt",
             "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-cls.pt"
        },
        "clip_b": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-cls.pt"
        },
        "clip_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-cls.pt"
        },
        "clip_h": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-cls.pt"   
        },
        "clip_i21k_ft_b": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch16_clip_224.laion2b_ft_in12k_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch16_clip_224.laion2b_ft_in12k_pool-cls.pt"
        },
        "clip_i21k_ft_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_clip_224.laion2b_ft_in12k_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_clip_224.laion2b_ft_in12k_pool-cls.pt"
        },
        "clip_i21k_ft_h": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_ft_in12k_pool-cls.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_ft_in12k_pool-cls.pt"
        },
        "bloom_560m": {
            "path": "./results/features/minhuh/prh/wit_1024/bigscience_bloomz-560m_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/bigscience_bloomz-560m_pool-avg.pt"
        },
        "bloom_1b1": {
            "path": "./results/features/minhuh/prh/wit_1024/bigscience_bloomz-1b1_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/bigscience_bloomz-1b1_pool-avg.pt"
        },
        "bloom_1b7": {
            "path": "./results/features/minhuh/prh/wit_1024/bigscience_bloomz-1b7_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/bigscience_bloomz-1b7_pool-avg.pt"
        },
        "bloom_3b": {
            "path": "./results/features/minhuh/prh/wit_1024/bigscience_bloomz-3b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/bigscience_bloomz-3b_pool-avg.pt"
        },
        "bloom_7b1": {
            "path": "./results/features/minhuh/prh/wit_1024/bigscience_bloomz-7b1_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/bigscience_bloomz-7b1_pool-avg.pt"
        },
        "openllama_3b": {
            "path": "./results/features/minhuh/prh/wit_1024/openlm-research_open_llama_3b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/openlm-research_open_llama_3b_pool-avg.pt"
        },
        "openllama_7b": {
            "path": "./results/features/minhuh/prh/wit_1024/openlm-research_open_llama_7b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/openlm-research_open_llama_7b_pool-avg.pt"
        },
        "openllama_13b": {
            "path": "./results/features/minhuh/prh/wit_1024/openlm-research_open_llama_13b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/openlm-research_open_llama_13b_pool-avg.pt"
        },
        "llama_7b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt"
        },
        "llama_13b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt"
        },
        "llama_30b": { # NOTE this is 33B https://huggingface.co/huggyllama/llama-30b
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt"
        },
        "llama_65b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt",   
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt"
        },
    }
}

from platonic.alignment import *