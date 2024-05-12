# NOTE: there are models we did not list, feel free to add more or your custom models
# here is the full list of precomputed features http://vision14.csail.mit.edu/prh/wit_1024/

SUPPORTED_DATASETS = {
    "wit_1024": {
        "dinov2_s": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_m": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_g": {
             "path": "./results/features/minhuh/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-none.pt",
             "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "clip_b": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-none.pt"
        },
        "clip_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-none.pt"
        },
        "clip_h": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-none.pt"   
        },
        "llama-7b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt"
        },
        "llama-13b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt"
        },
        "llama-30b": { # NOTE this is 33B https://huggingface.co/huggyllama/llama-30b
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt"
        },
        "llama_65b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt",   
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt"
        }
    }
}

from platonic.alignment import *