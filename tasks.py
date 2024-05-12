

def get_models(modelset, modality='all'):
    
    assert modality in ['all', 'vision', 'language']
    
    if modelset == 'val':
        llm_models = [
            # "bigscience/bloomz-560m",
            # "bigscience/bloomz-1b1",
            # "bigscience/bloomz-1b7",
            # "bigscience/bloomz-3b",
            # "bigscience/bloomz-7b1",
            # "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            # "openlm-research/open_llama_13b",
            # "huggyllama/llama-7b",
            # "huggyllama/llama-13b",
            # "huggyllama/llama-30b",
            # "huggyllama/llama-65b",
        ]

        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k",
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",
        ]
        
    elif modelset == 'test':
        llm_models = [
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            "mistralai/Mixtral-8x22B-v0.1",
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
        ]
        
        lvm_models = []
        
    elif modelset == 'custom':
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            # "mistralai/Mixtral-8x22B-v0.1", # was too big so did not use
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
        ]
        lvm_models = [
            "vit_giant_patch14_dinov2.lvd142m",
        ]
    else:
        raise ValueError(f"Unknown modelset: {modelset}")
    
    if modality == "vision":
        llm_models = []
    elif modality == "language":
        lvm_models = []

    return llm_models, lvm_models
