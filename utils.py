import os
import torch
import math


def to_feature_filename(output_dir, dataset, subset, model_name, pool=None, prompt=None, caption_idx=None):
    save_name = f"{model_name.replace('/', '_')}"

    if pool:
        save_name += f"_pool-{pool}"
    if prompt:
        save_name += f"_prompt-{prompt}"
    if caption_idx:
        save_name += f"_cid-{caption_idx}"
    
    save_path = os.path.join(
        output_dir, 
        dataset, 
        subset,
        f"{save_name}.pt"
    )
    return save_path


def to_alignment_filename(output_dir, dataset, modelset, 
                          modality_x, pool_x, prompt_x, 
                          modality_y, pool_y, prompt_y, 
                          metric, topk):
    save_path = os.path.join(
        output_dir,
        dataset,
        modelset,
        f"{modality_x}_pool-{pool_x}_prompt-{prompt_x}_{modality_y}_pool-{pool_y}_prompt-{prompt_y}",
        f"{metric}_k{topk}.npy" if 'knn' in metric else f"{metric}.npy"
    )
    return save_path
    
    
def cross_entropy_loss(llm_inputs, llm_outputs):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    mask = llm_inputs["attention_mask"][:, :-1]
    loss = mask * criterion(
        llm_outputs["logits"][:, :-1].permute(0, 2, 1),
        llm_inputs["input_ids"][:, 1:],
    )
    avg_loss = (loss.sum(-1) / mask.sum(-1))
    return loss, avg_loss


def cross_entropy_to_bits_per_unit(losses, input_strings, unit="byte"):
    """
    Convert cross-entropy losses from nats to bits per byte for each input string.

    Parameters:
    - losses (torch.Tensor): [batch x seq_len] (padding tokens should be 0)
    - input_strings (list of str): List of original input strings.

    Returns:
    - torch.Tensor: Tensor of bits per byte values, one per input string.
    """
    # nats to bits by multiplying with log base 2 of e (since log_e(2) = 1 / log_2(e))
    # sum over the sequence length (total bits for each input string)
    losses_in_bits = (losses.cpu() * torch.log2(torch.tensor(math.e))).sum(1)

    # calculate bytes for each input string and normalize losses (8 bits per character, so roughly num character * 8)
    if unit == "byte":
        bytes_per_input = torch.tensor([len(s.encode('utf-8')) for s in input_strings], dtype=torch.float32)
    elif unit == "char":
        bytes_per_input = torch.tensor([len(s) for s in input_strings], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported unit {unit}")

    # mormalize by the total number of bytes per input string
    bits_per_byte = losses_in_bits / bytes_per_input
    return bits_per_byte