import platonic
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint

platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is subset 
                    subset="wit_1024",  # <---- this sub-sub set
                    models=["dinov2", "clip"],
                    )   


if True:

    # load images
    texts = platonic_metric.get_data(modality="text")
    batch_size = 32

    model_name = "openlm-research/open_llama_7b"
    language_model = load_llm(model_name, qlora=False)
    device = next(language_model.parameters()).device
    tokenizer = load_tokenizer(model_name)

    tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
    
    llm_feats = []
    for i in trange(0, len(texts), batch_size):
        token_inputs = {k: v[i:i+batch_size].to(device).long() for (k, v) in tokens.items()}
        with torch.no_grad():
            llm_output = language_model(
                input_ids=token_inputs["input_ids"],
                attention_mask=token_inputs["attention_mask"],
            )
        feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3).cpu()
        mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1).cpu()
        feats = (feats * mask).sum(2) / mask.sum(2)
        llm_feats.append(feats)
        
    llm_feats = torch.cat(llm_feats)
    score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)
    pprint(score)
else:
    pass 
    # load images
    images = platonic_metric.get_data(modality="image")
    inputs = tokenizer(texts)



    features = model(inputs, extract_features=True)

    # compare against vision
    score = platonic.score(features, metric="cknna", topk=5, model="dinov2")

    # >>> print(score)
    # {dinov2: 0.48, "clip": 0.32}

    # what if i want to use custom alignmnet?
    platonic_metric = platonic.Alignment(dataset="custom")