import platonic
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint


# setup platonic metric
platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is the dataset 
                    subset="wit_1024",    # <--- this is the subset
                    models=["dinov2_g", "clip_h"],
                    ) # you can also pass in device and dtype as arguments

# load texts
texts = platonic_metric.get_data(modality="text")

# your model (e.g. we will use open_llama_7b as an example)
model_name = "openlm-research/open_llama_7b"
language_model = load_llm(model_name, qlora=False)
device = next(language_model.parameters()).device
tokenizer = load_tokenizer(model_name)

# extract features
tokens = tokenizer(texts, padding="longest", return_tensors="pt")        

llm_feats = []
batch_size = 16

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
    # import ipdb; ipdb.set_trace()
    
llm_feats = torch.cat(llm_feats)

# compute score
score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)
pprint(score) # it will print the score and the index of the layer the maximal alignment happened
