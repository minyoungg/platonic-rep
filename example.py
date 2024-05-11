import platonic

platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is subset 
                    subset="wit_1024",  # <---- this sub-sub set
                    models=["dinov2", "clip"],
                    )   


# load images
texts = platonic_metric.get_data(modality="text")

# load images
images = platonic_metric.get_data(modality="image")
assert False, 'hi'


# >>>> inside training loop <<<< #

inputs = tokenizer(texts)
features = model(inputs, extract_features=True)

# compare against vision
score = platonic.score(features, metric="cknna", topk=5, model="dinov2")

# >>> print(score)
# {dinov2: 0.48, "clip": 0.32}

# what if i want to use custom alignmnet?
platonic_metric = platonic.Alignment(dataset="custom")