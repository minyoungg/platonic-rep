<h1 align="center">The Platonic Representation Hypothesis</h1>

<h3 align="center"><a href="http://arxiv.org/abs/2405.07987" style="color: #E34F26;">paper</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://phillipi.github.io/prh/" style="color: #2088FF;">project page</a><br></h3>
<h5 align="center">
<a href="https://minyoungg.github.io/me/" style="color: #3178C6;">minyoung huh</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://briancheung.github.io/" style="color: #E34F26;">brian cheung</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://www.tongzhouwang.info/" style="color: #FCC624;">tongzhou wang</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://web.mit.edu/phillipi/" style="color: #4EAA25;">phillip isola</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
</h5>

<hr>

<h3> Requirements </h3>
<br />

Developed on  

`python = 3.11`
`PyTorch = 2.2.0`

You can install the rest of the requirements via

```bash
pip install -r requirements.txt
```

<hr>

<h3> Running alignment </h3>
<br />

<b> (1) Extracting features</b>

First, we extract features from the models.

```bash
# extract all language model features and pool them along each block
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality language --pool avg

# Extract last layer features of all vision models
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality vision --pool cls
```

The resulting features are stored in `./results/features` 

<b> (2) Measuring vision-language alignment</b>

After extracting the features, you can compute the alignment score by 

```bash
python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
        --modality_x language --pool_x avg --modality_y vision --pool_y cls
```

The resulting alignment scores will be stored in `./results/alignment`

```python
>>> fp = './results/alignment/minhuh/prh/val/language_pool-avg_prompt-False_vision_pool-cls_prompt-False/mutual_knn_k10.npy'
>>> result = np.load(fp, allow_pickle=True).item()
>>> print(results.keys()
dict_keys(['scores', 'indices'])
>>> print(result['scores'].shape) # 12 language models x 17 vision models
(12, 17)
```

<hr>
<h3> Scoring your own model for alignment to Platonic Representation Hypothesis </h3>

We provide code to compute alignment scores for your own model while training/evaluating.

<b> (1) Install library as pip package </b>
First install the library as a pip package

```bash
pip install -e .
```

<b> (2) Initiate the metric scoring function </b>

```python
import platonic

# setup platonic metric
platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh",
                    subset="wit_1024", 
                    models=["dinov2_g", "clip_h"],
                    ) # optional arguments device, dtype, save_dir (or path to your features)

# load texts
texts = platonic_metric.get_data(modality="text")
```

We provide some precomputed features, so you don't have to compute it yourself. It will automatically download them for you.
See `SUPPORTED_DATASETS` in `platonic.py`. <b>Note</b>: We will add more in the upcoming weeks.

<b> (3) Extract the features from your model </b> 

```python
# your model has to have `output_hidden_states=True`
with torch.no_grad():
        llm_output = language_model(
            input_ids=token_inputs["input_ids"],
            attention_mask=token_inputs["attention_mask"],
        )
        feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)

# using average pooling (only on valid tokens)
mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
feats = (feats * mask).sum(2) / mask.sum(2)

# compute score. the score is dict for each model where each entry contains the (scores, maximal alignment layer indices)
score = platonic_metric.score(feats, metric="mutual_knn", topk=10, normalize=True)
```

We provide examples for both vision and language in `examples`. You can run them via `python examples/example_language.py`. It will download the features in the local directory if you don't have it computed already.

<hr>

<h3> Customization / Questions </h3>
<br />

<b> ❔ Can I add additional models? </b><br>

To add your own set of models, add them and correctly modify the files in `tasks.py.` The `llm_models` should be auto-regressive models from huggingface and `lvm_models` should be ViT models from `huggingface/timm`. Most models should work without further modification. Currently, we do not support different vision architectures and language models that are not autoregressive.

<br />
<br />

<b> ❔ What are the metrics that I can use? </b><br>

To check all supported alignment metrics run 
```bash
>>> python -c 'from metrics import AlignmentMetrics; print(AlignmentMetrics.SUPPORTED_METRICS)'
['cycle_knn', 'mutual_knn', 'lcs_knn', 'cka', 'unbiased_cka', 'cknna', 'svcca', 'edit_distance_knn']
```
Feel free to add your own in `metrics.py`

<br />
<br />

<b> ❔ I want to use the metrics for my own repo. How do I use it? </b><br>
Simply copy the `metrics.py` file to your repo, and you can use it anywhere. It expects a tensor of shape `[batch x feats]`

```python
from metrics import AlignmentMetrics
import torch.nn.functional as F

feats_A = torch.randn(64, 8192)
feats_B = torch.randn(64, 8192)
feats_A = F.normalize(feats_A, dim=-1)
feats_B = F.normalize(feats_B, dim=-1)

# measure score
score = AlignmentMetrics.measure('cknna', feats_A, feats_B, topk=10)

# alternative
score = AlignmentMetrics.cknna(feats_A, feats_B, topk=10)
```

<br />
<br />

<b> ❔ I want to add my own custom features for `platonic` </b><br>
To add custom models, add it to `SUPPORTED_DATASETS`.


<br />
<br />

<b> ❔ Download URL is down. What do I do? </b><br>
If our download URL is down, please give it some time, as we will try to set it back up as soon as possible.
In the meantime, you can compute the same exact features by running the example code in the `extracting features` section above.

<br />
<br />


<b> ❔ Reporting alignment scores </b><br>

Note that numbers might vary with different precision and batch-size due to hardware/algorithm variabilities.
When evaluating alignment trends, we recommend you to regenerate the features using the same settings when reporting numbers.

<br />

<hr> 

<h3> Citation </h3>
<br />

```bib
@inproceedings{huh2024prh,
  title={The Platonic Representation Hypothesis},
  author={Huh, Minyoung and Cheung, Brian and Wang, Tongzhou, and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

