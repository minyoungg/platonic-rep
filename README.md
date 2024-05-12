<h1 align="center">The Platonic Representation Hypothesis</h1>

<h3 align="center"><a href="https://arxiv.org" style="color: #E34F26;">paper</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://example.com/project" style="color: #2088FF;">project page</a><br></h3>
<h5 align="center">
<a href="https://example.com/minyoung" style="color: #3178C6;">minyoung huh</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://example.com/brian" style="color: #E34F26;">brian cheung</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://example.com/tongzhou" style="color: #FCC624;">tongzhou wang</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
<a href="https://example.com/phillip" style="color: #4EAA25;">phillip isola</a>* &nbsp&nbsp&nbsp&nbsp&nbsp
</h5>

<hr>

<h3> Requirements </h3>
<br />

`python >= 3.10`
`PyTorch >= 2.0.0`

You can install the rest of the requirements via

```bash
pip install -r requirements.txt
```

<hr>

<h3> Running alignment </h3>
<br />

<b> (1) extracting features</b>

First, we extract features from the models.

```bash
# extract all language model features and pool them along each block
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality language --pool avg

# extract last layer features of all vision models
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality vision --pool none
```

The resulting features are stored in `./results/features` 

<b> (2) measuring vision-language alignment</b>
After extracting the features, you can compute the alignment score by 

```bash
python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
        --modality_x language --pool_x avg --modality_y vision --pool_y none
```

The resulting alignment scores will be stored in `./results/alignment`

```python
>>> fp = './results/alignment/minhuh/prh/val/language_pool-avg_prompt-False_vision_pool-none_prompt-False/mutual_knn_k10.npy'
>>> result = np.load(fp, allow_pickle=True).item()
>>> print(results.keys()
dict_keys(['scores', 'indices'])
>>> print(result['scores'].shape) # 12 language models x 17 vision models
(12, 17)
```

<hr>

<h3> Customization </h3>
<br />

<b> ❔ Can I add additional models? </b><br>

To add your own set of models, add them and correctly modify the files in `tasks.py.` The `llm_models` should be auto-regressive models from huggingface and `lvm_models` should be ViT models from `huggingface/timm`. Most models should work without further modification. Currently, we do not support different vision architectures and language models that are not autoregressive.

<br />
<br />

<b> ❔ What are the metrics that I can use? </b><br>

To check all supported alignment metrics run 
```bash
>>> python -c 'from metrics import alignment; print(alignment.SUPPORTED_METRICS)'
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

