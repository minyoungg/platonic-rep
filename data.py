import os
import io
import PIL.Image
import urllib
import traceback
import json

import numpy as np
from tqdm.auto import tqdm

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

USER_AGENT = get_datasets_user_agent()


def get_image(pil_data):
    return pil_data.convert("RGB")


def get_image_from_url(image_url, timeout=3, verbose=False):
    if verbose:
        print(image_url, end=" ")
    try:
        request = urllib.request.Request(
            image_url,
            data=None,
            headers={"user-agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as req:
            image = PIL.Image.open(io.BytesIO(req.read())).convert("RGB")
    except Exception as e:
        print(f"failed to load {image_url} with {e}")
        traceback.print_exc()
        image = None
    return image


def prepare_facebook_pmd_dataset(
    subset, save_dataset, save_subset, num_samples=4096, seed=None, save_dir='./data', upload_to_hub=True
    ):
    """ 
    This code only works for facebook/pmd datasets.
    Saving images and texts as numpy for faster loading later.
    
    NOTE: many caption datasets are not shuffled, hence downloading the full dataset is recommended. 
        This can generate roughly 50GB per dataset, so make sure you have enough space.
        The code will also through error messages if images/caption no longer exists.
        
    NOTE: (WARNING) original image set used in the paper downloaded 4096 samples, 
        sorted the files and took the first 1024 for the smaller subset.
        we used standard sort instead of natural sort so there is some 
        inconsistencies in the examples used. We uploaded the dataset used in the paper to huggingface.
    """
    
    save_dir = f"{save_dir}/{save_dataset}/{save_subset}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)

    if seed is None:
        seed = np.random.randint(0, 10000)

    print("downloading dataset ...")
    dataset = load_dataset(
        "facebook/pmd",
        subset,
        token=True,
        split="train",
        streaming=False,
    ).shuffle(seed=seed)
    print("done.")

    print("creating subset ...")
    pbar = tqdm(total=num_samples)
    
    dataset_idx = 0
    save_idx = 0

    metadata_file_path = f"{save_dir}/metadata.jsonl"
    # some datasets can have more than 1 mapping to captions so captions will always be in a list format
    
    if os.path.exists(metadata_file_path):
        os.remove(metadata_file_path)
    
    while save_idx < num_samples:

        d = dataset[dataset_idx]

        if d["image"] == None:
            get_image_fn = get_image_from_url
            image_key = "image_url"
        else:
            get_image_fn = get_image
            image_key = "image"

        im = get_image_fn(d[image_key])
        
        # make sure atleast 1 token long. it might be better to set it higher
        if (im is not None) and (len(d['text'].split(' ')) > 1):
        
            captions = [d['text']]
            # get original image name and store in data also
            
            if subset == "wit":
                ctx_sect = json.loads(d['meta'])['context_section_description']
                ctx_page = json.loads(d['meta'])['context_page_description']
                captions.extend([ctx_page, ctx_sect])
            
            # save image 
            relative_path = f'images/image_{save_idx}.jpg'
            image_save_path = f"{save_dir}/{relative_path}"
            im.save(image_save_path)

            # save text + metadata
            data = {
                'file_name': f"images/image_{save_idx}.jpg",
                'text': captions,  
                'origin': d['image_url'] 
            }

            with open(metadata_file_path, 'a') as file:
                json_line = json.dumps(data) + "\n"  # Convert dict to JSON string and add newline
                file.write(json_line)
            
            pbar.update(1)
            pbar.set_description(
                f"progress [{save_idx}/{num_samples}]"
            )
            save_idx += 1

        dataset_idx += 1

    # loading both images and captions is going to be wasteful for downstream case
    # so a simple naive workaround is by making two separate datasets
    
    print('saving image dataset')
    generated_dataset = load_dataset("imagefolder", data_dir=f"./data/prh/{save_subset}", split="train")
    
    if upload_to_hub:
        generated_dataset.push_to_hub(f"minhuh/{save_dataset}", revision=f"{save_subset}", private=True)
            
    print("done.\n")
    print(f"load the dataset via \t`load_dataset('minhuh/prh', revision={save_subset}, split='train')`")        
    return


if __name__ == "__main__":
    # example code of how the data partition was generated
    # see notes in prepare_facebook_pmd_dataset for minor details
    # prepare_facebook_pmd_dataset(subset="wit", save_dataset="prh", save_subset="wit_1024", num_samples=1024, seed=42)
    prepare_facebook_pmd_dataset(subset="wit", save_dataset="prh", save_subset="wit_4096", num_samples=4096, seed=42)