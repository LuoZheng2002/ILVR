from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="shuai22/comt", filename="TRAIN.jsonl", repo_type="dataset", cache_dir="./data")
hf_hub_download(repo_id="shuai22/comt", filename="TEST.jsonl", repo_type="dataset", cache_dir="./data")
hf_hub_download(repo_id="shuai22/comt", filename="comt.tar.gz", repo_type="dataset", cache_dir="./data")