Status: WIP

# About

This repo documents the code for the FEVER9/AVerImaTeC shared task submission from the paper "Take It All: Ensemble Retrieval for Multimodal Evidence Aggregation".

NOTE: This repo is an example version of our pipeline for reference. The code is a cleaned up version of scripts we used during our experiments, here written for experiments with setting different factors of shrinking input images in mind. 

Our original code can be found in `old/full_script.py`, a one-file version based on a jupyter notebook.

# Preparation

Please download the following files:

- [AVerImaTeC: val set](https://huggingface.co/datasets/Rui4416/AVerImaTeC/raw/main/val.json)
- [image directory](https://huggingface.co/datasets/Rui4416/AVerImaTeC/resolve/main/images.zip)
- The links to the knowledge store can be found [here](
https://fever.ai/task.html)

# Running

- Installation: run `python -m pip install requirements.txt` in a dedicated env
- set `IMAGE_SHRINK_FACTOR` to experiment with different images sizes either in `inference.py` or `old/full_script.py`
