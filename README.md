# CCISolver
This is the repository of TSE submission "CCISolver: End-to-End Detection and Repair of Method-Level Code-Comment Inconsistency"

## Repository Structure
```
├── Detector
│   ├── test.cmd
│   ├── ti_model.py
│   ├── ti_run.py
│   ├── train.cmd
│   └── updated_tokenizer
│       ├── added_tokens.json
│       ├── merges.txt
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── Fixer
│   ├── finetune_qwen2.py
│   ├── finetuneinsturct.py
│   ├── qwenc14b_ft_lora.sh
│   └── sm_predict.py
├── README.md
└── requirements.txt
```

## Dataset
We propose CCIBench, a high-quality dataset cleaned via syntactic rules and semantic-based LLM voting.

Download: The dataset is available at: https://drive.google.com/drive/folders/1c4iTYKoeUnXnV9qUXKwp95Fs6eolyLiY?usp=drive_link

## Requirements 
**Hardware**

GPU: Recommended A100 (40G) or equivalent (especially for the Fixer module).

RAM: 32GB+ System RAM.

**Software**

This project is implemented in Python 3.8+. To install the necessary dependencies, please run:
```
pip install -r requirements.txt
```

## CCIDetector
The detector uses microsoft/unixcoder-base as the backbone. It classifies whether a code change and comment are inconsistent.

To train the detector on CCIBench, use the provided command script. Ensure you point to the correct data path inside the script if necessary.
```
bash Detector/train.cmd
```

To evaluate the detector on the test set:
```
bash Detector/test.cmd
```
## CCIFixer
The fixer uses Qwen/Qwen2.5-Coder-14B-Instruct as the base model, fine-tuned using LoRA and aligned via KTO.

**Fine-tuning**
```
cd Fixer
# You may need to modify the paths in the .sh file to point to your dataset and base model
bash qwenc14b_ft_lora.sh
```

**Inference**
```
cd Fixer
python sm_predict.py
```