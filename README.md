# CCFT
This repository tracks the CCFT project's trajectory and maintains its code. 

This project implements fine-tuning of the [Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL) vision-language model using the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) framework. The implementation involves cloning both repositories and applying custom modifications to enable CCFT fine-tuning.

## Project Structure
```bash
.
├── LLaMA-Factory/          # Core framework with custom training scripts
│   ├── configs/            # Configuration files
│   ├── data/               # Custom dataset
│   ├── saves/              # The folder to store LoRA adapter checkpoints
│   ├── models/             # The folder to store the merged models (base + adapters), ready for inference 
│   ├── train_eval.sh       # The script to fine-tune models and call 'Qwen2.5-VL/inference_eval.py' to evaluation.
│   └── ... 
│
├── Qwen2.5-VL/             # Base model implementation
│   ├── eval_results/       # The folder to store evaluation results
│   ├── inference_eval.py   # The script to evaluate fine-tuned models (from 'LLaMA-Factory/models/') on the testing dataset
│   └── ...                 
│
└── README.md               # Project documentation 
```
## Environment Preparation
Due to dependency conflicts between the LLaMA Factory framework and Qwen2.5-VL requirements, this project requires **two separate Conda environments**.

### LLaMA-Factory environment
```bash
cd LLaMA-Factory
conda create --name llama-qw python=3.10 -y
conda activate llama-qw
pip install -e ".[torch,metrics]"
```
### Qwen2.5-VL environment
```bash
cd Qwen2.5-VL
conda create --name Qwen25VL python=3.10 -y
conda activate Qwen25VL
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm>0.7.2'
```

## Data Preparation
TBD. Where I can store the data.

## Fine-tuning and Evaluation
We provide an end-to-end fine-tuning and evaluation script. The fine-tuned models will be stored in '/LLaMA-Factory/models' and the evaluation results will be stored in '/Qwen2.5-VL/eval_results'.

Run ``` sh ./LLaMA-Factory/train_eval.sh ```

## Acknowledgments

This project stands on the shoulders of two significant open-source projects:
| Project | Contribution | License |
|---------|-------------|---------|
| **[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)** | Fine-tuning framework<br>LoRA implementation<br>Training pipeline | Apache 2.0 |
| **[Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL)** | Vision-language model<br>Multimodal architecture<br>Pretrained weights | Apache 2.0 |

We acknowledge the invaluable contributions of both teams to the open-source AI community.


## Citation
TBD