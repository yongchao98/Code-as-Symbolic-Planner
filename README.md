# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation (IROS'2025)

## 🚀 Get Started

### Direct usage (Inference)
First we create the environment for inference and SFT training.
```
git clone https://github.com/yongchao98/R1-Code-Interpreter.git
cd R1-Code-Interpreter
conda create -n llama_factory_infer python=3.11
conda activate llama_factory_infer
cd LLaMA-Factory
pip install -r requirements.txt
cd ..
```
(In benchmark_inference_test.py, fill your python local path of current directory in line 28 and choose desired model type in line 30; In generation_models.py and Search-R1/r1_code_inter/generation_models.py, fill in your OpenAI API for GPT-4o calling to extract the answer). Then we can run the testing R1-CI models with:
```
python benchmark_inference_test.py
```

## ✍️ Citation
```md
@article{chen2025code,
  title={Code-as-symbolic-planner: Foundation model-based robot planning via symbolic code generation},
  author={Chen, Yongchao and Hao, Yilun and Zhang, Yang and Fan, Chuchu},
  journal={arXiv preprint arXiv:2503.01700},
  year={2025}
}
```
