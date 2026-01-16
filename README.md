# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation (IROS'2025)

## 🚀 Get Started

### Direct usage (Inference)
First we create the environment for inference and SFT training.
```
git clone https://github.com/yongchao98/Code-as-Symbolic-Planner.git
pip install -r requirements.txt
```
(In generation_models.py, fill in your OpenAI API key for GPT-4o or other keys fore other models not from OpenAI). Then we can run the testing Code-as-Symbolic-Planer and the baselines methods with:
```
python Code-as-Symbolic-Planner.py
```
and
```
python Test-Baseline-Methods.py
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
