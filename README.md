

<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)


# WideSearch: Benchmarking Agentic Broad Info-Seeking
<a href="YOUR_ARXIV_LINK" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?style=for-the-badge&logo=arXiv&logoColor=white"
         alt="arXiv" />
</a>
<a href="https://widesearch-seed.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Homepage-blue.svg?style=for-the-badge&logo=google-chrome&logoColor=white"
         alt="Project Homepage" />
</a>
<a href="https://huggingface.co/datasets/ByteDance-Seed/WideSearch" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg?style=for-the-badge"
         alt="Hugging Face Dataset" />
</a>

---
We will release the arxiv paper soon! Stay tuned!
## News
[2025/08/11]🔥We release WideSearch Benchmark.


## Introduction
### From Tedious Labor to Autonomous Agent
Many real-world information-gathering tasks are not hard, just huge. Consider a financial analyst compiling key metrics for all companies in a sector, or a job seeker collecting every vacancy that meets their criteria. The challenge isn't cognitive complexity, but the sheer scale and repetitive nature of the work—a critical productivity bottleneck.

WideSearch is designed to evaluate an agent's ability to automate these tasks, shifting from laborious manual collection to efficient, automated workflows. This shift, however, introduces novel failure modes like hallucination and incompleteness, making rigorous evaluation essential.


### A New Paradigm: Wide vs. Deep
Current research primarily focuses on "deep" tasks. DeepSearch tackles the "I can't find it" problem of locating hidden facts, while DeepResearch addresses the "I can't write it well" problem of synthesizing reports.

In sharp contrast, WideSearch tackles the "I could do it, but the sheer volume is overwhelming" problem. It requires agents to systematically find and organize large-scale information into a structured output, shifting the primary challenge from deep search to achieving exhaustiveness and fidelity at scale.

## Experiments
We test both single-agent and multi-agent modes, and manually conducted end-to-end testing of the commercial AI system on the web interface. In addition, we randomly select 20 questions and invited human annotators to perform tests. The experiment results are as follows:
![experiments](figs/image.png)

## Quickstart

## Set up environment
Install dependencies, see `prepare-env.sh` for more details.
```
git clone https://github.com/ByteDance-Seed/WideSearch.git
cd WideSearch
sh prepare-env.sh
source .venv/bin/activate
```

## Configuration
1. Implement custom search tools in <a href="src/agent/tools.py">src/agent/tools.py</a>
2. Configure model parameters in <a href="src/utils/config.py">src/utils/config.py</a>

## Inference and Evaluation
Run the following command to perform inference and evaluation:
```
python3 scripts/run_infer_and_eval_batching.py \
--trial_num={your_trial_num} \
--model_config_name={your_model_config_name} \
--response_root={your_response_root} \
--result_save_root={your_result_save_root} \
--stage={infer/eval or both} 
``` 

## License
This project is licensed under MIT. See the <a href="LICENSE">LICENSE</a> file for details.

## Citation
If you find WideSearch useful for your research and applications, feel free to give us a star ⭐ and cite us using:

```bibtex
@misc{wong2025widesearchbenchmarkingagenticbroad,
      title={WideSearch: Benchmarking Agentic Broad Info-Seeking}, 
      author={Ryan Wong and Jiawei Wang and Junjie Zhao and Li Chen and Yan Gao and Long Zhang and Xuan Zhou and Zuo Wang and Kai Xiang and Ge Zhang and Wenhao Huang and Yang Wang and Ke Wang},
      year={2025},
      eprint={2508.07999},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07999}, 
}
```
