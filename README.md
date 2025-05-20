# Improving Generalization in Model-Based Reinforcement Learning with Sparse Transformers

## Abstract

This thesis introduces a novel sparse transformer architecture for model-based reinforcement learning in environments with sparse dependencies, where state transitions depend primarily on a small subset of state components. Traditional neural networks often struggle with generalization in such environments, as they consider all possible interactions between state components, leading to overfitting and poor sample efficiency. We formally define sparse-dependent environments and propose a simple yet effective modification to the standard transformer architecture that promotes sparsity in the attention mechanism through L1 regularization and thresholding.

Through extensive experiments on the Minigrid environment, we demonstrate that our sparse transformer consistently outperforms classical transformers in low-data regimes, achieving a 16.32\% higher validation accuracy ($0.7998$ vs $0.6876$, $p < 0.05$) and exhibiting significantly lower variance across random initializations ($\sigma$ = $0.0286$ vs $0.1042$). Statistical analysis confirms these improvements are significant at the 95\% confidence level with a large effect size (Cohen's d = $1.47$). Qualitative analysis of the learned attention patterns reveals that the sparse transformer successfully captures the minimal information requirements of the environment, focusing attention only on relevant state components. These findings suggest that incorporating sparsity as an inductive bias can significantly improve generalization, sample efficiency, and interpretability in reinforcement learning models.

Our research contributes to the ongoing effort to develop more efficient and robust reinforcement learning algorithms by demonstrating how architectural choices that align with environment structure can lead to improved generalization performance.

## Project Structure
`train.py` is the starting point. Executing this script will instantiate the pipeline define in `conf/train.yaml` which in turn looks into `conf/model/some-model.yaml`. The pipeline consists of modules defined in `src/modules/...`, with architecture implementations in `src/modules/training/models/`.

# License
This project is licensed under GNU GPLv3, including all original code written for this thesis. The project incorporates code from two external repositories, adapted for this use-case:
- U-Net implementation under `src/modules/training/models/unet.py`, by [milesial](https://github.com/milesial/Pytorch-UNet/) licensed under GNU GPLv3
- Transformer implementation under `src/modules/training/models/transformer/*.py`, by [Hyunwoong](https://github.com/hyunwoongko/transformer) licensed under Apache License 2.0

Due to the inclusion of GPLv3-licensed code, the entire project must be distributed under GPLv3 terms. A copy of the GNU GPLv3 license is included in the file `LICENSE` in this repository. The original Apache 2.0 licensed code has been incorporated in compliance with its license terms. This does not change the overall GPLv3 licensing of this project.