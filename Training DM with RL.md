### TRAINING DIFFUSION MODELS  WITH REINFORCEMENT LEARNING
> https://rl-diffusion.github.io/

---

#### **核心贡献**
1. **提出DDPO框架：去噪扩散策略优化**
   - 将扩散模型的去噪过程建模为**多步马尔可夫决策过程（MDP）**，使强化学习策略梯度方法可直接优化扩散模型。
   - DDPO通过策略梯度直接优化下游任务目标，公式表达为：
     - **DDPO_SF（策略梯度）**：  
       $$\nabla_{\theta}\mathcal{J}_{DDRL} = E\left[ \sum_{t=0}^{T} \nabla_{\theta}\log p_{\theta}(x_{t-1} \mid x_t, c) \cdot r(x_0, c) \right]$$
     - **DDPO_IS（重要性采样）**：  
       $$\nabla_{\theta}\mathcal{J}_{DDRL} = E\left[ \sum_{t=0}^{T} \frac{p_{\theta}(x_{t-1} \mid x_t, c)}{p_{\theta_{\text{old}}}(x_{t-1} \mid x_t, c)} \nabla_{\theta}\log p_{\theta}(x_{t-1} \mid x_t, c) \cdot r(x_0, c) \right]$$
       通过trust region解决重要性采样的偏差问题。

1. **多任务的奖励函数设计**
   - **可压缩性/不可压缩性**：基于JPEG压缩后的文件大小设计奖励，解决提示词无法控制的技术指标问题。
   - **审美质量**：利用LAION美学预测器（基于人类评分的CLIP线性模型）作为奖励函数，公式为：  
     $$r_{\text{aesthetic}}(x_0, c) = \text{LAION\_score}(x_0)$$
   - **提示词对齐**：通过视觉语言模型（VLM，如LLaVA）生成图像描述，计算与原始提示词的语义相似度（BERTScore），公式为：  
     $$r_{\text{align}}(x_0, c) = \text{BERTScore}(c, \text{LLaVA}(x_0))$$

---

#### **方法脉络**

1. **问题定义**
   - **目标**：优化扩散模型的下游任务奖励 $\mathcal{J}_{DDRL}(\theta) = E_{c, x_0 \sim p_\theta}[r(x_0, c)]$
   - **挑战**：扩散模型的精确似然计算困难，传统RL方法难以直接应用。

2. **方法对比：RWR vs. DDPO**
   - **RWR（奖励加权回归）**：
     - 基于变分下界近似，对样本加权后优化加权对数似然：  
       $$\mathcal{L}_{\text{RWR}} = E\left[ w_{\text{RWR}}(x_0, c) \cdot \mathcal{L}_{\text{DDPM}} \right]$$
     - 权重设计包括指数加权（$w_{\text{RWR}} \propto \exp(\beta r)$）和稀疏加权（仅保留高奖励样本）。
     - 本质为单步MDP，仅优化最终样本分布。
   - **DDPO（多步MDP建模）**：
     - 将去噪过程映射为T步MDP，状态为$(c, t, x_t)$，动作为$x_{t-1}$，策略为$p_\theta(x_{t-1}|x_t,c)$，奖励$R(s_t,a_t)=r(x_0,c)$ if $t=0$ else $0$ 仅在最终步给出。
     - 通过策略梯度直接优化轨迹累积奖励，利用策略的精确对数概率（高斯分布特性）计算梯度。