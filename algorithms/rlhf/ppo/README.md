# RLHF PPO (Learning Demo)

一个最小化的 RLHF PPO 实现，作为学习参考。改写自 [thomfoster/minRLHF](https://github.com/thomfoster/minRLHF)，带中文注释并修复了若干算法/实现问题。

## 运行

```bash
cd MLImpl/algorithms/rlhf
python train_ppo.py
```

---

## 一、算法回顾

### 1. PPO clipped surrogate objective

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\big( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \right]
$$

其中：

- $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$，$\pi_{\theta_\text{old}}$ 是 rollout 时的策略（buffer 里存的 `pi_t_logprobs`）
- $A_t$ 是 GAE 计算出的 advantage
- $\epsilon$ 是 clip 范围（`clip_ratio`，默认 0.2）

→ 实现见 `trainer.py:compute_actor_loss`。

### 2. GAE (Generalized Advantage Estimation)

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

可以理解为：把 $\delta_t$ 当作"折扣系数为 $\gamma\lambda$、奖励为 $\delta$"的折扣累积奖励。

→ 实现见 `buffer.py:_compute_advantages`，用 `discounted_cumsum_right` 一次性算累积和。

### 3. RLHF 中的 KL 惩罚

为了防止 actor 偏离 SFT 起点太远，把 KL 当作负奖励叠到 reward 上：

$$
r'_t = r_t - \beta \cdot \text{KL}\big(\pi_\theta(\cdot|s_t)\,\|\,\pi_\text{ref}(\cdot|s_t)\big)
$$

token-level KL 用 **k3 估计**（[Schulman, 2020](http://joschu.net/blog/kl-approx.html)）：

$$
\widehat{\text{KL}} = e^{-\Delta} - 1 + \Delta, \quad \Delta = \log\pi_\theta(a_t) - \log\pi_\text{ref}(a_t)
$$

k3 是无偏、恒非负、低方差的估计；优于 k1（方差大、单样本可负）和 k2（有偏）。

→ 实现见 `trainer.py:naive_logprob_augmenter`。

---

## 二、RLHF 三模型关系

| 模型 | 是否有梯度 | 角色 |
|---|---|---|
| **Actor** $\pi_\theta$ | 是 | 当前策略，被 PPO 优化 |
| **Critic** $V_\phi$ | 是 | 估计状态价值，给 advantage 提供 baseline |
| **Reference** $\pi_\text{ref}$ | 否 | 冻结的 SFT 模型，提供 KL 锚点 |
| **Reward Model** | 否 | 给完整 completion 打分（这份 demo 抽象成 `Environment.get_rewards`） |

→ 三个 LLM 的 forward 在 `trainer.py:get_rollout` 中分别在不同 device 上跑，通过 `gather_dict` 在设备间搬运中间结果。

---

## 三、模块结构

| 文件 | 作用 |
|---|---|
| `actor.py` | Actor 包装：`get_rollouts` (调 `generate`) + `get_logits` (算每个 token 的 logprob) |
| `critic.py` | Critic 包装：value head 前向得到 (B, L) 的 V(s) 估计 |
| `environment.py` | 抽象 `RLHFEnvironment`：`sample_prompts` / `get_rewards`；reward 是稀疏的（只在最后一个有效 token 上） |
| `buffer.py` | Rollout 池 + GAE / critic target / advantage 归一化 |
| `trainer.py` | 主训练循环：rollout → 填 buffer → 多个 PPO epoch 优化 actor & critic |
| `utils.py` | mask 操作、跨 device 搬运字典 |

---

## 四、训练循环

```
for epoch in range(num_epochs):
    # === Rollout 阶段（不带梯度） ===
    for _ in range(rollout_batches_per_epoch):
        prompts ← env.sample_prompts()
        completions ← actor.generate(prompts)              # actor device
        pi_t_logprobs ← actor(completions)                 # actor device
        pi_0_logprobs ← reference(completions)             # ref device
        rewards ← env.get_rewards(completions)             # cpu
        values ← critic(completions)                       # critic device
        buffer.store(...)

    # 在 buffer.get() 中计算 KL augmentation / advantages / critic targets
    # advantages 做全局 token-level 归一化

    # === PPO Update 阶段 ===
    for ppo_epoch in range(actor_train_iters):
        for mini_batch in buffer.get(actor_train_batch_size):
            loss = ppo_clip_loss(...)
            loss.backward(); optimizer.step()    # 每个 mini-batch 都 step
        if avg_kl > 1.5 * target_kl:
            break                                # PPO epoch 末尾 KL 早停

    for _ in range(critic_train_iters):
        for mini_batch in buffer.get(critic_train_batch_size):
            mse_loss(...).backward(); optimizer.step()
```

---

## 五、关键超参

| 参数 | 含义 | 典型值 |
|---|---|---|
| `clip_ratio` | PPO clip 的 ε | 0.2 |
| `gamma` | 折扣因子；RLHF 序列任务常设 1.0（轨迹短） | 1.0 |
| `gae_lambda` | GAE 的 λ；1.0 退化为 Monte Carlo，0 退化为 TD(0) | 0.95~1.0 |
| `beta` | KL 惩罚强度 | 0.01~0.1 |
| `target_kl` | KL 早停阈值；超过 1.5×target_kl 就中断本 epoch 后续 PPO 步 | 0.05 |
| `actor_train_iters` | 一份 buffer 数据被 PPO 重用的 epoch 数（K） | 2~4 |

---

## 六、已知改进点 / TODO

### 数据 / 接口
1. 找一个有实际意义的数据集进行训练
2. 支持向 Env 中传入 reward funcs / reward model
3. 将 data (dict) 封装成 `BufferItem` / `Experience`（可参考 OpenRLHF）

### 性能 / 代码质量
4. **`actor.py:42-51`** output_mask 用了双重 for 循环，可用 `cumsum` 向量化（长序列 + 大 batch 下是瓶颈）
5. **`actor.py:74-80`** `logits[:, 0, :] = 1e10` 的 trick 可换成直接置 0，反正 prompt 段会被 mask 掉，更直观
6. **`buffer.py:_compute_critic_targets` / `_compute_advantages`** 逐 episode 调 `discounted_cumsum_right`，可批量处理

### 算法严谨性
7. **`buffer.py:_compute_advantages`** GAE 中 terminal 状态 $V_{T+1}$ 没强制置 0；当前依赖 `completion_mask` 抹掉非有效区域，但严格的边界处理应区分 terminal vs non-terminal step

---

## 参考

- [PPO 论文 (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [GAE 论文 (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
- [Approximating KL Divergence (Schulman, 2020)](http://joschu.net/blog/kl-approx.html)
- [InstructGPT 论文 (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [thomfoster/minRLHF](https://github.com/thomfoster/minRLHF)（本 demo 的起点）
