import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from collections import defaultdict

from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter

from .actor import Actor
from .critic import Critic
from .environment import RLHFEnvironment
from .utils import gather_dict
from .buffer import Buffer



class RLHFPPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: nn.Module,
        reference_model: nn.Module,
        env: RLHFEnvironment,
        actor_pad_token_id: int,
        max_episode_length: int = 100,
        rollout_batch_size: int = 32,
        rollout_batches_per_epoch: int = 4,
        num_epochs: int = 1000,
        actor_train_batch_size: int = 16,
        actor_train_iters: int = 4,  # ppo epochs
        actor_lr: float = 1e-5,
        critic_train_batch_size: int = 16,
        critic_train_iters: int = 8,
        critic_lr: float = 1e-5,
        target_kl: float = 0.05,
        clip_ratio: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        beta: float = 0.05,
        save_steps: int = 50,
        log_stesp: int = 1,
        log_smoothing_val: float = 0.95,
        working_dir: str = 'workspace'
    ):
        """Constructor for PPOTrainer class.
        
        Args:
            actor (Model): Needs to implement generate and forward.
            
            critic (Model): Needs to forward method.
            
            env (RLHFEnvironment): Env used to generate initial prompts and score the actor's generations.
        """


        self.actor_model = actor_model
        self.critic_model = critic_model
        self.reference_model = reference_model
        self.env = env
        self.max_episode_length = max_episode_length
        self.rollout_batch_size = rollout_batch_size
        self.rollout_batches_per_epoch = rollout_batches_per_epoch
        self.num_epochs = num_epochs
        self.actor_train_batch_size = actor_train_batch_size
        self.actor_train_iters = actor_train_iters
        self.actor_lr = actor_lr
        self.critic_train_batch_size = critic_train_batch_size
        self.critic_train_iters = critic_train_iters
        self.critic_lr = critic_lr
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.beta = beta
        self.save_steps = save_steps
        self.log_steps = log_stesp
        self.log_smoothing_val = log_smoothing_val
        self.working_dir = working_dir

        # setup actor
        self.actor = Actor(actor_model, pad_token_id=actor_pad_token_id, max_length=self.max_episode_length)
        self.actor_optimizer = AdamW(self.actor.model.parameters(), lr=self.actor_lr)
        self.actor_lr_scheduler = get_scheduler('linear', optimizer=self.actor_optimizer, num_warmup_steps=0, num_training_steps=num_epochs)

        # setup critic
        self.critic = Critic(critic_model)
        self.critic_optimizer = AdamW(self.critic.model.parameters(), lr=self.critic_lr)
        self.critic_lr_scheduler = get_scheduler('linear', optimizer=self.critic_optimizer, num_warmup_steps=0, num_training_steps=num_epochs)

        # setup others
        self.reference = Actor(self.reference_model, pad_token_id=actor_pad_token_id, max_length=self.max_episode_length)
        self.env = env

        def naive_logprob_augmenter(buf: Buffer) -> None:       # pi_\theta 和 pi_ref 的 KL 散度
            buf.reward_augmentation_buffer[:, :] = -((buf.pi_t_logprobs_buffer - buf.pi_0_logprobs_buffer) ** 2)/2

        self.buffer = Buffer(
            max_episodes=self.rollout_batch_size * self.rollout_batches_per_epoch,   # buffer size = rollout_batch_size * rollout_batches_per_epoch
            max_episode_length=self.max_episode_length,
            reward_augmenter=naive_logprob_augmenter,
            device=torch.device('cpu')
        ) # TODO: Might need to do this at train time

        self.summary_writer = SummaryWriter(os.path.join(self.working_dir, f'logs'))

    def compute_actor_loss(self, data):
        data = gather_dict(data, self.actor.device, keys=['ids', 'prompt_mask', 'completion_mask', 'advantages', 'pi_0_logprobs', 'pi_t_logprobs'])

        # compute logprobs with gradients for most recent policy
        logprobs, pi = self.actor.get_logits(data['ids'], data['prompt_mask'], data['completion_mask'])
        # mask and flatten into sequence of decisions
        masked_logprobs = logprobs.masked_select(data['completion_mask'].to(torch.bool))
        masked_old_logprobs = data['pi_t_logprobs'].masked_select(data['completion_mask'].to(torch.bool))
        masked_advantages = data['advantages'].masked_select(data['completion_mask'].to(torch.bool))

        # compute ppo loss
        ratio = torch.exp(masked_logprobs - masked_old_logprobs)
        cliped_ratio = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        loss = -(torch.min(ratio * masked_advantages, cliped_ratio * masked_advantages)).mean()

        # useful metrics for logging
        kld_t = (masked_logprobs - masked_old_logprobs).mean().item()
        masked_logprobs_0 = data['pi_0_logprobs'].masked_select(data['completion_mask'].to(torch.bool))
        kld_0 = (masked_logprobs - masked_logprobs_0).mean().item()
        entropy = pi.entropy().masked_select(data['completion_mask'].to(torch.bool)).mean().item()

        metrics = {
            'loss': loss,
            'kld_t-1': kld_t,
            'kld_0': kld_0,
            'entropy': entropy
        }

        return loss, metrics

    def compute_critic_loss(self, data):
        data = gather_dict(data, self.critic.device, keys=['ids', 'prompt_mask', 'completion_mask', 'critic_targets'])

        # recompute logits with gradient (buffer stores with no grad)
        train_logits = self.critic.get_value_estimates(
            data['ids'], data['prompt_mask'], data['completion_mask']
        ).masked_select(data['completion_mask'].to(torch.bool))

        train_targets = data['critic_targets'].masked_select(data['completion_mask'].to(torch.bool))

        loss = ((train_logits - train_targets) ** 2).mean()
        mae = (train_logits - train_targets).abs().mean()

        metrics = {
            'loss-mse': loss.item(),
            'mae': mae.item()
        }

        return loss, metrics

    @torch.no_grad()
    def get_rollout(self, batch_size: int):
        data = {}   # 将过程中的变量都存在这个字典里，配合 gather_dict 在各设备间移动

        data['prompt_ids'], data['prompt_mask'] = self.env.sample_prompts(batch_size)     # 采样 prompt 

        # 数据挪到 actor device 上，计算 actor 的 logprobs
        # 注意这里的 pi_t_logprobs 相当于是 pi_\theta_old 的 logprobs
        data = gather_dict(data, self.actor.device, keys=['prompt_ids', 'prompt_mask'])
        data['completion_ids'], data['completion_mask'] = self.actor.get_rollouts(data['prompt_ids'], data['prompt_mask'])
        data['pi_t_logprobs'], _ = self.actor.get_logits(data['completion_ids'], data['prompt_mask'], data['completion_mask'])

        # 数据挪到 reference model 的 device 上，计算 ref model 的 logprobs
        # 注意这里的回答（completion_ids）不重新采样，用 actor model rollout 出来的
        # 注意这里的 pi_0_logprobs 相当于是 pi_ref 的 logprobs
        data = gather_dict(data, self.reference.device, keys=['completion_ids', 'prompt_mask', 'completion_mask'])
        data['pi_0_logprobs'], _ = self.reference.get_logits(data['completion_ids'], data['prompt_mask'], data['completion_mask'])

        # 在 cpu 上计算 reward
        data = gather_dict(data, 'cpu', keys=['completion_ids', 'prompt_mask', 'completion_mask'])
        data['reward'] = self.env.get_rewards(data['completion_ids'], data['prompt_mask'], data['completion_mask'])

        # 在 critic device 上计算 critic values
        data = gather_dict(data, self.critic.device, keys=['completion_ids', 'prompt_mask', 'completion_mask'])
        data['value_estimates'] = self.critic.get_value_estimates(data['completion_ids'], data['prompt_mask'], data['completion_mask'])

        # Buffer 保存的都是 (batch, seq_len) 形状的 tensor，所以给 prompt_mask pad 一下，和其他 tensor 形状一致
        pad_length = data['completion_mask'].shape[1] - data['prompt_mask'].shape[1]
        data['prompt_mask'] = F.pad(data['prompt_mask'], (0, pad_length))

        # Do some key mangling to make the return dict map to our buffer correctly
        # 适配一下 Buffer 中各个 Key 的名字
        data.pop('prompt_ids')
        data['state'] = data.pop('completion_ids')

        return data

    
    def train(self):
        print('start training')
        all_metrics = defaultdict(list)
        for epoch in range(self.num_epochs):
            self.buffer.reset()

            # 采样 rollout_batches_per_epoch * rollout_batch_size 条数据, 并存到 buffer 中
            for rollout_batch_idx in range(self.rollout_batches_per_epoch):
                # print(f'generating rollout data {rollout_batch_idx}')
                rollouts = self.get_rollout(batch_size=self.rollout_batch_size)
                rollouts = gather_dict(rollouts, self.buffer.device)
                self.buffer.store(**rollouts)

            # 存满 buffer 后，从 buffer 中采样 batch 数据并训练 actor/critic model
            for actor_train_step in range(self.actor_train_iters):
                # Buffer.get 方法会采样数据，并计算 critic targets 和 advantages
                # Buffer.get 返回的是一个迭代器
                actor_train_batches = self.buffer.get(self.actor_train_batch_size, self.gamma, self.gae_lambda, self.beta)

                average_actor_loss_info = defaultdict(list)
                for batch_idx, buf_data in enumerate(actor_train_batches):
                    actor_loss, actor_loss_info = self.compute_actor_loss(buf_data)
                    actor_loss.backward()       # 多次 backward，梯度累积

                    for k, v in actor_loss_info.items():
                        average_actor_loss_info[k].append(v)

                for k, v in average_actor_loss_info.items():
                    v_avg = sum(v) / len(v)
                    all_metrics[k].append(v_avg)
                    self.summary_writer.add_scalar(f"actor-{k}", v_avg, actor_train_step + epoch * self.actor_train_iters)
                    
                avg_kl = torch.tensor(average_actor_loss_info['kld_t-1']).mean().item()
                if avg_kl > 1.5 * self.target_kl:
                    print(f'(Epoch:{epoch} actor iter: {actor_train_step}) Early stopping due to kl of ~', avg_kl)
                    break

                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()

            self.actor_lr_scheduler.step()

            # Use the rollouts to optimise the critic

            for critic_train_step in range(self.critic_train_iters):
                critic_train_batches = self.buffer.get(self.critic_train_batch_size, self.gamma, self.gae_lambda, self.beta)

                average_critic_loss_info = defaultdict(list)
                for batch_idx, buf_data in enumerate(critic_train_batches):
                    # print(f'Getting critic loss for step {critic_train_step} and batch {batch_idx}')
                    critic_loss, critic_loss_info = self.compute_critic_loss(buf_data)
                    critic_loss.backward()

                    for k, v in critic_loss_info.items():
                        average_critic_loss_info[k].append(v)

                for k, v in average_critic_loss_info.items():
                    v_avg = sum(v) / len(v)
                    all_metrics[k].append(v_avg)
                    self.summary_writer.add_scalar(f"critic-{k}", v_avg, critic_train_step + epoch * self.critic_train_iters)
                    
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()
                
            self.critic_lr_scheduler.step()

            # Logging
            for k, v in self.buffer.summary().items():
                all_metrics[k].append(v)
                self.summary_writer.add_scalar(f"rollout-{k}", v, epoch)


            if (epoch + 1 ) % self.log_steps == 0:
                print(f"[EP {epoch:4d}] ", end='')
                for k, v in all_metrics.items():
                    print(f' {k}: {v[-1]:+3.3f} |', end='')
                print()

            if (epoch + 1)%self.save_steps == 0:
                model_fpath = os.path.join(self.working_dir, f'actor_{epoch}.model')
                self.actor.model.save_pretrained(model_fpath)
                print(f'(Epoch: {epoch}) Saved model to {model_fpath}')

    






