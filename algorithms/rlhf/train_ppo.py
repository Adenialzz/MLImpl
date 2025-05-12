import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from transformers.pipelines import pipeline

from ppo.trainer import RLHFPPOTrainer
from ppo.environment import RLHFEnvironment



class MyEnv(RLHFEnvironment):

    def __init__(self, reward_model, tokenizer):
        self.reward_model = reward_model
        super().__init__(tokenizer)

    def get_input_prompt(self):
        return random.choice([
            'I went for a walk one day and',
            'A long time ago, in a galaxy far far away',
            'Oops! I'
        ])

    def score_generation(self, text: str):
        sentiment_scores = self.reward_model(text)[0]
        sentiment_scores = {d['label']: d['score'] for d in sentiment_scores}
        return sentiment_scores['joy']
        

def main():
    model_id_or_path = 'gpt2'      
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    actor_model = AutoModelForCausalLM.from_pretrained(model_id_or_path).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id_or_path).to(device)
    critic_model = AutoModelForTokenClassification.from_pretrained(model_id_or_path, num_labels=1).to(device)
    reward_model = pipeline(
        "text-classification",
        model='bhadresh-savani/distilbert-base-uncased-emotion', 
        return_all_scores=True
    )

    env = MyEnv(reward_model, tokenizer)

    trainer = RLHFPPOTrainer(
        actor_model=actor_model,
        critic_model=critic_model,
        reference_model=ref_model,
        env=env,
        actor_pad_token_id=tokenizer.pad_token_id,
        max_episode_length=100,
        log_stesp=10,
        num_epochs=1000,
        save_steps=200,
        actor_lr=1e-5,
        critic_lr=1e-5,
        working_dir=f'workspace-{time.time()}'
    )

    trainer.train()

if __name__ == '__main__':
    main()



