from neural_net import simple_nn
import torch
from numpy import mean, sum

class PPO():
    def __init__(self, maze, epochs = 50, batch_size = 5000,
                 discount_rate = 0.98, updates_per_batch = 5):
        
        self.maze = maze
        self.input_size = self.maze.observation_space
        self.output_size = self.env.action_space
        self.actor = simple_nn(self.input_size, self.output_size)
        self.critic  = simple_nn(self.input_size, 1)

        self.epochs = epochs
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.updates_per_batch = updates_per_batch
        self.clip = 0.2

    def train(self):
        for i in range(self.epochs):
            batch_obs, batch_actions, batch_log_probs, batch_rew, batch_lens = self.get_batch()
            print(f"Epoch average exit time: {mean(batch_lens)}")
            print(f"")
            batch_rtgs = self.get_rtgs(batch_rew)
            
            for i in range(self.updates_per_batch):
                state_values = self.get_state_values(batch_obs)
                advantage = batch_rtgs - state_values.detach()

                # normalize advantage to reduce variance
                advantage = (advantage - torch.mean(advantage))/ torch.std(advantage) + 1e-10
                current_log_prob = self.get_log_probs(batch_obs, batch_actions)
                log_prob_ratios = torch.exp(current_log_prob - batch_log_probs)

                surrogate1 = log_prob_ratios * advantage
                surrogate2 = torch.clamp(log_prob_ratios, 1-self.clip, 1+ self.clip) * advantage

                actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                critic_loss = torch.mean(torch.pow((batch_rtgs - state_values), 2))
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

    
    def get_batch(self):
        batch_obs = []
        batch_rew = []
        batch_act = []
        batch_log_probs = []
        batch_lens = []

        # we need seperate arrays for episode rewards to aid reward-to-go calculations
        episode_rew = []
        obs = self.env.reset()[0]
        current_t = 0

        while True:

            batch_obs.append(obs)
            action, log_prob = self.get_action(obs)
            obs,reward,done = self.maze.step(action)

            episode_rew.append(reward)
            batch_log_probs.append(log_prob)
            batch_act.append(action)
            current_t += 1

            if done:
                obs = self.env.reset()[0]
                batch_rew.append(episode_rew)
                batch_lens.append(len(episode_rew))
                print(len(batch_obs))
                episode_rew = []
                
                if current_t > self.batch_size:
                    break

        return batch_obs, batch_act, torch.as_tensor(batch_log_probs, dtype= torch.float32), batch_rew, batch_lens

    def get_action(self, obs):
        logits = self.actor(obs)
        distribution = torch.distributions.Categorical(logits = logits)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
        
    def get_log_probs(self, batch_obs, batch_actions):
        logits = self.actor(batch_obs)
        distributon = torch.distributions.Categorical(logits = logits)
        return distributon.log_prob(torch.as_tensor(batch_actions, dtype=torch.float32))
    
    def get_state_values(self, batch_obs):
        # batch_obs is a nested array [[obs1], [obs2], ...]
        return self.critic(batch_obs).squeeze()


    def get_rtgs(self, batch_rew):
        rtgs = []
        # for rew in batch_rew:
        #     print(len(rew),end=" ")
        # print()
        # current = 0
        
        for episode_rew in reversed(batch_rew):
            discounted_rew = 0
            # print(f"hi: {len(episode_rew)}", end= " ")
            for rew in reversed(episode_rew):
                discounted_rew = rew + self.discount_rate * discounted_rew
                # original methods use insert(0, discounted_rew) which is 0(n)
                rtgs.append(discounted_rew)  
            print(len(rtgs),end = " ")
        print()

        rtgs.reverse()
        return torch.as_tensor(rtgs, dtype=torch.float32)
    

if __name__ == "__main__":
    PPO('CartPole-v1').train()