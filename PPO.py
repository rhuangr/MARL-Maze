from neural_net import simple_nn
import torch
import numpy as np
import multiprocessing
import os

ACTOR_PATH = 'actor.pth'
CRITIC_PATH = 'critic.pth'

class PPO():
    def __init__(self, maze, epochs = 5000, batch_size = 6000,
                 discount_rate = 0.99, updates_per_batch = 5, mbatch_size = 64):
        
        self.maze = maze
        self.input_size = self.maze.observation_space
        self.output_size = self.maze.action_space
        self.actor = simple_nn([self.input_size, 250, 250, 250, self.output_size])
        self.critic  = simple_nn([self.input_size, 250, 1])

        self.epochs = epochs
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.updates_per_batch = updates_per_batch
        self.clip = 0.2
        self.beta = 0.01
        self.mbatch_size = 64
        self.max_grad = 0.5

        self.load_parameters()

    def train(self, maze_size_range = [8,12]):
        for i in range(self.epochs):
            batch_obs, batch_actions, batch_log_probs, batch_rew, batch_shortest_paths, episode_lens, entropies = self.get_batch(maze_size_range)
            print(f"Epoch{i+1}")
            for i in range(len(episode_lens)):
                print(f"Run {i+1}: exit time:{episode_lens[i]}, shortest path length: {batch_shortest_paths[i]}")
            print()

            rtgs = self.get_rtgs(batch_rew)
            # mbatch_updates = self.batch_size // self.mbatch_size
            index_list = np.arange(len(batch_obs))
            np.random.shuffle(index_list)

            for _ in range(self.updates_per_batch):

                for start in range(0, self.batch_size, self.mbatch_size):
                    end = start + self.mbatch_size
                    mbatch_indices = index_list[start:end]

                    m_obs = batch_obs[mbatch_indices]
                    m_actions = batch_actions[mbatch_indices]
                    m_log_probs = batch_log_probs[mbatch_indices]
                    m_rtgs = rtgs[mbatch_indices]
                    m_state_values = self.get_state_values(m_obs)
                    m_advantage = m_rtgs - m_state_values.detach()

                    m_entropies = entropies[mbatch_indices]

                    # normalize advantage to reduce variance
                    advantage = (m_advantage - torch.mean(m_advantage))/ torch.std(m_advantage) + 1e-10
                    current_log_prob = self.get_log_probs(m_obs, m_actions)
                    log_prob_ratios = torch.exp(current_log_prob - m_log_probs)

                    surrogate1 = log_prob_ratios * m_advantage
                    surrogate2 = torch.clamp(log_prob_ratios, 1-self.clip, 1+ self.clip) * m_advantage

                    actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
                    actor_loss = actor_loss - self.beta * torch.mean(torch.as_tensor(m_entropies))
                    self.actor.optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    critic_loss = torch.mean(torch.pow((m_rtgs - m_state_values), 2))
                    self.critic.optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                    critic_loss.backward()
                    self.critic.optimizer.step()

            self.save_parameters()
    
    def get_batch(self):
        
        batch_obs = []
        batch_rew = []
        batch_act = []
        episode_lens = []
        batch_log_probs = []
        batch_shortest_paths = []
        batch_entropies = []

        # we need seperate arrays for episode rewards to aid reward-to-go calculations
        episode_rew = []
        obs, action_mask = self.maze.reset()
        total_timesteps = 0

        while True:
            batch_obs.append(obs)
            action, log_prob, entropy = self.get_action(obs, action_mask)
            obs, action_mask, reward,done = self.maze.step(action)
            # print(f"action: {action}, prob: {torch.exp(log_prob)}")
            episode_rew.append(reward)
            batch_log_probs.append(log_prob)
            batch_entropies.append(entropy)
            batch_act.append(action)
            total_timesteps += 1
            # print(f"{self.maze.agent.x}, {self.maze.agent.y}")
            if done:
                batch_shortest_paths.append(self.maze.shortest_path_len)
                obs, action_mask = self.maze.reset()
                batch_rew.append(episode_rew)
                episode_lens.append(len(episode_rew))
                episode_rew = []
                
                if total_timesteps > self.batch_size:
                    break
            
            
        batch_obs = torch.as_tensor(batch_obs, dtype= torch.float32)
        batch_act = torch.as_tensor(batch_act, dtype= torch.float32)
        batch_log_probs = torch.as_tensor(batch_log_probs, dtype= torch.float32)
        batch_entropies = torch.as_tensor(batch_entropies, dtype= torch.float32)

        return batch_obs, batch_act, batch_log_probs, batch_rew, batch_shortest_paths, episode_lens, batch_entropies

    def get_action(self, obs, action_mask):
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
        logits = self.actor(obs)
        adjusted_logits = torch.where(action_mask, logits, torch.tensor(-1e+8))
        distribution = torch.distributions.Categorical(logits = adjusted_logits)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action), distribution.entropy()
        
    def get_log_probs(self, batch_obs, batch_actions):
        logits = self.actor(batch_obs)
        distribution = torch.distributions.Categorical(logits = logits)
        return distribution.log_prob(torch.as_tensor(batch_actions, dtype=torch.float32))
    
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

        rtgs.reverse()
        return torch.as_tensor(rtgs, dtype=torch.float32)
    
    def save_parameters(self):
        torch.save(self.actor.state_dict(), ACTOR_PATH)
        torch.save(self.critic.state_dict(), CRITIC_PATH)
        # print("parameters successfully saved")

    def load_parameters(self):
        if os.path.exists(ACTOR_PATH) and os.path.exists(CRITIC_PATH) and os.path.exists(ACTOR_PATH):
            print("successfuly loaded existing parameters")
            self.actor.load_state_dict(torch.load(ACTOR_PATH))
            self.critic.load_state_dict(torch.load(CRITIC_PATH))
    
    


