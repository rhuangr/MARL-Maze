from networks import Actor,Critic
import maze
import torch
import numpy as np
import multiprocessing
import os

torch.manual_seed(3234)

ACTOR_PATH = 'actor.pth'
CRITIC_PATH = 'critic.pth'

class PPO():
    def __init__(self, epochs=5000, batch_size=5000, discount_rate=0.99, lam=0.9,
                  updates_per_batch=5, mbatch_size=64, clip=0.25, beta=0.05, max_grad=0.5):
        
        self.maze = None
        self.actor = Actor([180,180,180,180])
        self.critic  = Critic(hidden_sizes=[264,264])

        self.epochs = epochs
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.lam = lam
        self.updates_per_batch = updates_per_batch
        self.mbatch_size = mbatch_size
        self.clip = clip    
        self.beta = beta
        self.max_grad = max_grad

        self.load_parameters()

    def train(self):
        for epoch in range(self.epochs):
            b_obs, b_actions, b_log_probs, b_rew, b_shortest_paths, episode_lens, b_masks, b_advs, b_vals = self.get_batch()
            print(f"-------------------- Epoch #{epoch} --------------------")
            print(f"Mazes solved in current epoch: {len(episode_lens)}")
            print(f"Average Exit Time: {np.mean(episode_lens)}")
            print(f"Average Length of Shortest Path: {np.mean(b_shortest_paths)}")
            # print(f"Timesteps So Far: {t_so_far}", flush=True)
            # print(f"Iteration took: {delta_t} secs", flush=True)
            print(f"--------------------------------------------------", flush=True)
            print()
            # print(f"Epoch {epoch+1}")
            # for epoch in range(len(episode_lens)):
            #     print(f"Run {epoch+1}: exit time:{episode_lens[epoch]}, shortest path length: {batch_shortest_paths[epoch]}")
            # print()
            b_rtgs = self.get_rtgs(b_rew)
            b_advs = b_rtgs - b_vals.detach()
            # rtgs = b_advs + b_vals.detach()
            # normalize advantage to reduce variance
            b_advs = (b_advs - torch.mean(b_advs))/ torch.std(b_advs) + 1e-10
            index_list = np.arange(len(b_obs))
            np.random.shuffle(index_list)

            for update in range(self.updates_per_batch):
                for start in range(0, self.batch_size, self.mbatch_size):
                    end = start + self.mbatch_size
                    mbatch_indices = index_list[start:end]

                    # minibatch variables
                    m_obs = b_obs[mbatch_indices]
                    m_actions = b_actions[mbatch_indices]
                    m_log_probs = b_log_probs[mbatch_indices]
                    m_rtgs = b_rtgs[mbatch_indices]
                    m_state_values = self.get_state_values(m_obs)
                    m_advantage = b_advs[mbatch_indices]
                    m_masks = b_masks[mbatch_indices]
                    current_log_prob = 0
                    for i in range(len(self.maze.agents)):
                        current_log_prob += self.get_log_probs(i, m_obs, m_actions, m_masks)
                    log_prob_ratios = torch.exp(current_log_prob - m_log_probs)

                    if start == 0 and update == 0:
                        print(log_prob_ratios)

                    surrogate1 = log_prob_ratios * m_advantage
                    surrogate2 = torch.clamp(log_prob_ratios, 1-self.clip, 1+ self.clip) * m_advantage
                    
                    # actor loss calculations
                    actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
                    actor_loss = actor_loss #- self.beta * torch.mean(torch.as_tensor(m_entropies))
                    self.actor.optimizer.zero_grad()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                    actor_loss.backward()
                    self.actor.optimizer.step()

                    critic_loss = torch.nn.MSELoss()(m_state_values, m_rtgs)
                    self.critic.optimizer.zero_grad()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                    critic_loss.backward()
                    self.critic.optimizer.step()
                            
            self.save_parameters()
    
    def get_batch(self):
        
        batch_obs = []
        batch_rew = []
        batch_act = []
        batch_log_probs = []
        batch_masks = []
        batch_advantages = []
        batch_vals = []

        # used to display learning progress
        batch_shortest_paths = []
        episode_lens = []

        episode_rew = []
        episode_vals = []
        episode_dones = []
        obs, action_mask = self.maze.reset()
        total_timesteps = 0

        while True:
            batch_obs.append(obs)
            batch_masks.append(action_mask)

            MA_actions = []
            MA_log_probs = 0
            for i in range(len(self.maze.agents)):
                observation, mask = obs[i], action_mask[i]            
                action, log_prob = self.get_action(observation, mask)
                MA_actions.append(action)
                MA_log_probs += log_prob

            obs, action_mask, reward, done = self.maze.step(MA_actions)
            # print(f"action: {action}, prob: {torch.exp(log_prob)}")
            batch_log_probs.append(torch.sum(MA_log_probs))
            batch_act.append(MA_actions)
            episode_rew.append(reward)
            episode_vals.append(self.critic(obs))
            episode_dones.append(done)

            total_timesteps += 1
            if done:
                batch_shortest_paths.append(self.maze.shortest_path_len)
                obs, action_mask = self.maze.reset()
                batch_rew.append(episode_rew)
                episode_lens.append(len(episode_rew))
                batch_vals.extend(episode_vals)
                batch_advantages.extend(self.get_GAEs(episode_rew, episode_vals, episode_dones))
                episode_rew = []
                episode_vals = []
                episode_dones = []
                
                if total_timesteps > self.batch_size:
                    break
            
        batch_obs = torch.as_tensor(batch_obs, dtype= torch.float32)
        batch_act = torch.as_tensor(batch_act, dtype= torch.float32)
        batch_log_probs = torch.as_tensor(batch_log_probs, dtype= torch.float32)
        batch_masks = torch.as_tensor(batch_masks, dtype=torch.float32)
        batch_advantages = torch.as_tensor(batch_advantages, dtype=torch.float32)
        batch_vals = torch.as_tensor(batch_vals, dtype=torch.float32)

        return (batch_obs, batch_act, batch_log_probs, batch_rew, batch_shortest_paths,
                 episode_lens, batch_masks, batch_advantages, batch_vals)
    
    def get_log_probs(self, i, batch_obs, batch_actions, batch_masks):
        batch_moves, batch_marks, batch_signals = batch_actions[:,i,0], batch_actions[:,i,1], batch_actions[:,i,2]
        move_logits, mark_logits, signal_logits = self.actor(batch_obs[:,i,:])

        move_logits.masked_fill_(~torch.as_tensor(batch_masks[:,i,0:5], dtype=torch.bool), float('-inf'))
        distribution = torch.distributions.Categorical(logits=move_logits)  
        move_probs = distribution.log_prob(batch_moves)

        mark_logits = mark_logits.squeeze()
        mark_logits.masked_fill_(~torch.as_tensor(batch_masks[:,i,5], dtype=torch.bool), float('-inf'))
        mark_prob = torch.sigmoid(mark_logits)
        mark_prob = torch.where(torch.as_tensor(batch_marks, dtype=torch.bool), mark_prob, 1 - mark_prob)
        
        signal_logits = signal_logits.squeeze()
        signal_logits.masked_fill_(~torch.as_tensor(batch_masks[:,i,6], dtype=torch.bool), float('-inf'))
        signal_prob = torch.sigmoid(signal_logits)
        signal_prob = torch.where(torch.as_tensor(batch_signals, dtype=torch.bool), signal_prob, 1 - signal_prob)
        
        # calculating log prob
        log_probs = move_probs + torch.log(mark_prob) + torch.log(signal_prob)
        # print(f"{move_probs} + {torch.log(mark_prob)} = {log_probs}")
        return log_probs

    def get_action(self, obs, action_mask): 
        move_logits, mark_logits, signal_logits = self.actor(obs)

        # sampling move
        adjusted_logits = torch.where(torch.as_tensor(action_mask[0:5], dtype=torch.bool), move_logits, torch.tensor(-float('inf')))
        distribution = torch.distributions.Categorical(logits=adjusted_logits)
        move = distribution.sample()
        
        # sampling mark
        mark_prob = torch.sigmoid(mark_logits) if action_mask[5] == True else torch.tensor([[0]],dtype=torch.float32)
        mark = torch.bernoulli(mark_prob)
        mark_prob = mark_prob if mark == 1 else 1-mark_prob # p(marking) = mark prob, p(not marking) = 1 - p(marking)

        # sampling signal
        signal_prob = torch.sigmoid(signal_logits) if action_mask[6] == True else torch.tensor(0,dtype=torch.float32)
        signal = torch.bernoulli(signal_prob)
        signal_prob = signal_prob if signal == 1 else 1-signal_prob
        
        # calculating jointlog probability of all moves
        log_prob = distribution.log_prob(move) + torch.log(mark_prob) + torch.log(signal_prob)
        
        return [move.item(), mark.item(), signal.item()], log_prob
    
    
    def get_state_values(self, batch_obs):
        batch_values = self.critic(batch_obs)
        return batch_values.squeeze()

    def get_GAEs(self, ep_rew, ep_values, ep_dones):
        advantages = np.zeros_like(ep_rew)
        advantage = 0
        for t in reversed(range(len(ep_rew))):
            if t+1 == len(ep_rew):
                delta = ep_rew[t] - ep_values[t]
            else:
                delta = ep_rew[t] + self.discount_rate*ep_values[t+1]*(1-ep_dones[t+1]) - ep_values[t]
            advantage = delta + self.discount_rate* self.lam * (1 - ep_dones[t]) * advantage
            advantages[t] = advantage
        return advantages
    
    def get_rtgs(self, batch_rew):
        rtgs = []     
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
            return True
        return False

if __name__ == "__main__":
    maze = maze.Maze()
    brain = PPO(maze=maze)
    obs, mask = maze.reset()
    action, logprob, _ = brain.get_action(obs, torch.as_tensor(mask, dtype=torch.bool))
    logporbbb = brain.get_log_probs(obs, torch.as_tensor(action, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.bool))
            


