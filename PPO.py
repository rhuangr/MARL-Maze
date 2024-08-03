from networks import Actor,Critic
import maze
import torch
import numpy as np
import os

torch.manual_seed(3234)

MODEL_PATH = 'PPO.pth'

class PPO():
    def __init__(self, agent_amount, epochs=200, batch_size=17500, lr = 0.0001, discount_rate=0.995, lam=0.9,
                  updates_per_batch=5, clip=0.2, max_grad=0.5):
        
        self.maze = None
        self.actor = Actor([200,220,240])
        self.critic  = Critic(agent_amount, hidden_sizes=[64,64])
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr = lr)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.discount_rate = discount_rate
        self.lam = lam
        self.updates_per_batch = updates_per_batch
        self.mbatch_size = self.batch_size//10
        self.clip = clip    
        self.max_grad = max_grad

        self.load_parameters()
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr
            
    def train(self):
        for epoch in range(self.epochs):
            b_obs, b_actions, b_log_probs, b_shortest_paths, episode_lens, b_masks, b_advs, b_vals = self.get_batch()
            worst = np.max(episode_lens)
            print(f"-------------------- Epoch #{epoch} --------------------")
            print(f"Mazes solved in current epoch: {len(episode_lens)}")
            print(f"Average Exit Time: {np.mean(episode_lens)}")
            print(f"Average Length Excluding Worst: {(np.sum(episode_lens) - worst)/(len(episode_lens)-1)}")
            print(f"Best Exit Time: {np.min(episode_lens)} and Worst Exit Time: {np.max(episode_lens)}")
            print(f"Average Length of Shortest Path: {np.mean(b_shortest_paths)}")
            print(f"--------------------------------------------------", flush=True)
            print()
            
            b_rtgs = b_advs + b_vals.detach()
            b_advs = (b_advs - torch.mean(b_advs))/ (torch.std(b_advs) + 1e-10)
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
                    prob_ratios = torch.exp(current_log_prob - m_log_probs)

                    # if start == 0 and update == 0:
                    #     print(prob_ratios)

                    surrogate1 = prob_ratios * m_advantage
                    surrogate2 = torch.clamp(prob_ratios, 1-self.clip, 1+ self.clip) * m_advantage
                    
                    # actor loss calculations
                    actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                    self.actor_optim.step()

                    critic_loss = torch.nn.MSELoss()(m_state_values, m_rtgs)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                    self.critic_optim.step()
                            
            self.save_parameters()
    
    def get_batch(self):
        
        batch_obs = []
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

        testrtgs= []
        while True:
            batch_obs.append(obs)
            batch_masks.append(action_mask)
            episode_vals.append(self.critic(obs))
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
            episode_dones.append(done)

            total_timesteps += 1
            if done:
                # print(f"maze len: {self.maze.shortest_path_len}, exitted in: {len(episode_rew)}", flush=True)
                batch_shortest_paths.append(self.maze.shortest_path_len)
                obs, action_mask = self.maze.reset()
                episode_lens.append(len(episode_rew))
                batch_vals.extend(episode_vals)
                batch_advantages.extend(self.get_GAEs(episode_rew, episode_vals, episode_dones))
                testrtgs.append(self.get_rtgs([episode_rew])[0])
                
                episode_rew = []
                episode_vals = []
                episode_dones = []
                
                if total_timesteps > self.batch_size:
                    break
        print()
        print(np.mean(testrtgs))
        batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        batch_act = torch.as_tensor(batch_act, dtype=torch.float32)
        batch_log_probs = torch.as_tensor(batch_log_probs, dtype= torch.float32)
        batch_masks = torch.as_tensor(batch_masks, dtype=torch.bool)
        batch_advantages = torch.as_tensor(batch_advantages, dtype=torch.float32)
        batch_vals = torch.as_tensor(batch_vals, dtype=torch.float32)

        return (batch_obs, batch_act, batch_log_probs, batch_shortest_paths,
                 episode_lens, batch_masks, batch_advantages, batch_vals)
    
    def get_log_probs(self, i, batch_obs, batch_actions, batch_masks):
        batch_moves, batch_marks = batch_actions[:,i,0], batch_actions[:,i,1]
        move_logits, mark_logits = self.actor(batch_obs[:,i,:])
        move_logits.masked_fill_(~batch_masks[:,i,0:4], float('-inf'))
        distribution = torch.distributions.Categorical(logits=move_logits)  
        move_probs = distribution.log_prob(batch_moves)

        mark_logits = mark_logits.squeeze()
        mark_logits.masked_fill_(~batch_masks[:,i,4], float('-inf'))
        mark_prob = torch.sigmoid(mark_logits)
        mark_prob = torch.where(torch.as_tensor(batch_marks, dtype=torch.bool), mark_prob, 1 - mark_prob)
        
        # signal_logits = signal_logits.squeeze()
        # signal_logits.masked_fill_(~batch_masks[:,i,5], float('-inf'))
        # signal_prob = torch.sigmoid(signal_logits)
        # signal_prob = torch.where(torch.as_tensor(batch_signals, dtype=torch.bool), signal_prob, 1 - signal_prob)
        
        # calculating log prob
        log_probs = move_probs + torch.log(mark_prob) #+ torch.log(signal_prob)
        return log_probs

    def get_action(self, obs, action_mask): 
        move_logits, mark_logits = self.actor(obs)
        
        # sampling moves
        adjusted_logits = torch.where(torch.as_tensor(action_mask[0:4], dtype=torch.bool), move_logits, torch.tensor(-float('inf')))
        distribution = torch.distributions.Categorical(logits=adjusted_logits)
        move = distribution.sample()
        
        # sampling mark
        mark_prob = torch.sigmoid(mark_logits) if action_mask[4] == True else torch.tensor([[0]],dtype=torch.float32)
        mark = torch.bernoulli(mark_prob)
        mark_prob = mark_prob if mark == 1 else 1-mark_prob # p(marking) = mark prob, p(not marking) = 1 - p(marking)

        # sampling signal
        # signal_prob = torch.sigmoid(signal_logits) if action_mask[5] == True else torch.tensor([[0]],dtype=torch.float32)
        # signal = torch.bernoulli(signal_prob)
        # signal_prob = signal_prob if signal == 1 else 1-signal_prob
        
        # calculating jointlog probability of all moves
        log_prob = distribution.log_prob(move) + torch.log(mark_prob)

        return [move.item(), mark.item()], log_prob
    
    
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
        print(f'maze size: {self.maze.height} total discounted reward" {rtgs[0]}')
        return rtgs
    
    def save_parameters(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic' : self.critic.state_dict(),
            'actor_optim' : self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()}, MODEL_PATH)
       

    def load_parameters(self):
        if os.path.exists(MODEL_PATH):
            state_dicts = torch.load(MODEL_PATH)
            self.actor.load_state_dict(state_dicts['actor'])
            self.critic.load_state_dict(state_dicts['critic'])
            self.actor_optim.load_state_dict(state_dicts['actor_optim'])
            self.critic_optim.load_state_dict(state_dicts['critic_optim'])
            print("successfuly loaded existing parameters")
            return True
        return False

