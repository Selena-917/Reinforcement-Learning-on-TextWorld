import textworld.gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import re
from collections import defaultdict

class SimpleAgent(textworld.gym.Agent):
    def __init__(self, agent_mode = "random", seed=None):
        self.agent_mode = agent_mode
        self.seed = seed
    
    def get_env_infos(self):
        return textworld.EnvInfos(admissible_commands=True, max_score = True)
    
    def action(self, observations, score, done, infos):
        if self.agent_mode == "random":
            if self.seed:
                np.random.seed(self.seed)
            return np.random.choice(infos["admissible_commands"])
        elif self.agent_mode == "human":
            print(observations)
            return input()
        
class AgentNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device="cuda"):
        super(AgentNetwork, self).__init__()
        self.hidden_size  = hidden_size
        self.device = device
        self.embedding    = torch.nn.Embedding(input_size, hidden_size)
        self.gru_input  = torch.nn.GRU(hidden_size, hidden_size)
        self.gru_command  = torch.nn.GRU(hidden_size, hidden_size)
        self.gru_state    = torch.nn.GRU(hidden_size, hidden_size)
        self.hidden_state = self.init_hidden(1)
        self.linear       = torch.nn.Linear(hidden_size, 1)
        self.linear_command = torch.nn.Linear(hidden_size * 2, 1)
    
    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        
    def forward(self, observations, commands):
        batch_size, num_commands = observations.shape[1], commands.shape[1]

        embed_obs = self.embedding(observations)
        output_encoder, hidden_encoder = self.gru_input(embed_obs)
        output_state, self.hidden_state = self.gru_state(hidden_encoder, self.hidden_state)
        value = self.linear(output_state)

        embed_commands = self.embedding.forward(commands)
        output_commands, hidden_commands = self.gru_command.forward(embed_commands) 
        input_commands = torch.stack([self.hidden_state] * num_commands, 2)
        hidden_commands = torch.stack([hidden_commands] * batch_size, 1) 
        input_commands = torch.cat([input_commands, hidden_commands], dim=-1)

        scores = F.relu(self.linear_command(input_commands)).squeeze(-1)  
        probs = F.softmax(scores, dim=2) 
        index = probs[0].multinomial(num_samples=1).unsqueeze(0)
        
        return scores, index, value
    
class NLPAgent:
    def __init__(self, max_vocab_num=1000, update_freq=10, log_freq=1000, gamma=0.9, lr=1e-5, device="cuda"):
        self.max_vocab_num = max_vocab_num
        self.update_freq = update_freq
        self.log_freq = log_freq
        self.gamma = gamma
        self.lr = lr
        self.device = device
        
        self.idx2word = ["<PAD>", "<UNK>"]
        self.word2idx = {self.idx2word[i]:i for i in range(len(self.idx2word))}
        
        self.agent_model = AgentNetwork(self.max_vocab_num, 128, self.device).to(device)
        self.optimizer = optim.Adam(self.agent_model.parameters(), lr=self.lr)
        
    def test(self):
        self.run_mode = "test"
        self.agent_model.init_hidden(1)
        
    def train(self):
        self.run_mode = "train"
        self.agent_model.init_hidden(1)
        
        self.stats = {"scores": [], "rewards": [], "policy": [], "values": [], "entropy": [], "confidence": []}
        self.infos_per_update = []
        self.last_score = 0
        self.num_step_train = 0
        
    def get_env_infos(self):
        return textworld.EnvInfos(admissible_commands=True, max_score = True, description=True, inventory=True, won=True, lost=True)
    
    def _tokenize_text(self, texts):
        texts = re.sub(r"[^a-zA-Z0-9\- ]", " ", texts)
        words_list = texts.split()
        words_idx = []
        for word in words_list:
            if len(self.word2idx) >= self.max_vocab_num:
                words_idx.append(self.word2idx["<UNK>"])
            else:
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.word2idx)
                words_idx.append(self.word2idx[word])         
            
        return words_idx
    
    def _preprocess_texts(self, texts):
        tokenized_texts = []
        max_len = 0
        for text in texts:
            tokens = self._tokenize_text(text)
            tokenized_texts.append(tokens)
            max_len = max(max_len, len(tokens))

        padded = np.ones((len(tokenized_texts), max_len)) * self.word2idx["<PAD>"]

        for i, text in enumerate(tokenized_texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        padded_tensor = padded_tensor.permute(1, 0) # Not batch first
        return padded_tensor
    
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        r = last_values.data
        for i in reversed(range(len(self.infos_per_update))):
            rewards, _, _, values = self.infos_per_update[i]
            r = rewards + self.gamma * r
            returns.append(r)
            advantages.append(r - values)
            
        return returns[::-1], advantages[::-1]
    
    def action(self, observations, score, done, infos):
        
        agent_input = "{}\n{}\n{}".format(observations, infos["description"], infos["inventory"])
        agent_input_tensor = self._preprocess_texts([agent_input]).to(self.device)
        commands_tensor = self._preprocess_texts(infos["admissible_commands"]).to(self.device)
        
        output, index, value = self.agent_model(agent_input_tensor, commands_tensor)
        action_step = infos["admissible_commands"][index[0]]
        
        # Test Mode, return action_step directly
        if self.run_mode == "test":
            if done:
                self.agent_model.init_hidden(1)
            return action_step
        
        # Train Mode
        self.num_step_train += 1
        
        if self.infos_per_update:
            reward = score - self.last_score 
            self.last_score = score
            if infos["won"]:
                reward += 100
            if infos["lost"]:
                reward -= 100
                
            self.infos_per_update[-1][0] = reward
        
        self.stats["scores"].append(score)
        
        # Update agent_model
        if self.num_step_train % self.update_freq == 0:
            returns, advantages = self._discount_rewards(value)
            
            loss = 0
            for infos_update, r, advantage in zip(self.infos_per_update, returns, advantages):
                reward, indexes, outputs, values = infos_update
                
                advantage        = advantage.detach()
                probs            = F.softmax(outputs, dim=2)
                log_probs        = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes)
                policy_loss      = (-log_action_probs * advantage).sum()
                value_loss       = (.5 * (values - r) ** 2.).sum()
                entropy     = (-probs * log_probs).sum()
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy
                
                self.stats["rewards"].append(reward)
                self.stats["policy"].append(policy_loss.item())
                self.stats["values"].append(value_loss.item())
                self.stats["entropy"].append(entropy.item())
                self.stats["confidence"].append(torch.exp(log_action_probs).item())
            
            if self.num_step_train % self.log_freq == 0:
                print("Total step: {:6d}  reward: {:3.3f}  value: {:3.3f}  entropy: {:3.3f}  max_score: {:3d}  num_vocab: {}".format(self.num_step_train, np.mean(self.stats["rewards"]), np.mean(self.stats["values"]), np.mean(self.stats["entropy"]), np.max(self.stats["scores"]), len(self.idx2word)))
                self.stats = {"scores": [], "rewards": [], "policy": [], "values": [], "entropy": [], "confidence": []}
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            self.infos_per_update = []
            self.agent_model.init_hidden(1)
        else:
            self.infos_per_update.append([None, index, output, value])
        
        if done:
            self.last_score = 0 
        
        return action_step