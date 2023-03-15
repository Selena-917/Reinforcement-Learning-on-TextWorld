import textworld.gym
import gym
import glob
import os
import numpy as np
import agents
import torch

def play_game(agent, game_path, max_steps=100, num_episodes=10, seed=None):
    
    if seed:
        torch.manual_seed(seed)
    
    if os.path.isdir(game_path):
        game_files = glob.glob(os.path.join(game_path, "*.z8"))
    else: game_files = [game_path]
    
    request_info = agent.get_env_infos()
    env_id = textworld.gym.register_games(game_files, request_infos=request_info, max_episode_steps=max_steps)
    env = gym.make(env_id)
    
    num_moves = []
    scores = []
    normalized_scores = []
    for episode in range(num_episodes):
        observations, infos = env.reset()
        
        num_moves_per_episode = 0
        score = 0
        done = False
        while not done:
            agent_command = agent.action(observations, score, done, infos)
            observations, score, done, infos = env.step(agent_command)
            num_moves_per_episode += 1
        
        agent.action(observations, score, done, infos)
        
        num_moves.append(num_moves_per_episode)
        scores.append(score)
        normalized_scores.append(score / infos["max_score"])
    env.close()
    
    if os.path.isdir(game_path):
        print("Average steps used: {:.2f}; Average normalized score: {:.2f}/1".format(np.mean(num_moves), np.mean(normalized_scores)))
    else:
        print("Average steps used: {:.2f}; Average score: {:.2f}/{}".format(np.mean(num_moves), np.mean(scores), infos["max_score"]))
      
        
if __name__ == '__main__': 
    print("Random Agent (do random action) --------------------------------------")
    random_agent = agents.SimpleAgent("random")
    play_game(random_agent, "./tw_games/tw-rewardsDense_goalDetailed.z8", 100, 10) 
    
    print("----------------------------------------------------------------------")
    
    print("NLP Agent (train the model) ------------------------------------------")
    
    nlp_agent = agents.NLPAgent(lr=0.00003)
    nlp_agent.train()
    play_game(nlp_agent, "./tw_games/tw-rewardsDense_goalDetailed.z8", 100, 500) 
    os.makedirs('models', exist_ok=True)
    torch.save(nlp_agent, 'models/nlp_agent_trained.pt')
    
    print("NLP Agent (test the model) ------------------------------------------")
    nlp_agent.test()
    play_game(nlp_agent, "./tw_games/tw-rewardsDense_goalDetailed.z8", 100, 10) 