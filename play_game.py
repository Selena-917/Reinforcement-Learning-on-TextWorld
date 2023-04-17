import textworld.gym
import gym
import glob
import os
import numpy as np
import agents
import torch
import time
import argparse

import sys

def play_game(agent, game_path, max_steps=100, num_episodes=10, seed=None):
    
    if seed:
        torch.manual_seed(seed)
    
    if os.path.isdir(game_path):
        game_files = glob.glob(os.path.join(game_path, "*.z8"))
    else: game_files = [game_path]
    print("game_files:", game_files)
    
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
        
        
def play_game_dqn(agent, game_path, max_steps=100, num_episodes=10, seed=None):
    
    if seed:
        torch.manual_seed(seed)
    
    if os.path.isdir(game_path):
        game_files = glob.glob(os.path.join(game_path, "*.z8"))
    else: game_files = [game_path]
    print("game_files:", game_files)
    
    request_info = agent.get_env_infos()
    env_id = textworld.gym.register_games(game_files, request_infos=request_info, max_episode_steps=max_steps)
    env = gym.make(env_id)

    num_moves = []
    scores = []
    normalized_scores = []
    epsilon = 1.0
    epsilon_decay = 0.995
    for episode in range(num_episodes):
        observations, infos = env.reset()
        
        agent_input = "{}\n{}\n{}".format(observations, infos["description"], infos["inventory"])
        agent_input_tensor = agent._preprocess_texts([agent_input])
        commands_tensor = agent._preprocess_texts(infos["admissible_commands"])
            
        num_moves_per_episode = 0
        last_score = 0
        done = False
        while not done:
            agent_command = agent.epsilon_greedy_action_selection(epsilon, agent_input_tensor, commands_tensor, infos, done)
            next_observations, score, done, infos = env.step(agent_command)
            
            next_agent_input = "{}\n{}\n{}".format(next_observations, infos["description"], infos["inventory"])
            next_agent_input_tensor = agent._preprocess_texts([next_agent_input])
            next_commands_tensor = agent._preprocess_texts(infos["admissible_commands"])
            
            if agent.run_mode == "train":
                agent.memory.push(agent_input_tensor, commands_tensor, agent_command, next_agent_input_tensor, next_commands_tensor, score-last_score)
                agent.replay(32)
                
            agent_input_tensor = next_agent_input_tensor
            commands_tensor = next_commands_tensor
            last_score = score
                
            num_moves_per_episode += 1
        
        if agent.run_mode == "train":
            epsilon *= epsilon_decay
            agent.update_model_handler(episode, 10)
        
        num_moves.append(num_moves_per_episode)
        scores.append(score)
        normalized_scores.append(score / infos["max_score"])
    env.close()
    
    if os.path.isdir(game_path):
        print("Average steps used: {:.2f}; Average normalized score: {:.2f}/1".format(np.mean(num_moves), np.mean(normalized_scores)))
    else:
        print("Average steps used: {:.2f}; Average score: {:.2f}/{}".format(np.mean(num_moves), np.mean(scores), infos["max_score"]))
      

def main(args):
    # Play the game by yourself
    if args.play_method == "human":
        print("Human Agent (play by yourself) --------------------------------------")
        random_agent = agents.SimpleAgent("human")
        play_game(random_agent, args.single_gamefile, 100, 1)
    
    # Train the agent to play a single game
    elif args.play_method == "single":
        print("Random Agent (do random action) --------------------------------------")
        random_agent = agents.SimpleAgent("random")
        play_game(random_agent, args.single_gamefile, 100, 10) 
        
        print("----------------------------------------------------------------------")
        
        save_model_name = args.single_gamefile[args.single_gamefile.rfind("/")+1:args.single_gamefile.rfind(".")]
        if args.model_type == "gru":
        
            print("\nNLP Agent GRU (train the model) ------------------------------------------\n")
            
            nlp_agent_gru = agents.NLPAgent(model_type="gru", lr=0.0005) # May need to tune this lr
            print("NLP Agent GRU (acc before training) --------------------------------------")
            nlp_agent_gru.test()
            if not args.dqn:
                play_game(nlp_agent_gru, args.single_gamefile, 100, 10) 
            else:
                play_game_dqn(nlp_agent_gru, args.single_gamefile, 100, 10) 
            
            start_time = time.time()
            print("\nNLP Agent GRU (start training) -------------------------------------------")
            nlp_agent_gru.train()
            if not args.dqn:
                play_game(nlp_agent_gru, args.single_gamefile, 100, num_episodes=200) # May need to tune num_episodes
            else:
                play_game_dqn(nlp_agent_gru, args.single_gamefile, 100, num_episodes=200)
                
            os.makedirs('checkpoints', exist_ok=True) 
            torch.save(nlp_agent_gru, "checkpoints/GRU-"+save_model_name+".pt")
            print("Total training time:", time.time()-start_time)
            
            print("\nNLP Agent GRU (test the model) ------------------------------------------")
            nlp_agent_gru.test()
            if not args.dqn:
                play_game(nlp_agent_gru, args.single_gamefile, 100, 10) 
            else:
                play_game_dqn(nlp_agent_gru, args.single_gamefile, 100, 10) 
            
            print("----------------------------------------------------------------------")
        
        elif args.model_type == "gpt-2":
            print("\nNLP Agent GPT (train the model) ------------------------------------------\n")
            nlp_agent_gpt = agents.NLPAgent(model_type="gpt-2", lr=0.0001)
            print("NLP Agent GPT (acc before training) --------------------------------------")
            nlp_agent_gpt.test()
            play_game(nlp_agent_gpt, args.single_gamefile, 100, 10) 
            
            start_time = time.time()
            print("\nNLP Agent GPT (start training) -------------------------------------------")
            nlp_agent_gpt.train()
            play_game(nlp_agent_gpt, args.single_gamefile, 100, num_episodes=300) 
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(nlp_agent_gpt, "checkpoints/GPT-"+save_model_name+".pt")
            print("Total training time:", time.time()-start_time)
            
            print("\nNLP Agent GPT (test the model) ------------------------------------------")
            nlp_agent_gpt.test()
            play_game(nlp_agent_gpt, args.single_gamefile, 100, 10) 
            
            print("----------------------------------------------------------------------")

        elif args.model_type == "bert_gru": 
            print("\nNLP Agent BERT GRU (train the model) ------------------------------------------\n")
            
            nlp_agent_bert_gru = agents.NLPAgent(model_type="bert_gru", lr=0.00005)
            print("NLP Agent BERT GRU (acc before training) --------------------------------------")
            nlp_agent_bert_gru.test()
            play_game(nlp_agent_bert_gru, args.single_gamefile, 100, 10) 
            
            start_time = time.time()
            print("\nNLP Agent BERT GRU (start training) -------------------------------------------")
            nlp_agent_bert_gru.train()
            play_game(nlp_agent_bert_gru, args.single_gamefile, 100, num_episodes=100) 
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(nlp_agent_bert_gru, "checkpoints/BERT-GRU-"+save_model_name+".pt")
            print("Total training time:", time.time()-start_time)
            
            print("\nNLP Agent BERT GRU (test the model) ------------------------------------------")
            nlp_agent_bert_gru.test()
            play_game(nlp_agent_bert_gru, args.single_gamefile, 100, 10) 
            
            print("----------------------------------------------------------------------")
    
    # Train the agent to play multiple games
    elif args.play_method == "multiple":
    
        print("Training on multiple games------------------------------------------")
        nlp_agent_gru = agents.NLPAgent(model_type="gru", lr=0.00005)

        nlp_agent_gru.train() 
        start_time = time.time()
        play_game(nlp_agent_gru, args.multiple_games_folder, 100, num_episodes=100 * 10)  # 100 games, each game will be played 10 episodes
        print("Total training time:", time.time()-start_time)

        os.makedirs('checkpoints', exist_ok=True)
        torch.save(nlp_agent_gru, 'checkpoints/agent_trained_on_multiple_games.pt')
        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model_type', type=str, default="gru", help='choose from [gru, gpt-2, bert_gru]')
    parser.add_argument('--play_method', type=str, default="single", help='choose from [human, single, multiple]')
    parser.add_argument('--single_gamefile', type=str, default="./tw_games/tw-rewardsDense_goalDetailed.z8", help='File name of the single game')
    parser.add_argument('--multiple_games_folder', type=str, default="tw-simple_games/", help='Name of the folder containing multiple games')
    parser.add_argument('--dqn', type=bool, default=False, help='Whether to use DQN or not')
    
    args = parser.parse_args()
    
    print(args)
    main(args)
    
    # nlp_agent_gru = torch.load('checkpoints/GRU-tw-coin_collector_level-7.pt')
    # play_game(nlp_agent_gru, args.single_gamefile, 100, 10) 