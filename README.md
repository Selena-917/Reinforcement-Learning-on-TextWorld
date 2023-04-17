# Reinforcement-Learning-on-TextWorld

To train the NLPAgent to play a TextWorld game:

First make a textworld game, such as:

```
tw-make tw-simple --rewards dense --goal detailed --seed 1 --test --silent -f --output tw_games/tw-rewardsDense_goalDetailed.z8
```
Basically, there are four types of games in TextWorld which are tw-simple, tw-coin_collector, tw-treasure_hunter, and tw-cooking games. You can check this [documentation](https://textworld.readthedocs.io/en/stable/) for more details. Also the notebook "make_tw_games.ipynb" contains codes to make these four types of games. 

<br/>

Then run

```
python play_game.py --model_type gru --play_method single --single_gamefile /path/to/gamefile
```

<br/>

Some results we got is shown in the below table. (Note: the global seed for creating game files is 1)

|              Game               | Score (GRU) | Score (GPT) | Running Time (GRU) | Running Time (GPT) |
|:-------------------------------:|:-----------:|:-----------:|:------------------:|:------------------:|
| tw-rewardsDense_goalDetailed    |    6.9/8    |    6.8/8    |     212.98 s       |      1313.84 s     |
| tw-rewardsBalanced_goalDetailed |    0.8/3    |    0.8/3    |     416.87 s       |      1318.39 s     |
| tw-rewardsSparse_goalDetailed   |    0.0/1    |    0.0/1    |     212.82 s       |      1316.20 s     |
| tw-rewardsDense_goalBrief       |    8.0/8    |    6.9/8    |     421.40 s       |      1190.60 s     |
| tw-rewardsBalanced_goalBrief    |    0.5/3    |    0.9/3    |     400.49 s       |      1202.86 s     |
| tw-rewardsSparse_goalBrief      |    0.0/1    |    0.0/1    |     408.45 s       |      1203.70 s     |
| tw-rewardsDense_goalNone        |    6.1/8    |    7.1/8    |     412.68 s       |      1328.44 s     |
| tw-rewardsBalanced_goalNone     |    0.4/3    |    0.6/3    |     412.21 s       |      1306.45 s     |
| tw-rewardsSparse_goalNone       |    0.0/1    |    0.0/1    |     399.63 s       |      1328.71 s     |