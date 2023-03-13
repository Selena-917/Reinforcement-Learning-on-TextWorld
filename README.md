# Reinforcement-Learning-on-TextWorld

To train the NLPAgent to play a TextWorld game:

First make a textworld game, such as:

```
tw-make tw-simple --rewards dense --goal detailed --seed 18 --test --silent -f --output tw_games/tw-rewardsDense_goalDetailed.z8
```

Then run

```
python play_game.py
```