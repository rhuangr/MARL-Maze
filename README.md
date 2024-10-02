# Maze
## Projct Description
This project was an attempt to create a *much* smaller-scale version of [Open Ai's Hide and Seek research](https://openai.com/index/emergent-tool-use/) or Google Deepmind's [Capture the flag](https://deepmind.google/discover/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/). Two agents are placed within a randomly generated maze and must find a key to exit the maze together. 

Agents are given 6 possible actions: move in any cardinal direction, stop, or mark a tile with its respective color. As opposed to the researches used as reference, the agents' action space is discrete. Otherwise, the training process would end up taking an eternity on my toaster computer. Initially, the project was intended to support a dynamic amount of agents, but since training time increases exponentially with the number of agents, I had to sacrifice this feature for faster debugging and training. Just like Open Ai's research, this project uses Proximal Policy Optimization to train its agents.

This repository contains all the files necessary to train agents from scratch. Additionally, it contains the file `PPO.pth`, which represents the policy of the agents after training for about 300 epochs.

Shown below is a rendered version of the maze, where flags represent the start and exit. 
> [!NOTE]
> Although **we** see the whole layout of the maze, the agents **do not**.
> The `S` button allows you to switch between the POVs of different agents while visualizing the current policy.
<img width="300" alt="Screenshot 2024-09-06 at 3 32 58 AM" src="https://github.com/user-attachments/assets/77122e8f-0a25-475e-9783-a3198c3c71bc">
<img width="300" alt="Screenshot 2024-09-06 at 3 33 14 AM" src="https://github.com/user-attachments/assets/1e549a23-e4fd-4042-80e1-54a1939cb462">

## Installation
1.  Clone the repository.
2.  Install the numpy, pygame, and pytorch libraries if not yet installed.
3.  Run main.py to see the result of my training.  _You can paste the following commands while in the cloned repository directory 👍  `python main.py`_

## Final notes
- When visualizing the policy, click the following buttons:
  1. `Space` to pause/unpause agent movement
  2. `Q` to generate a new maze and reset agents at the start of the maze
  3. `W` to print all agent observations in the terminal
  4. `E` to take a single step for both agents.
  5. `S` to switch between different POVs.

- If you wish to train agents from scratch, delete `PPO.pth` and uncomment `brain.train()` to start training.
- Hyperparameters of the PPO algorithm can all be modified.
- The layers sizes of the Neural Networks used for both the Actor and Critic can also be modified.

> [!WARNING]
> Please do not change the variable `agent_amount` in the PPO class :(
