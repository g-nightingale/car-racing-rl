# CarRacing-v0

In this environment, the goal of the agent is to maximize points by successfully navigating around a randomly generated racetrack. The agent loses points for each frame of game-play elapsed, and earns points for visiting previously unvisited track tiles.

## 0. Environment Setup
        conda create -c conda-forge -n gymenv swig pip  
        conda activate gymenv  
        pip install gym==0.19   # Use 0.17.3 to render in the notebook, however this has memory leak issues because of xvfb
        pip install Box2D gym
        pip install gym[all]

        pip install tensorflow
        pip install matplotlib

        pip install gym pyvirtualdisplay   
        sudo apt-get install -y xvfb python-opengl ffmpeg  

        # The following steps are so that I can access the new env in Azure ML notebooks
        conda install ipykernel
        python -m ipykernel install --user --name gymenv --display-name "Python (gymenv)"


## 1. Problem definition
### 1.1 Environment details
At the beginning of each episode, a race track is randomly generated consisting of between 200 and 300 track tiles. An episode finishes when either all of the track tiles have been visited, the car leaves the environment perimeter or 1,000 frames have elapsed. The game runs at 50 frames per second, therefore each episode lasts up to 20 seconds.

The environment state is represented as a 96x96x3 RGB image of the current game screen, displaying an overheard view of the car and track, along with current score and telemetry readouts of speed, ABS sensors, steering position, and gyroscope. As the game progresses, the car remains in a fixed position whilst the game screen is re-orientated based on the movements of the car. 

**Figure 1. State representation**
![alt text](/images/state.png "Title")

### 1.2 Rewards and solving condition
The agent receives a reward of -0.1 for each frame elapsed and a reward of +1000/N for each new track tile visited, where N is the total number of tiles in the track. The agent receives a score of -100 if it ventures far beyond the racetrack and leaves the wider environment perimeter. To achieve maximum rewards, the agent must navigate quickly around the track without missing any track tiles.

The environment is solved when the agent achieves an average score of at least 900 points over 100 episodes. 

### 1.3 Action space and transition dynamics
The agent interacts with the environment through throttle, steering and braking inputs. These actions are continuous and have the following ranges:
- Throttle (t) in the range $[0.0, 1.0]$, where 0.0 is no throttle, and 1.0 is full throttle.
- Steering (s) in the range $[-1.0, 1.0]$, where -1.0 is full left turn and 1.0 is full right turn.
- Braking (b) in the range $[0.0, 1.0]$, where 0.0 is no braking and 1.0 is full braking.

Actions are supplied to the environment in the vector form $[t, s, b]$. After taking an action, the game state is updated according to an unobserved, internal physics model within the environment. The next state is computed based on the agents current speed and trajectory in combination with the new action inputs, and is returned in the form of an updated game screen along with rewards earned.

## 2. Background
There are up to $256^{3\times96^2}$ possible states for the CarRacing-v0 environment, as each 96x96 game screen consists of three RGB layers that can each have 256 unique values. In reality, the actual state space is much smaller (the game uses only a handful of colours for example), however it is prohibitively large to consider tabular based reinforcement learning approaches.

We therefore look towards methods which use function-approximation to estimate q-values such as Deep Q-Networks.


## 3. Method
### 3.1 State representation
In a similar approach to Mnih et al. (2013) we make several modifications to the state representation:
- Firstly, we convert the RGB image to grayscale to reduce dimensionality.
- Secondly, we censor the score meter in the bottom left of the screen by setting the pixels in this region to zero. For some approaches we also trial removing further track details such as grass and track tile shading.
- Thirdly, we stack N grayscale game states to form a 96x96xN multidimensional vector, which is used as input in the CNN. We trial different intervals between consecutive game screens.Thirdly, we stack N grayscale game states to form a 96x96xN multidimensional vector, which is used as input in the CNN. We trial different intervals between consecutive game screens.

**Figure 2: Image preprocessing**  
![alt text](/images/preprocessing1.png "Title")

### 3.2 Action discretisation
We reduce the continuous action space to the following five discrete actions:
1. Full throttle: [0.0, 1.0, 0.0]
2. Full left turn: [-1.0, 0.0, 0.0]
3. Full right turn: [1.0, 0.0, 0.0]
4. 80% brakes: [0.0, 0.0, 0.8]
5. No action: [0.0, 0.0, 0.0]

### 3.3 Initial frames of environment
The first 50 frames of an environment depict the camera zooming into the track. During these frames, the car does not move, however throttle can be accumulated. We apply full throttle during these initial 50 frames, to propel the car forward once the game starts. We exclude these initial 50 frames from the replay buffer as they are not representative of states observed during normal game play.

### 3.4 Deep Q-Network
Deep Q-Networks (DQNs) were first introduced by Mnih et al. (2013), where the authors trained RL agents to play a range of Atari video games using only knowledge of the current game screen, available controller actions, and resulting rewards and next states from taking said actions.

DQNs use a neural network as a non-linear function approximator, where the network takes the representation of a state $s$ as input (in our case, the pre-processed game screen) and outputs an estimate of the action-value $q$ for each available action. Previous attempts to use neural networks for reinforcement learning had largely failed due to stability issues during training. Mnih et al. (2013) address these issues through the use of separate target networks and experience replay.

Using this approach, q-value updates are given as: \\
$Q(s,a;\theta) \rightarrow r + max_{a}Q(s', a;\theta))$

We trial two configurations of the DQN:
- DQN 1: A DQN closely resembling that of Mnih et al (2013) in terms of CNN architecture and hyper-parameters used.  
- DQN 2: A modified DQN with two key changes. Firstly, instead of applying an action to a single frame, we apply the action to four consecutive frames, and accumulate the rewards over these four frames. Secondly, instead of stacking consecutive frames to form the state-representation, we use every fourth frame. The hypothesis is that consecutive frames are very similar and may not provide sufficient variation in information to facilitate effective learning.

### 3.5 Double Deep Q-Network
Hasselt, Guez and Silver (2015) demonstrate that the DQN algorithm produces overoptimistic estimates of action-values, causing instability during learning and producing sub-optimal policies. 

As a solution, they propose the use of a double DQN network that leverages the existing architecture of the DQN without requiring additional networks or parameters. Specifically, the action-value model is used to select the optimal action, whilst the target model is used to estimate the action-value, effectively removing the upward bias in action-value estimates. Hasselt, Guez and Silver (2015) find that double DQN produces better policies and obtains state of the art results on Atari 2600 games.

Following this logic, our update rule changes to: \\
$Q(s,a;\theta) \rightarrow r + \gamma max Q(s', argmaxQ(s', a';\theta);\theta')$

We trial two configurations of double DQN:
- Double DQN 1: We trial a double DQN with the same modifications as DQN 2, to understand the impact of bias reduction in this environment.
- Double DQN 2: We add learning rate decay and $L2$ regularisation to the CNN as per Rodrigues and Vieira (2020) in order to further improve learning stability and convergence. We also increase the capacity of the CNN from 256 to 512 neurons in the final dense layer and add additional image pre-processing steps, removing track details and reducing the number of shades used.

## 4. Results
We consider two sets of results. Firstly, we examine agent performance during training to understand terminal scores and convergence properties. Secondly, we examine the performance of trained agents in an inference environment over 100 episodes, where agents execute their learned policies without any randomly chosen actions.

### 4.1 Training results
Results from training are presented in Table 1. Our DQN 1 agent achieves an average score of 213 points over 100 episodes. DQN 2 achieves an average score of 724. This result is consistent with our hypothesis that repeating actions over multiple frames and increasing the inter-temporal gap in state representation improves agent learning.

The double DQN 1 agent achieves an average training score of 807 points - demonstrating the benefits bias reduction in the action-value estimates. Double DQN 2 improves these results, achieving an average training score of 875. Using learning rate decay and regularisation, the volatility of scores reduces as training matures and our agent converges towards a good policy.

**Table 1: Training results**
| Algorithm | Average score (100) |
|-----------|---------------------|
| DQN 1     | 213                 |
| DQN 2     | 724                 |
| Double DQN 1 | 807              |
| Double DQN 2 | 867              |

### 4.2 Inference results
Inference results are presented in Table 2.

DQN 1 also performs poorly, achieving an average score of only 122 points - significantly under-performing training results. This is potentially due to the higher weighting assigned to moving forward when taking a random action, thus causing the training score to be inflated. DQN 2 improves performance significantly - scoring 690 points on average and achieving near human levels of performance.

Double DQN 1 scores 782 points, surpassing our human level benchmark. Double DQN 2 is our best performing agent, reaching an average score of 903 $\pm$ 26, which successfully solves this environment. We note the success of this agent appears to be correlated with track configuration, as this agent tends to perform better on tracks with fewer sharp corners

**Table 2: Inference results**
| Algorithm | Average score (100) |
|-----------|---------------------|
| DQN 1     | 122 $\pm$ 113       |
| DQN 2     | 690 $\pm$ 250       |
| Double DQN 1 | 782 $\pm$ 223    |
| Double DQN 2 | 903 $\pm$ 26     |

### References
van Hasselt, H., Guez, A. and Silver, D., 2015. Deep Reinforcement Learning with Double Q-learning (v.3). arXiv:1509.06461 [cs] [Online]. Available from: http://arxiv.org/abs/1509.06461 [Accessed 23 April 2022].

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602 [cs] [Online]. Available from: http://arxiv.org/abs/1312.5602 [Accessed 23 April 2022].

Rodrigues, P. and Vieira, S., 2020. Optimizing Agent Training with Deep Q-Learning on a Self-Driving
Reinforcement Learning Environment. 2020 IEEE Symposium Series on Computational Intelligence (SSCI),
[Online], 2020 IEEE Symposium Series on Computational Intelligence (SSCI). pp.745–752. Available from:
https://doi.org/10.1109/SSCI47803.2020.9308525.
