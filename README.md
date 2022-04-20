# Reinforcement Learning: Second Assignment

## 1. Problem definition
*A clear, precise and concise description of your chosen problem, including the states, actions, transition dynamics, and the reward function. You will lose marks for an unclear, incorrect, or incomplete problem definition.*

We have chosen the CarRacing-v0 enviroment from the OpenAI gym suite of reinforcement learning environments.

This is a top down car racing environment, where the goal of the agent is to maximize points earned by sucessfully navigating around a randomly generated racetrack.

We have chosen this problem as it provides unique challenges given the image state representation and continuous action space, and also enables the application of a wide range of reinforcement learning methods.

### 1.1 Track generation and episode details 
A track is randomly generated each time the environment is reset. Each track consists of between 200 and 300 tiles and each episode consists of 1,000 frames. 

The game runs at 50 frames per second, therefore each episode lasts 20 seconds, unless terminated early.

### 1.2 State representation
Each state is represented as a 96x96 RGB image of the current gamescreen. The gamescreen displays the car and track, along with current score and telemetry readouts of true speed, ABS sensors, steering wheel position, and gyroscope.

### 1.3 Rewards
The agent receives a reward of -0.1 for each frame elapsed and a reward of +1000/N for each new track tile visited, where N is the total number of tiles in the track. 

According to the OpenAI Gym documentation, the agent receives a score of -100 for leaving the wider track boundary.

### 1.4 Episode termination
The official OpenAI gym documentation states that an episode finishes when either all of the tiles are visited or the car leaves the playfield - which is the wider boundary beyond the race track and grass areas.

We also note that an episode finishes after 1,000 frames have elapsed.

Additionally, for some algorithm implementations we add additional termination criteria.

### 1.5 Action space
The action space in the CarRacing-v0 environment is continuous and consists of 
- Acceleration (a) in the range [0.0, 1.0], where 0.0 is no throttle, and 1.0 is full throttle.
- Steering (s) in the range [-1.0, 1.0], where -1.0 is full left turn and 1.0 is full right turn.
- Braking (b) [0.0, 1.0], where 0.0 is no braking and 1.0 is full braking.

Actions are supplied to the environment in the vector form [a, s, b].

### 1.6 Solved conditions
As per the OpenAI gym documentation, the environment is solved when the agent achieves an average score of at least 900 points over 100 episodes.

## 2. Background
*A discussion of reinforcement learning methods that may be effective at solving your chosen problem, their strengths and weaknesses for your chosen problem, and any existing results in the scientific literature (or publicly available online) on your chosen problem or similar problems.*

As this is a continuous control task with an image based state representation we draw heavily from the pioneering work of Mnih et al. (2013) and subsequent research into reinforcement learning methods applied to Atari games.

Specifically, we focus on methods used for tasks which use image based state representations.

TODO

From our research, this specific environment has not been extensively studied in the scientific literature.


## 3. Method
*A description of the method(s) used to solve your chosen problem, an explanation of how these methods work (in your own words), and an explanation of why you chose these specific methods.*

### 3.1 Pre-processing
The following pre-processing steps have been applied to all of the algorithms trialled.

#### 3.1.1 State representation
In a similar approach to Mnih et al. (2013) we make several modifications to the state representation:
- Firstly, we convert the RGB image to grayscale as colour is not important for learning in this environment.
- Secondly, we censor the score meter in the bottom left of the screen by setting the pixels in this region to zero. The current score should not influence the actions taken by an agent and we observe strange behaviour in agents once they attain near maximal scores (discussed further in results).
- Thirdly, we stack N grayscale game states to form a 96x96xN multidimensional vector, which is used as input in the CNN. We trial different intervals between consecutive game screens.

**Figure 1: Image preprocessing**
![alt text](/images/preprocessing1.png "Title")

**Figure 2: Stacking of state frames**
![alt text](/images/preprocessing2.png "Title")

#### 3.1.2 Action discretisation
We reduce the continuous action space to the following five discrete actions:
1. Full throttle: [0.0, 1.0, 0.0]
2. Full left turn: [-1.0, 0.0, 0.0]
3. Full right turn: [1.0, 0.0, 0.0]
4. 80% brakes: [0.0, 0.0, 0.8]
5. No action: [0.0, 0.0, 0.0]

#### 3.1.3 Initial frames of environment
The first 50 frames of an environment depict the camera zooming into the track. During these frames, the car does not move, however throttle can be accumulated. We apply full throttle during these initial 50 frames, to propel the car forward once the game starts. 

These initial 50 frames are excluded from the replay buffer as they are not representative of states observed during "real" gameplay.

**Figure 3: Initial frames**
![alt text](/images/initial_frames.png "Title")  

### 3.1 DQN
Deep Q-Networks were first introduced in the seminal paper by Mnih et al. (2013), where RL agents were successfully trained to play a range of Atari video games using only knowledge of the current screen state, available controller actions, and resulting rewards from taking actions.

Earlier attempts at using neural networks had failed, however several key innovations introediced by Mnih et al. (2013) including experience replay, huber loss function and fixed target networks.

DQNs leverage neural networks as a function approximator, where the input is a state representation and the output is an estimate action-values. 

Action values are estimated as ... 

As per Mnih et al (2013), a convolutional neural network is used to estimate action-values. Mnih

We trial two configurations of the DQN:
1. DQN 1: A DQN closely resembling that of Mnih et al (2013) in terms of CNN structure and hyperparameters used.
2. DQN 2: A modified DQN where we capture every fourth frame to form the state representation instead of consecutive frames. The rationale for this change is that using consecutive frames may not provide much variation in information to the agent as consecutive frames are extremely similar. Increasing the temporal differences between frames should provide more information of the recent trajectory of the agent, and therefore aid learning.


### 3.2 Double-DQN
Hasselt, Guez and Silver (2015) propose the use of a double-DQN network to overcome shortcomings of the original DQN algorithm. Specifically, by using the argmax function in selecting action-values, action value estimates are always biased to be larger than their true vales. This can cause issues because...

As a remedy, Hasselt, Guez and Silver (2015) prescribe the use of two separate networks to ensure that q-value estimates are unbiased. The simplest approach being

We adopt the simplest implementation, where ...
We also explore the use of 

We trial three configurations of DDQN:
1. DDQN 1: with every fourth frame sampled (as per our )
2. DDQN 2: with soft-parameter updates
    - Drawing inspiration from Lillipcrap et al. (2019), we leverage soft-parameter updates. Instead of refreshing the weights of the target model every 5,000 steps, we copy a small portion of the main model weights at each step using the following update rule. Ex expect this change to increase the speed of learning and reduce any sharp spkies dirven by instantaeous model weight changes.
3. DDQN 3: with soft-parameter updates
    - From the results of DDQN 2 we observe strange results of the agent. As our agent learns a 

### 3.3 Actor-critic 
TODO

### 3.4 DDPG 
TODO

### 3.5 Policy gradients
TODO

## 4. Results
*A presentation of your results, showing how quickly and how well your agent(s) learn (i.e., improve their policies). Include informative baselines for comparison (e.g. the best possible performance, the performance of an average human, or the performance of an agent that selects actions randomly).*

### 4.1 Benchmarks
As a benchmarks, we use a human score captured over 30 trials.

The enviornment



### 4.2 Training comparison

**Figure 4: learning curves**
![alt text](/images/results.png "Title")

### 4.3 Inference comparison
TODO  
**Figure 5: inference comparison**

## 5. Discussion
*An evaluation of how well you solved your chosen problem.*





## 6. Future work
*A discussion of potential future work you would complete if you had more time.*

## 7. Personal experience
*A discussion of your personal experience with the project, such as difficulties or pleasant surprises you encountered while completing it.*

- Environment setup
    - TODO

- Compute resources
    - The compute resources available to each team member were wide-ranging. Some team members had access to only basic compute on a local laptop, whilst other team members had access high peformance GPUs and scalable cloud compute. These differences had a direct and significant impact on the implementations and results achieved. For example, machines with more RAM were able to store a much larger replay buffer - a factor which we find to be highly correlated with agent performance. The disparate availability of compute resources creates a conflict between finding implementations that can solve the environment vs implementations which are directly comparable.

- Training time
    - The amount of training time required in this environment was significant, even on machines with substantial compute resource. Often it would be hours into training before an agent would start making progress in learning. This meant that 

- Personal commitments
    - TODO

## 7. References



## 8. Appendices
### 8.1 DQN structure
For DQN and Double DQN algoritrhms, we use a Convolutional Neural Network with an architecture similar to that of Mnih et al. (2013) with the addition of an extra convolutional layer in order to reduce the number of model parameters.

The model consists of three convolutional layers with the following attributes:
- Number of filters: 32, 64, 64
- Filter sizes: 8, 4, 3 
- Stride length: 4, 2, 1

The Relu activation function has been used for each convolutional layer. 
Following on from the convolutional layers, outputs are flattened and passed to a 256 dimensional dense layer with a Relu activation function. Finally, these outputs are passed to a 5-dimensional dense layer with a linear activation function.

### 8.2 Trial configurations

### 8.2 CNN analysis






