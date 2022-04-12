import numpy as np
import matplotlib.pyplot as plt


def image_processing(s, img_len, crop_dims=None):
    """Crop image, convert to grayscale and reshape."""
    
    if crop_dims is not None:
        s = s[crop_dims[0], crop_dims[1], :]
    s = np.dot(s[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.int16)
    s = np.reshape(s, (1, img_len, img_len))  

    return s


def get_stacked_state(grayscale_state_buffer, img_dim):
    """Returns a stacked image from the grayscale buffer."""

    # Create image from grayscale buffer
    img = np.zeros(img_dim)
    for i in range(img_dim[2]):
        img[:, :, i] = grayscale_state_buffer[i]
    img = np.reshape(img, (1,) + img_dim)

    # Remove first image from buffer
    grayscale_state_buffer.popleft()
    return img


def get_stacked_state_live(grayscale_state_buffer, img_dim, skip_frames):
    """Returns a stacked image from the grayscale buffer.
    
        Used in live running of DQN agent.
    """

    # Create image from grayscale buffer
    img = np.zeros(img_dim)
    for i in range(img_dim[2]):
        img[:, :, i] = grayscale_state_buffer[i * skip_frames]
    img = np.reshape(img, (1,) + img_dim)

    # Remove first image from buffer
    grayscale_state_buffer.popleft()
    return img


def plot_rewards(episode_rewards):
    """Plot episode rewards."""
    plt.figure(figsize=(8, 6))
    plt.title('DQN Agent')
    plt.plot(episode_rewards, label='Episode reward', alpha=0.5)
    plt.plot([np.mean(episode_rewards[::-1][i:i+100]) for i in range(len(episode_rewards))][::-1], label='Average reward (last 100 episodes)')
    plt.ylim((-50, 800))
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()


def plot_reward_comparison(agent_episode_rewards, labels, ylim=(-50, 500)):
    """Plot episode rewards."""
    plt.figure(figsize=(8, 6))
    plt.title('DQN Agent Comparison - Average reward (last 100 episodes)')
    for i, episode_rewards in enumerate(agent_episode_rewards):
        plt.plot([np.mean(episode_rewards[::-1][i:i+100]) for i in range(len(episode_rewards))][::-1], label=labels[i])
    plt.ylim(ylim)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()
