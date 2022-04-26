import numpy as np
import matplotlib.pyplot as plt


def image_processing_old(s, img_len):
    """Convert to grayscale and reshape."""
    
    s = np.dot(s[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    s = np.reshape(s, (1, img_len, img_len))  
    s[0, 86:, :16] = 0.0

    return s

def get_stacked_state_live_old(grayscale_state_buffer, img_dim, frame_interval):
    """Returns a stacked image from the grayscale buffer.
    
        Used in live running of DQN agent.
    """

    # Create image from grayscale buffer
    img = np.zeros(img_dim)
    for i in range(img_dim[2]):
        img[:, :, i] = grayscale_state_buffer[i * frame_interval]
    img = np.reshape(img, (1,) + img_dim)

    # Remove first image from buffer
    grayscale_state_buffer.pop(0)
    return img

def image_processing_new(s, img_len):
    """Convert to grayscale, simplify image and reshape."""
    
    # Convert to grayscale
    s = np.dot(s[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Simplify track part of image
    s[:85, :][s[:85, :]==106.0]=101.0
    s[:85, :][s[:85, :]==104.0]=101.0
    s[:85, :][s[:85, :]==161.0]=255.0
    s[:85, :][s[:85, :]==177.0]=255.0
    s[:85, :][s[:85, :]==254.0]=255.0
    s[:85, :][s[:85, :]==76.0]=255.0

    # Simplify bottom bar of image
    s[86:, :12] = 0.0
    s[86:, :][s[86:, :]==254.0]=255.0
    s[86:, :][s[86:, :]==29.0]=255.0
    s[86:, :][s[86:, :]==44.0]=255.0
    s[86:, :][s[86:, :]==149.0]=255.0
    s[86:, :][s[86:, :]==76.0]=255.0

    # Reshape
    s = np.reshape(s, (1, img_len, img_len))  

    return s


def get_stacked_state_live_new(grayscale_state_buffer, img_dim, frame_interval):
    """Returns a stacked image from the grayscale buffer."""

    # Create image from grayscale buffer
    img = np.zeros(img_dim)
    for i in range(img_dim[2]):
        img[:, :, i] = grayscale_state_buffer[::-1][i * frame_interval]
    img = np.reshape(img, (1,) + img_dim)

    return img


def live_trial(env, model, actions, frame_stack_num, frame_interval, img_dim, 
               image_processing_fn=None, get_stacked_state_live_fn=None, trials=1, 
               max_frames=10000, plot_frequency=None, render=False):
    """Do a live trial."""

    all_rewards = []
    for t in range(trials):
        # Reset environment and process image
        s_0 = env.reset()
        s_0 = image_processing_fn(s_0, 96)
        grayscale_state_buffer = [s_0] * frame_stack_num * frame_interval
        s_0_stacked = get_stacked_state_live_fn(grayscale_state_buffer, img_dim, frame_interval)

        episode_rewards = 0

        i = 0
        while True:
            # Full throttle for first 50 frames
            if i < 50:
                a_0 = 0
            else:
                a_0 = np.argmax(model.predict(s_0_stacked))
            s_1, reward, done, _ = env.step(actions[a_0])

            episode_rewards += reward

            if plot_frequency is not None:
                if i%plot_frequency == 0 and i>50:
                    s_1_gs = np.dot(s_1[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                    s_1_gs[:85, :][s_1_gs[:85, :]==106.0]=101.0
                    s_1_gs[:85, :][s_1_gs[:85, :]==104.0]=101.0
                    s_1_gs[:85, :][s_1_gs[:85, :]==161.0]=255.0
                    s_1_gs[:85, :][s_1_gs[:85, :]==177.0]=255.0
                    s_1_gs[:85, :][s_1_gs[:85, :]==254.0]=255.0
                    s_1_gs[:85, :][s_1_gs[:85, :]==76.0]=255.0

                    s_1_gs[86:, :12] = 0.0
                    s_1_gs[86:, :][s_1_gs[86:, :]==254.0]=255.0
                    s_1_gs[86:, :][s_1_gs[86:, :]==29.0]=255.0
                    s_1_gs[86:, :][s_1_gs[86:, :]==44.0]=255.0
                    s_1_gs[86:, :][s_1_gs[86:, :]==149.0]=255.0
                    s_1_gs[86:, :][s_1_gs[86:, :]==76.0]=255.0

                    plt.figure()
                    plt.imshow(s_1_gs)
                    plt.axis('off')
                    plt.show()

            if done or i > max_frames:
                break

            # Append latest state and get new stacked representation
            s_1 = image_processing_fn(s_1, 96)
            grayscale_state_buffer.append(s_1)
            s_0_stacked = get_stacked_state_live_fn(grayscale_state_buffer, img_dim, frame_interval)

            i += 1

        all_rewards.append(episode_rewards)
        print(t, episode_rewards, np.mean(all_rewards))
        if render:
            show_video()
        env.close()
        
    print()

    return all_rewards
