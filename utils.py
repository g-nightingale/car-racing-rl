import numpy as np


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
