import numpy as np
import inference_util as iu
import gym
from tensorflow import keras
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


# Definte constants
PATH_DQN_1 = "../runs/20220418-131529"
PATH_DQN_2 = "../runs/20220419-065418"
PATH_DDQN_1 = "../runs/20220412-110508"
PATH_DDQN_2 = "../runs/20220418-072859"
PATH_DDQN_3a = "../runs/20220425-210405"
PATH_DDQN_3b = "../runs/20220425-210418"

ACTIONS = np.array([[0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0]]
                    )


if __name__ == "__main__":

    # DQN 1
    model = keras.models.load_model(PATH_DQN_1 + '/model_final.h5')
    env = gym.make("CarRacing-v0", verbose=0)
    dqn1_rewards = iu.live_trial(env, 
                                model,
                                ACTIONS,
                                frame_stack_num=4, 
                                frame_interval=1, 
                                img_dim=(96, 96, 4), 
                                image_processing_fn=iu.image_processing_old, 
                                get_stacked_state_live_fn=iu.get_stacked_state_live_old,
                                trials=100, 
                                render=False)
    with open('dqn1_rewards.pickle', 'wb') as handle:
        pickle.dump(dqn1_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # DQN 2
    # model = keras.models.load_model(PATH_DQN_2 + '/model_final.h5')
    # env = gym.make("CarRacing-v0", verbose=0)
    # dqn2_rewards = iu.live_trial(env, 
    #                             model,
    #                             frame_stack_num=4, 
    #                             frame_interval=1, 
    #                             img_dim=(96, 96, 4), 
    #                             preprocessing_fn=iu.image_processing_old, 
    #                             get_stacked_state_live_fn=iu.get_stacked_state_live_old,
    #                             trials=100, 
    #                             render=False)
    # with open('dqn2_rewards.pickle', 'wb') as handle:
    #     pickle.dump(dqn2_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # DDQN 1
    # model = keras.models.load_model(PATH_DDQN_1 + '/model_final.h5')
    # env = gym.make("CarRacing-v0", verbose=0)
    # ddqn1_rewards = iu.live_trial(env, 
    #                             model,
    #                             frame_stack_num=4, 
    #                             frame_interval=4, 
    #                             img_dim=(96, 96, 4), 
    #                             preprocessing_fn=iu.image_processing_old, 
    #                             get_stacked_state_live_fn=iu.get_stacked_state_live_old,
    #                             trials=100, 
    #                             render=False)
    # with open('ddqn1_rewards.pickle', 'wb') as handle:
    #     pickle.dump(ddqn1_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # # DDQN 2
    # model = keras.models.load_model(PATH_DDQN_2 + '/model_final.h5')
    # env = gym.make("CarRacing-v0", verbose=0)
    # ddqn2_rewards = iu.live_trial(env, 
    #                             model,
    #                             frame_stack_num=4, 
    #                             frame_interval=4, 
    #                             img_dim=(96, 96, 4), 
    #                             preprocessing_fn=iu.image_processing_old, 
    #                             get_stacked_state_live_fn=iu.get_stacked_state_live_old,
    #                             trials=100, 
    #                             render=False)
    # with open('ddqn2_rewards.pickle', 'wb') as handle:
    #     pickle.dump(ddqn2_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # DDQN 3a
    model = keras.models.load_model(PATH_DDQN_3a + '/model_final.h5')
    env = gym.make("CarRacing-v0", verbose=0)
    ddqn3a_rewards = iu.live_trial(env, 
                                model,
                                ACTIONS,
                                frame_stack_num=4, 
                                frame_interval=4, 
                                img_dim=(96, 96, 4), 
                                image_processing_fn=iu.image_processing_new, 
                                get_stacked_state_live_fn=iu.get_stacked_state_live_new,
                                trials=100, 
                                render=False)
    with open('ddqn3a_rewards.pickle', 'wb') as handle:
        pickle.dump(ddqn3a_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # DDQN 3b
    model = keras.models.load_model(PATH_DDQN_3b + '/model_final.h5')
    env = gym.make("CarRacing-v0", verbose=0)
    ddqn3b_rewards = iu.live_trial(env, 
                                model,
                                ACTIONS,
                                frame_stack_num=4, 
                                frame_interval=4, 
                                img_dim=(96, 96, 4), 
                                image_processing_fn=iu.image_processing_new, 
                                get_stacked_state_live_fn=iu.get_stacked_state_live_new,
                                trials=100, 
                                render=False)
    with open('ddqn3b_rewards.pickle', 'wb') as handle:
        pickle.dump(ddqn3b_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
