import gym
import numpy as np
import tensorflow as tf
import random
import time
from IPython.display import clear_output
from collections import deque
from tensorflow import keras

#print(gym.envs.registry.all())

# Create the environment obeject
env = gym.make('AirRaid-v0')

# Initializing the Q-learning parameters
train_episodes = 300

def agent(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(replay_memory, model, target_model, done):
    learning_rate = 0.7
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    
    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observaton, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
learning_rate = 0.1
discount_rate = 0.99

def main():
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    # Initialize the main and target models
    model = agent(env.observation_space.shape, env.action_space.n)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen = 50_000)

# ------------Watch before training------------
    for episode in range(1):
        state = env.reset()
        done = False
        print("*****EPISODE ", episode+1, "*****\n\n\n\n")
        time.sleep(1)

        for step in range(100):
            clear_output(wait=True)
            env.render()
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                encoded = state
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            else:
                action = env.action_space.sample() 
            new_state, reward, done, info = env.step(action)
            state = new_state

            if done:
                clear_output(wait=True)
                env.render()
                # if reward == 1:
                #     #print("****You reached the goal!****")
                #     #time.sleep(3)
                # else:
                #     #print("****You fell through a hole!****")
                #     #time.sleep(3)
                #     clear_output(wait=True)
                #     break
    env.close()
# ------------Watch before training------------

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        state = env.reset()
        done = False
        rewards_current_episodes = 0

        while not done:
            steps_to_update_target_model += 1
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                encoded = state
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)

            # Append the replay_memory
            replay_memory.append([state, action, reward, new_state, done])

            # Update main NN using Bellman's equatiom
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model, done)

            state = new_state
            rewards_current_episodes += reward

            if done == True:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(rewards_current_episodes, episode, reward))
                
                if steps_to_update_target_model >= 100:
                    print("Copying main NN weights to the target NN weights")
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
    
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

# ------------Watch after training------------
    for episode in range(1):
        state = env.reset()
        done = False
        print("*****EPISODE ", episode+1, "*****\n\n\n\n")
        time.sleep(1)

        for step in range(100):
            clear_output(wait=True)
            print(env.render())
            time.sleep(0.3)
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > exploration_rate:
                encoded = state
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            else:
                action = env.action_space.sample()     
            new_state, reward, done, info = env.step(action)
            state = new_state

            if done:
                clear_output(wait=True)
                print(env.render())
                # if reward == 1:
                #     #print("****You reached the goal!****")
                #     time.sleep(3)
                # else:
                #     print("****You fell through a hole!****")
                #     #time.sleep(3)
                #     clear_output(wait=True)
                #     break
    env.close()
# ------------Watch after training------------

if __name__ == '__main__':
    main()