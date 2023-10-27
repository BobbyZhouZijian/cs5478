import gym
import time
import openai
import numpy as np

# set the openai api key
openai.api_key = "" # your openai api key

env = gym.make('ma_gym:PongDuel-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

def get_prompt(states, player):
    state0 = states[player]
    direction = np.argmax(state0[4:])

    if player == 0:
        if direction == 0:
            direction = "left-top"
        elif direction == 1:
            direction = "left"
        elif direction == 2:
            direction = "left-down"
        elif direction == 3:
            direction = "right-down"
        elif direction == 4:
            direction = "right"
        elif direction == 5:
            direction = "right-top"
    else:
        if direction == 0:
            direction = "right-top"
        elif direction == 1:
            direction = "right"
        elif direction == 2:
            direction = "right-down"
        elif direction == 3:
            direction = "left-down"
        elif direction == 4:
            direction = "left"
        elif direction == 5:
            direction = "left-top"

    prompt = """
I have a two player Pong game.

Image there is a 2d board. you can control a point moving up or down on the left boundary. There exists another player controlling another point moving on the right boundary.

There exists a ball moving on the plane. It has 6 directions. Specifically, the direction vector can be selected from the set (left, right, left-top, left-bottom, right-top, right-bottom)

When the ball hit your segments or the top / down boundary, it will be rebounded to the opposite direction. If the ball is touching left boundary, you will loss.

Given at a certain round the ball is at ({:.1f}, {:.1f}) and moving {}, you are at ({:.1f}, {:.1f}), what will you do? You can move up, move down or noop.

Answer only "noop", "up" or "down"
    """.format(state0[3], 1 - state0[2], direction, state0[1], 1 - state0[0])
    return prompt

obs_n = env.reset()
iter = 50
for _ in range(iter):
    env.render()
    state = env.get_agent_obs()
    # create openai chat agent
    prompt = get_prompt(state, 0)
    messages = [
        {"role": "system", "content": "You are an expert in planning and decision making."},
        {"role": "user", "content": prompt},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    # get the text from the response
    action = response.choices[0].message['content']
    print(action)
    print(f'---------End of Iter {_}----------')
    # filter the keyword "up" or "down" from the response
    # check if the word "up" is in the response
    if "down" in action:
        action_1 = 2
    elif "up" in action:
        action_1 = 1
    else:
        action_1 = 0

    # do the same for the second player
    # prompt = get_prompt(state, 1)
    # messages = [
    #     {"role": "system", "content": "You are an expert in planning and decision making."},
    #     {"role": "user", "content": prompt},
    #     ]
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     temperature=0.8,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     stop=["\n"]
    # )
    # action = response.choices[0].message['content']
    # print(action)
    # if "down" in action:
    #     action_2 = 2
    # elif "up" in action:
    #     action_2 = 1
    # else:
    #     action_2 = 0
    
    action_2 = np.random.randint(0, 3)

    # take the actions
    action_n = [action_1, action_2]
    for _ in range(3):
        obs_n, reward_n, done_n, info = env.step(action_n)
        ep_reward += sum(reward_n)
        time.sleep(0.03)
        env.render()
env.close()