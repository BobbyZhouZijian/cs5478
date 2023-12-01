import openai
import numpy as np
from ma_gym.wrappers import Monitor

# set the openai api key
openai.api_key = "" # your openai api key

def get_prompt(states):
    x1 = states[0]
    y1 = states[1]
    x2 = states[2]
    y2 = states[3]
    x1, y1 = unnormalize_coordinate(x1, y1)
    x2, y2 = unnormalize_coordinate(x2, y2)
    prompt = """
    Given at a certain round, the player 1 is at ({:.1f}, {:.1f}), player 2 is at ({:.1f}, {:.1f}). How should the player 1 and player 2 move?
    Answer "down", "left" "up" "right" or "noop" for each player with justification.
    """.format(x1, y1, x2, y2)
    return prompt

def get_answer_prompt(text, player=1):
    prompt = """
{}

The above is the description of two players' decision about the action in a Switch game. What decision did the palyer {} make? 
Answer "down" "left" "up" "right" or "noop" only without any justification.
""".format(text, str(player))
    return prompt

def map_action_to_number(action):
    if "down" in action:
        return 0
    elif "left" in action:
        return 1
    elif "up" in action:
        return 2
    elif "right" in action:
        return 3
    else:
        return 4
    
def unnormalize_coordinate(x, y):
    return int(round(x*2)), int(round(y*6))


def get_action_from_llm(state, agent_idx):
    prompt = get_prompt(state)
    context = '''
    You are an expert in planning and decision making. Now there are two players playing a coorperative game, which is Switch2-v1 provided by OpenAI gym.

    The players are in a 2D board represented as a 2d array with 3 rows and 7 columns. The first row is the top row, and the third row is the bottom row. The first column is the leftmost column, and the seventh column is the rightmost column.
    Location of the player will be represented as (x, y), where x represents the row index, and y represents the column index. For example, (0, 0) represents the top-left corner, and (2, 6) represents the bottom-right corner.
    
    The game starts with two players at (0, 0) and (0, 6) respectively. The goal of the game is to move the two players to (0, 6) and (0, 0) respectively, i.e, switch their location. The game ends when both players reach their goals.
    
    Notice that in the 2d board, grid (0, 2), (0, 3), (0, 4), (2, 2), (2, 3), (2, 4) is not accessible. The players cannot move to these grids and cannot move outside the board.
    
    Given current location (x, y) of a player, the movement of the player follows the following rules:
    1. If moving right, the position will be changed to (x, y+1)
    2. If moving left, the position will be changed to (x, y-1)
    3. If moving up, the position will be changed to (x-1, y)
    4. If moving down, the position will be changed to (x+1, y)
    5. If moving noop, the position will not be changed

    Heurstics to win:
    1. To win the game, the player need to approach the goal, if the goal is on the right, the player should move right, if the goal is on the left, the player should move left.
    2. You notice that from column 2 to column 4, only the middle row is accessible. So if there is already a player on that row, the other player should not move to that row. Otherwise, the two players will be stuck. 
    3. An appropriate action is to wait for the other player to passing through the middle row. after it goes out, the other player can move to the middle row so that the two players will not block each other.

    Your goal is to win the cooperative game by providing action based on the heuristics.
    '''
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # get the text from the response
    action = response.choices[0].message['content']
    print('--------------Justification---------------')
    print(action)
    action = get_answer_prompt(action, agent_idx + 1)
    messages = [
        {"role": "system", "content": "You are an expert in summarizing things."},
        {"role": "user", "content": action},
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
    action = response.choices[0].message['content']
    print('--------------Answer---------------')
    print(action)
    action = map_action_to_number(action)
    return action
        