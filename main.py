import typing
import numpy as np
from server import run_server
import torch

device = torch.device("cpu")

policy = torch.load('best_model.pth', map_location=device)
policy.to(device)
policy.eval()


def info() -> typing.Dict:
  return {
      "apiversion": "1",
      "author": "g08",
      "color": "#00ff00",
      "head": "all-seeing",
      "tail": "gauge",
  }


def start(game_state: typing.Dict):
  print("GAME START")


def end(game_state: typing.Dict):
  print("GAME OVER\n")


def move(game_state: typing.Dict) -> typing.Dict:
  global policy
  my_body = game_state["you"]["body"]
  my_health = game_state["you"]["health"]
  op_body = []
  op_health = 100
  BOARD_WIDTH = game_state["board"]["width"]
  BOARD_HEIGHT = game_state["board"]["height"]

  for snake in game_state["board"]["snakes"]:
    if snake["id"] != game_state["you"]["id"]:
      op_body = snake["body"]
      op_health = snake["health"]

  result_into_move = {
      "up": (0, 1),
      "down": (0, -1),
      "left": (-1, 0),
      "right": (1, 0),
  }

  # Preprocess game state to obtain input observation
  policy.eval()
  obs = preprocess_game_state(game_state)
  obs = obs.to(device)

  result = {"move": "up"}

  with torch.no_grad():
    # Assuming obs is your input observation
    _, action, _, _ = policy.act(obs, None, None)
    # Get the actual action value
    action_value = action.item()

    # Change up and down if you train by yourself and change the gameinstance.cpp file.
    if action_value == 0:
      result = {"move": "down"}
    elif action_value == 1:
      result = {"move": "up"}
    elif action_value == 2:
      result = {"move": "left"}
    elif action_value == 3:
      result = {"move": "right"}

  # if there is no opponent, then return the reinforcement learning result
  if not op_body:
    return result

  def run_into_wall(result: dict) -> bool:
    next_head = {
        "x": my_body[0]["x"] + result_into_move[result["move"]][0],
        "y": my_body[0]["y"] + result_into_move[result["move"]][1]
    }
    return next_head["x"] < 0 or next_head["x"] >= BOARD_WIDTH or next_head[
        "y"] < 0 or next_head["y"] >= BOARD_HEIGHT

  def run_into_body(result: dict) -> bool:
    next_head = {
        "x": my_body[0]["x"] + result_into_move[result["move"]][0],
        "y": my_body[0]["y"] + result_into_move[result["move"]][1]
    }
    my_next_body_without_head = my_body[:-1] if my_health != 100 else my_body
    op_next_body_without_head = op_body[:-1] if op_health != 100 else op_body
    return next_head in my_next_body_without_head or next_head in op_next_body_without_head

  # check reinforcement learning result is safe
  if not run_into_wall(result) and not run_into_body(
      result):
    return result
  
  # TODO: You can add your own rule-based algorithm here
  # Such as alpha-beta pruning, minimax, monte carlo tree search, etc.

  my_all_possible_move = []
  for move in ["up", "down", "left", "right"]:
    algorithm_result = {"move": move}
    if not run_into_wall(algorithm_result) and not run_into_body(
        algorithm_result):
      my_all_possible_move.append(move)

  # if there is no safe move, then return the reinforcement learning result
  if not my_all_possible_move:
    return result

  # if there is safe move, then return random safe move
  return {"move": np.random.choice(my_all_possible_move)}


def preprocess_game_state(game_state: typing.Dict) -> torch.Tensor:
  """
  If you want to know each layer's meaning, you can go to the following file
  gym_battlesnake/src/gamewrapper.cpp from line 129 to 284
  """
  board_info = game_state["board"]
  foods = board_info["food"]
  your_snake_info = game_state["you"]
  your_snake_id = your_snake_info["id"]
  your_snake_length = your_snake_info["length"]
  snakes = game_state["board"]["snakes"]
  headx_to_mid = 11 - your_snake_info["head"]["x"]
  heady_to_mid = 11 - your_snake_info["head"]["y"]

  # Create a 4D tensor for observations (obs)
  obs = np.zeros((1, 17, 23, 23), dtype=np.float32)
  # Fill the position of your snake head

  # Fill the position of your snake body
  for snake in snakes:
    obs[0, 0, snake["head"]["x"] + headx_to_mid,
        snake["head"]["y"] + heady_to_mid] = snake["health"]
    this_snake_id = snake["id"]
    this_snake_length = snake["length"]
    if this_snake_id != your_snake_id and this_snake_length >= your_snake_length:
      obs[0, 3, snake["head"]["x"] + headx_to_mid,
          snake["head"]["y"] + heady_to_mid] = 1
    this_snake_health = snake["health"]
    seg = this_snake_length
    double_tail = True if (snake["body"][-1] == snake["body"][-2]
                           or this_snake_health == 100) else False
    for snake_body in snake["body"]:
      obs[0, 1, snake_body["x"] + headx_to_mid,
          snake_body["y"] + heady_to_mid] = 1
      obs[0, 2, snake_body["x"] + headx_to_mid,
          snake_body["y"] + heady_to_mid] = seg
      seg -= 1
      if this_snake_id != your_snake_id and this_snake_length >= your_snake_length:
        obs[0, 8, snake_body["x"] + headx_to_mid, snake_body["y"] +
            heady_to_mid] = this_snake_length - your_snake_length + 1
      elif this_snake_id != your_snake_id and this_snake_length < your_snake_length:
        obs[0, 9, snake_body["x"] + headx_to_mid, snake_body["y"] +
            heady_to_mid] = your_snake_length - this_snake_length
      if double_tail:
        obs[0, 7, snake_body["x"] + headx_to_mid,
            snake_body["y"] + heady_to_mid] = 1

  for food in foods:
    obs[0, 4, food["x"] + headx_to_mid,
        food["y"] + heady_to_mid] = 1  # layer4: food

  # Fill the game board area
  for i in range(headx_to_mid, headx_to_mid + 11):
    for j in range(heady_to_mid, heady_to_mid + 11):
      obs[0, 5, i, j] = 1  # layer5: game board
      obs[0, 10 + len(snakes) - 2, i, j] = 1  # layer10: opponent is alive

  obs[0, 6, 11, 11] = 1  # layer6: head_mask

  return torch.tensor(obs, dtype=torch.float32)


if __name__ == "__main__":
  run_server({"info": info, "start": start, "move": move, "end": end})
