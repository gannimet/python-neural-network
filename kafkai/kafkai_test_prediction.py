import tiktoken
import numpy
import json
import sys
import os
import time
from kafkai_utils import model_token_to_one_hot, one_hot_to_model_token

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils
from neural_network import NeuralNetwork, softmax, leaky_relu

enc = tiktoken.get_encoding("o200k_base")
MIN_TOKEN_COUNT = 4
NUM_CONTEXT_TOKENS = 6

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
)

filename = utils.select_model_file(utils.ModelMode.KAFKAI)
nn.load_from_file(filename)

with open("kafkai/kafkai_tokens.json", "r") as json_file:
    unique_tokens_lookup = json.load(json_file)

    seed_text = input("Enter prompt: ")
    seed_raw_tokens = enc.encode(seed_text)[-NUM_CONTEXT_TOKENS:]
    seed_model_tokens = [raw_token + 1 for raw_token in seed_raw_tokens]

    while True:
        known_seed_model_tokens = [0 if model_token not in unique_tokens_lookup else model_token for model_token in seed_model_tokens]
        seed_one_hots = [model_token_to_one_hot(token, unique_tokens_lookup) for token in known_seed_model_tokens]
        seed_input = numpy.vstack(seed_one_hots)
        prediction = nn.predict(seed_input)
        predicted_token = one_hot_to_model_token(prediction, unique_tokens_lookup)
        predicted_text = "<unknown>" if predicted_token == 0 else enc.decode([predicted_token - 1])
        
        print(predicted_text)
        
        seed_model_tokens.append(predicted_token)
        seed_model_tokens.pop(0)
        
        time.sleep(1)