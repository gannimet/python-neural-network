import tiktoken
import numpy
from collections import Counter
import json
import sys
import os
from kafkai_utils import model_token_to_one_hot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural_network import NeuralNetwork, softmax, leaky_relu

enc = tiktoken.get_encoding("o200k_base")
MIN_TOKEN_COUNT = 4
NUM_CONTEXT_TOKENS = 6

def one_hot_sequence_to_training_data(one_hot_sequence):
    training_data = []
    
    for i in range(len(one_hot_sequence) - NUM_CONTEXT_TOKENS):
        inputs_sequence = one_hot_sequence[i:i + NUM_CONTEXT_TOKENS]
        input_vector = numpy.vstack(inputs_sequence)
        output_vector = one_hot_sequence[i + NUM_CONTEXT_TOKENS]
        training_data.append((input_vector, output_vector))
    
    return training_data

def get_model_tokens_from_file(file):
    file_content = file.read()
    raw_tokens = enc.encode(file_content)
    
    # Shift original tokens forward by 1 to make way for 0 as "unknown" token
    model_tokens = [raw_token + 1 for raw_token in raw_tokens]
    return model_tokens

with (
    open("kafkai/prozess.txt", "r", encoding="utf-8") as prozessFile,
    open("kafkai/amerika.txt", "r", encoding="utf-8") as amerikaFile,
    open("kafkai/schloss.txt", "r", encoding="utf-8") as schlossFile,
):
    print("Reading files and tokenizing …")
    prozess_tokens = get_model_tokens_from_file(prozessFile)
    amerika_tokens = get_model_tokens_from_file(amerikaFile)
    schloss_tokens = get_model_tokens_from_file(schlossFile)
    
    # Map all tokens occurring less than MIN_TOKEN_COUNT times to "unknown token 0"
    print("Applying thresholds to tokens …")
    all_model_tokens = [*prozess_tokens, *amerika_tokens, *schloss_tokens]
    all_model_token_counts = Counter(all_model_tokens)
    prozess_thresholded_model_tokens = [token if all_model_token_counts[token] >= MIN_TOKEN_COUNT else 0 for token in prozess_tokens]
    amerika_thresholded_model_tokens = [token if all_model_token_counts[token] >= MIN_TOKEN_COUNT else 0 for token in amerika_tokens]
    schloss_thresholded_model_tokens = [token if all_model_token_counts[token] >= MIN_TOKEN_COUNT else 0 for token in schloss_tokens]
    all_thresholded_model_tokens = [*prozess_thresholded_model_tokens, *amerika_thresholded_model_tokens, *schloss_thresholded_model_tokens]
    
    # Map tokens to one-hot vectors
    print("Preparing training data …")
    unique_tokens_lookup = sorted(list(set([*all_thresholded_model_tokens, 0])))
    
    with open("kafkai/kafkai_tokens.json", "w") as json_file:
        json.dump(unique_tokens_lookup, json_file)
    
    prozess_one_hot_sequence = [model_token_to_one_hot(token, unique_tokens_lookup) for token in prozess_thresholded_model_tokens]
    amerika_one_hot_sequence = [model_token_to_one_hot(token, unique_tokens_lookup) for token in amerika_thresholded_model_tokens]
    schloss_one_hot_sequence = [model_token_to_one_hot(token, unique_tokens_lookup) for token in schloss_thresholded_model_tokens]
    prozess_training_data = one_hot_sequence_to_training_data(prozess_one_hot_sequence)
    amerika_training_data = one_hot_sequence_to_training_data(amerika_one_hot_sequence)
    schloss_training_data = one_hot_sequence_to_training_data(schloss_one_hot_sequence)
    all_training_data = [*prozess_training_data, *amerika_training_data, *schloss_training_data]
    print(f"Number of training examples: {len(all_training_data)}")
    
    nn = NeuralNetwork(
        structure=[NUM_CONTEXT_TOKENS * len(unique_tokens_lookup), 300, 300, 300, len(unique_tokens_lookup)],
        hidden_activation_func=leaky_relu,
        output_activation_func=softmax,
        eta=0.05,
        batch_size=50_000,
        n_iterations=100
    )
    print("Training network …")
    nn.train(all_training_data)
    
    nn.save_to_file(f"kafkai_models/kafka_weights_i{nn.n_iterations}_s{nn.batch_size}.npz")
    nn.plot()
    