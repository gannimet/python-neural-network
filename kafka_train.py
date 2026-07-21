import tiktoken
import numpy
from collections import Counter
from neural_network import NeuralNetwork, softmax, leaky_relu

enc = tiktoken.get_encoding("o200k_base")
MIN_TOKEN_COUNT = 3
NUM_CONTEXT_TOKENS = 8

def model_token_to_one_hot(token, unique_tokens_lookup):
    num_tokens = len(unique_tokens_lookup)
    token_index = 0 if token not in unique_tokens_lookup else unique_tokens_lookup.index(token)
    one_hot = numpy.zeros((num_tokens, 1))
    one_hot[token_index, 0] = 1
    return one_hot

def one_hot_to_model_token(one_hot, tokens_lookup):
    # TODO implement sampling from distribution rather than fixed lookup
    token_index = numpy.argmax(one_hot)
    return tokens_lookup[token_index]

def one_hot_sequence_to_training_data(one_hot_sequence):
    training_data = []
    
    for i in range(len(one_hot_sequence) - NUM_CONTEXT_TOKENS):
        inputs_sequence = one_hot_sequence[i:i + NUM_CONTEXT_TOKENS]
        input_vector = numpy.vstack(inputs_sequence)
        output_vector = one_hot_sequence[i + NUM_CONTEXT_TOKENS]
        training_data.append((input_vector, output_vector))
    
    return training_data

with open("kafkai/prozess.txt", "r", encoding="utf-8") as file:
    print("Reading file …")
    file_content = file.read()
    print("Tokenizing text …")
    raw_tokens = enc.encode(file_content)
    
    # Shift original tokens forward by 1 to make way for 0 as "unknown" token
    model_tokens = [raw_token + 1 for raw_token in raw_tokens]
    
    # Map all tokens occurring less than MIN_TOKEN_COUNT times to "unknown token 0"
    model_token_counts = Counter(model_tokens)
    thresholded_model_tokens = [token if model_token_counts[token] >= MIN_TOKEN_COUNT else 0 for token in model_tokens]
    
    # Map tokens to one-hot vectors
    print("Preparing network training data …")
    unique_tokens_lookup = sorted(list(set([*thresholded_model_tokens, 0])))
    one_hot_sequence = [model_token_to_one_hot(token, unique_tokens_lookup) for token in thresholded_model_tokens]
    training_data = one_hot_sequence_to_training_data(one_hot_sequence)
    
    nn = NeuralNetwork(
        structure=[NUM_CONTEXT_TOKENS * len(unique_tokens_lookup), 300, 200, 100, len(unique_tokens_lookup)],
        hidden_activation_func=leaky_relu,
        output_activation_func=softmax,
        eta=0.01,
        batch_size=100,
        n_iterations=100_000
    )
    print("Training network …")
    nn.train(training_data)
    
    seed_text = "K. war telephonisch verständigt worden, daß am nächsten Sonntag eine kleine Untersuchung in seiner Angelegenheit stattfinden würde."
    seed_raw_tokens = enc.encode(seed_text)[-NUM_CONTEXT_TOKENS:]
    seed_model_tokens = [raw_token + 1 for raw_token in seed_raw_tokens]
    known_seed_model_tokens = [0 if model_token not in unique_tokens_lookup else model_token for model_token in seed_model_tokens]
    seed_one_hots = [model_token_to_one_hot(token, unique_tokens_lookup) for token in known_seed_model_tokens]
    seed_input = numpy.vstack(seed_one_hots)
    
    prediction = nn.predict(seed_input)
    print(f"Prediction vector: {prediction}")
    
    predicted_token = one_hot_to_model_token(prediction, unique_tokens_lookup)
    print(f"Predicted model token: {predicted_token}")
    
    predicted_text = "<unknown>" if predicted_token == 0 else enc.decode([predicted_token - 1])
    print(f"Predicted text: '{predicted_text}'")
    
    nn.save_to_file(f"kafkai_models/kafka_weights_i{nn.n_iterations}_s{nn.batch_size}.npz")
    nn.plot()
    