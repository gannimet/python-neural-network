import numpy

def model_token_to_one_hot(token, unique_tokens_lookup):
    num_tokens = len(unique_tokens_lookup)
    token_index = 0 if token not in unique_tokens_lookup else unique_tokens_lookup.index(token)
    one_hot = numpy.zeros((num_tokens, 1), dtype=numpy.bool)
    one_hot[token_index, 0] = 1
    return one_hot

def one_hot_to_model_token(one_hot, tokens_lookup):
    token_index = numpy.random.choice(len(tokens_lookup), p=one_hot.flatten())
    return tokens_lookup[token_index]