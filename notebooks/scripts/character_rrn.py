import tensorflow as tf

################################
## 1. Stateless RNN
################################
def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    """Creates a tf.data.Dataset tailored for sequence-to-sequence prediction models.

    Takes a sequence of tokens (e.g., characters or words) and processes it into
    sliding windows of size `length`, generating pairs of inputs (X) and target
    labels (Y) shifted by one token in time.

    Args:
        sequence (tf.Tensor or array-like): The original encoded sequence tensor.
        length (int): The length of the input sequence that the model will receive.
        shuffle (bool, optional): Whether to randomly shuffle the sequences.
            Defaults to False.
        seed (int, optional): Random seed for the shuffle buffer.
            Defaults to None.
        batch_size (int, optional): The batch size for training the model.
            Defaults to 32.

    Returns:
        tf.data.Dataset: A dataset yielding tuples of (X, Y) where:
            - X has dimensions (batch_size, length)
            - Y has dimensions (batch_size, length)
    """
    # 1. Convert the 1D tensor/sequence into a TensorFlow Dataset
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    
    # 2. Create sliding windows of size (length + 1) shifting by 1 character at a time.
    #    (The +1 is required to fit both the input sequence and the target token).
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    
    # 3. Flatten each sub-dataset produced by window() into an individual tensor of size (length + 1)
    ds = ds.flat_map(lambda window_: window_.batch(length + 1))
    
    # 4. Shuffle the sequences if shuffle=True using an in-memory buffer
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
        
    # 5. Group the individual sequences into batches of size batch_size
    ds = ds.batch(batch_size)
    
    # 6. Split each batch into Inputs X (from index 0 up to the second-to-last token) 
    #    and Targets Y (from index 1 to the end), then optimize pipeline execution with prefetch.
    return ds.map(lambda window_: (window_[:, :-1], window_[:, 1:])).prefetch(1)

################################
## 2. Stateful RNN
################################

def to_dataset_for_stateful_rnn(sequence, length):
    """Prepares a sequential, non-overlapping dataset for stateful RNN training.

    Transforms a single sequence of token IDs into non-overlapping windowed pairs 
    of inputs (X) and shifted targets (Y), structured for a batch size of 1 to 
    maintain continuous temporal hidden states across consecutive iterations.

    Args:
        sequence (tf.Tensor or array-like): The continuous, encoded sequence 
            of tokens.
        length (int): The sequence length/window size expected as input by the model.

    Returns:
        tf.data.Dataset: A dataset emitting tuples of (inputs, targets) where both 
            tensors have shape (1, length) and temporal continuity is preserved 
            across steps.
    """
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length + 1)).batch(1)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

class ResetStatesCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "reset_states"):
                layer.reset_states()

def stateful_weights_stateless(n_tokens, stateful_model_path):
    best_stateful_model = tf.keras.models.load_model(stateful_model_path)

    stateless_model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,)),
        tf.keras.layers.Embedding(
            input_dim=n_tokens,
            output_dim=16
        ),
        tf.keras.layers.GRU(
            128,
            return_sequences=True
        ),
        tf.keras.layers.Dense(
            n_tokens,
            activation="softmax"
        )
    ])
    stateless_model.set_weights(best_stateful_model.get_weights())
    return stateless_model    


################################
## Helper functions : Char prediction
################################

def next_char(text, model, vocabulary, temperature=1.0):
    """Generates the next predicted character for a given text using stochastic sampling.

    Applies a temperature factor to the model's logits to control the diversity
    and creativity of the generated text prior to sampling a character.

    Args:
        text (tf.Tensor): The input text (or list of texts) upon which
            the model will make the prediction.
        model (tf.keras.Model): The trained language model for sequence prediction.
        vocabulary (list[str]): The list of vocabulary items mapping IDs to characters.
        temperature (float, optional): Controls the randomness of the prediction.
            - Close to 0: Generates deterministic, rigid text (favors high-probability characters).
            - Close to 1: Preserves the original learned probabilities.
            - > 1: Increases diversity and creativity (flattening the probability distribution).
            Defaults to 1.0.

    Returns:
        str: The selected character sampled according to the adjusted probability distribution.
    """
    # 1. Obtain the predicted probability distribution for the last token/character in the sequence
    y_proba = model(text)[0, -1:]
    
    # 2. Convert probabilities back to logits and scale by the temperature factor.
    #    Dividing logits by T > 1 flattens the distribution (increases diversity);
    #    Dividing by T < 1 sharpens the distribution (increases conservatism).
    rescaled_logits = tf.math.log(y_proba) / temperature 
    
    # 3. Randomly sample 1 token based on the rescaled logit distribution
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    
    # 4. Decode the sampled ID into its corresponding character from the vocabulary.
    #    An offset of +2 is added to account for special/reserved tokens in the vocabulary.
    return vocabulary[char_id + 2]


def extend_text(text, model, vocabulary, n_chars=500, temperature : float = 1.0):
    for _ in range(n_chars):
        text += next_char(text, model, vocabulary, temperature)
    return text

def to_string(tensor):
    """
    Safely extracts a UTF-8 string from a TensorFlow tensor or NumPy array
    containing byte strings.
    """

    numpy_val = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    
    if numpy_val.shape == ():
        bytes_val = numpy_val
    else:
        bytes_val = numpy_val.flat[0]
    
    # Decode bytes to string
    return bytes_val.decode('utf-8')