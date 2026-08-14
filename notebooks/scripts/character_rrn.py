import tensorflow as tf

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