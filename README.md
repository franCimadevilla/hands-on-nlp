# Hands-On Natural Language Processing with RNNs & Attention

This repository contains implementation code, experiments, and notes based on **Chapter 16: Natural Language Processing with RNNs and Attention** from *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)* by Aurélien Géron.

## Topics Covered

* **Character-Level Text Generation**: Tokenization via `TextVectorization`, windowed dataset building with `tf.data.Dataset`, and training a Character-RNN.
* **Stateless vs. Stateful RNNs**: Managing hidden states across sequence batches and converting trained stateful models to stateless models for variable-length inference.
* **Temperature-Based Text Sampling**: Controlling randomness in text generation via logit rescaling and categorical sampling.
* **Sentiment Analysis & Sequence Classification**: Building end-to-end NLP pipelines using Recurrent Neural Networks (GRU/LSTM).
* **Encoder-Decoder Architectures**: Implementing sequence-to-sequence (Seq2Seq) models for translation and sequence transformation tasks.
* **Attention Mechanisms**: Applying Bahdanau and Luong attention layers to capture long-range sequence dependencies.

---

## Project Structure

```text
├── notebooks/
├── images/
│   └── nlp/                  # Generated figures and evaluation plots
├── models/                   
├── requirements.txt
└── README.md

```

---

## Setup & Environment

### Prerequisites

* Python $\ge 3.7$
* TensorFlow $\ge 2.8.0$
* GPU highly recommended

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/franCimadevilla/hands-on-nlp.git
cd hands-on-nlp

```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```

---

## Technical Compatibility Notice

Due to major architectural changes in Keras 3 regarding stateful RNNs, ragged tensors, and TensorFlow Hub integration, this project explicitly configures TensorFlow to run with **Keras 2 (`tf_keras`)**:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras

```

---

## License & Acknowledgments

* **Original Code**: Adapted from [Aurélien Géron's handson-ml3 repository](https://github.com/ageron/handson-ml3).
* **Dataset**: Shakespeare text dataset derived from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).
