# utils
def extract_config(config, *argv):
    assert len(argv) > 0, "No keys to extract"
    config_values = []
    for key in argv:
        assert key in config, f"Key '{key}' not in config"
        config_values.append(config[key])

    return tuple(config_values) if len(argv) > 1 else config_values[0]


def validate_config(config):
    embedding_dimension, n_attention_heads = extract_config(
        config, "embedding_dimension", "n_attention_heads")

    # embedding dimension must be divisible by n_attention_heads
    assert embedding_dimension % n_attention_heads == 0, f"Embedding dimension ({embedding_dimension}) must be divisible by n_attention_heads ({n_attention_heads})"


def emb_to_string(emb, vocab):
    embeddings = vocab.itos
    words = [embeddings[item] for item in emb]
    return ' '.join(words)


if __name__ == "__main__":
  config = {
    "embedding_dimension": 200,
    "ff_dimension": 200,
    "n_attention_heads": 2,
    "n_encoder_layers": 0,
    "n_decoder_layers": 2,
    "dataset": "Dataset.PennTreebank",
    "segmentation": "Segmentation.Word",
    "max_seq_len": 35,
    "batch_size": 20,
    "eval_batch_size": 10,
    "dropout": 0.2,
    "n_epochs": 3,
    "learning_rate": 0.5,
    "loss_criterion": "CrossEntropyLoss"
  }

  validate_config(config)

  embedding_dimension, n_attention_heads, n_encoder_layers, n_decoder_layers, ff_dimension, dropout, batch_size, eval_batch_size, learning_rate = extract_config(
      config, "embedding_dimension", "n_attention_heads", "n_encoder_layers", "n_decoder_layers", "ff_dimension", "dropout", "batch_size", "eval_batch_size", "learning_rate")

  print("utils.py run complete")