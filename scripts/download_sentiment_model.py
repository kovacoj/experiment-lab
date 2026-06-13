from pathlib import Path
import os

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv(
    "SENTIMENT_MODEL_ID",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
)

MODEL_DIR = os.getenv(
    "SENTIMENT_MODEL_DIR",
    ".models/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment",
)


def main() -> None:
    target = Path(MODEL_DIR)
    target.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=MODEL_ID,
        repo_type="model",
        local_dir=str(target),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer*",
            "sentencepiece*",
            "special_tokens_map.json",
            "config.json",
        ],
        ignore_patterns=[
            "*.h5",
            "*.msgpack",
            "*.onnx",
            "*.tflite",
            "tf_model*",
            "flax_model*",
        ],
    )

    print("Downloaded sentiment model")
    print(f"repo: {MODEL_ID}")
    print(f"path: {path}")


if __name__ == "__main__":
    main()
