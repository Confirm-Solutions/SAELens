import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

# Pick device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Minimal config for Pythia-70M, ridge SAE
cfg = LanguageModelSAERunnerConfig(
    model_name="EleutherAI/pythia-70m",  # HuggingFace model name
    model_class_name="HookedTransformer",
    hook_name="blocks.0.hook_mlp_out",  # Standard MLP output hook for layer 0
    hook_layer=0,
    d_in=512,  # Pythia-70M MLP width
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # Tokenized dataset
    is_dataset_tokenized=True,
    streaming=True,
    architecture="ridge",  # <--- THIS IS THE KEY BIT
    expansion_factor=8,  # Reasonable expansion
    train_batch_size_tokens=2048,
    context_size=128,
    n_batches_in_buffer=8,
    training_tokens=1_000_000,  # Small for demo, increase for real training
    store_batch_size_prompts=8,
    lr=2e-4,
    l1_coefficient=5.0,
    device=device,
    dtype="float32",
    seed=42,
    log_to_wandb=False,  # Set True if you want logging
)

# Run training
runner = SAETrainingRunner(cfg)
runner.run()
