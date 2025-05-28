import json
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformer_lens import HookedTransformer

from sae_lens.evals import EvalConfig, run_evals
from sae_lens.sae import SAE
from sae_lens.training.activations_store import ActivationsStore


def try_load_hydra_config(checkpoint_path: str) -> dict[str, Any]:
    """
    If hydra_config.yaml exists in the checkpoint directory, load it and return as dict.
    """
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    hydra_config_path = checkpoint_dir / "hydra_config.yaml"
    if hydra_config_path.exists():
        with open(hydra_config_path) as f:
            hydra_cfg = OmegaConf.load(f)
        return dict(hydra_cfg)
    return {}


def override_eval_cfg_with_hydra(
    eval_cfg: EvalConfig, hydra_cfg: dict[str, Any]
) -> None:
    # Only override if present in hydra_cfg
    for k in ["hook_name", "context_size", "model_name", "dataset"]:
        if k in hydra_cfg and hydra_cfg[k] is not None:
            setattr(eval_cfg, k, hydra_cfg[k])


@hydra.main(config_path="../config", config_name="config_eval", version_base=None)
def main(cfg: DictConfig):
    # Check checkpoint path
    if not os.path.exists(cfg.sae_checkpoint_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {cfg.sae_checkpoint_path}")

    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    # Optionally load hydra config from checkpoint dir and override fields
    hydra_cfg = try_load_hydra_config(cfg.sae_checkpoint_path)
    if hydra_cfg:
        print(
            f"Loaded hydra_config.yaml from checkpoint. Overriding fields: {[k for k in ['hook_name', 'context_size', 'model_name', 'dataset'] if k in hydra_cfg]}"
        )

    # Prepare eval config
    eval_config = EvalConfig(
        batch_size_prompts=cfg.batch_size_prompts,
        n_eval_reconstruction_batches=cfg.n_eval_reconstruction_batches,
        compute_kl=cfg.compute_kl,
        compute_ce_loss=cfg.compute_ce_loss,
        n_eval_sparsity_variance_batches=cfg.n_eval_sparsity_variance_batches,
        compute_l2_norms=cfg.compute_l2_norms,
        compute_sparsity_metrics=cfg.compute_sparsity_metrics,
        compute_variance_metrics=cfg.compute_variance_metrics,
        compute_featurewise_density_statistics=cfg.compute_featurewise_density_statistics,
        compute_featurewise_weight_based_metrics=cfg.compute_featurewise_weight_based_metrics,
    )
    # Override with hydra config if present
    override_eval_cfg_with_hydra(eval_config, hydra_cfg)

    # Load SAE using load_from_disk
    sae = SAE.load_from_disk(cfg.sae_checkpoint_path, device=device)

    # Load model (possibly overridden)
    model_name = getattr(eval_config, "model_name", cfg.model_name)
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device)

    # Prepare activation store (possibly overridden)
    context_size = getattr(eval_config, "context_size", cfg.context_size)
    dataset = getattr(eval_config, "dataset", cfg.dataset)
    activation_store = ActivationsStore.from_sae(
        model, sae, context_size=context_size, dataset=dataset
    )
    activation_store.shuffle_input_dataset(seed=42)

    # Run evaluation
    metrics, feature_metrics = run_evals(
        sae=sae,
        activation_store=activation_store,
        model=model,
        eval_config=eval_config,
        ignore_tokens={
            getattr(model.tokenizer, "pad_token_id", None),
            getattr(model.tokenizer, "eos_token_id", None),
            getattr(model.tokenizer, "bos_token_id", None),
        },
        verbose=cfg.verbose,
    )

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "sae_checkpoint_path": cfg.sae_checkpoint_path,
        "model_name": model_name,
        "hook_name": getattr(eval_config, "hook_name", cfg.hook_name),
        "dataset": dataset,
        "context_size": context_size,
        "metrics": metrics,
        "feature_metrics": feature_metrics,
    }
    out_path = output_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {out_path}\n")
    print("Summary of metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
