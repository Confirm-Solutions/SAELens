import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    PretokenizeRunnerConfig,
)
from sae_lens.sae_training_runner import SAETrainingRunner

load_dotenv()


def setup_custom_models():
    """Add custom models to transformer_lens before training."""
    import transformer_lens.loading_from_pretrained as loading_module

    # Add your custom Llama-based model
    custom_models = [
        "SimpleStories/SimpleStories-1.25M",
        "SimpleStories/SimpleStories-5M",
        "SimpleStories/SimpleStories-11M",
        "SimpleStories/SimpleStories-30M",
        "SimpleStories/SimpleStories-35M",
    ]

    for model in custom_models:
        if model not in loading_module.OFFICIAL_MODEL_NAMES:
            loading_module.OFFICIAL_MODEL_NAMES.append(model)
            # Optionally add aliases
            loading_module.MODEL_ALIASES[model] = [model.split("/")[-1]]


def resolve_device(device_config: str) -> str:
    """Resolve device configuration."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def hydra_cfg_to_lm_sae_cfg(cfg: DictConfig) -> LanguageModelSAERunnerConfig:
    """Convert Hydra config to LanguageModelSAERunnerConfig."""
    # Resolve device configuration
    if "device" in cfg:
        cfg.device = resolve_device(cfg.device)

    return LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distribution)
        model_name=cfg.model_name,
        model_class_name=cfg.model_class_name,
        hook_name=cfg.hook_name,
        hook_eval=cfg.hook_eval,
        hook_layer=cfg.hook_layer,
        hook_head_index=cfg.hook_head_index,
        dataset_path=cfg.dataset_path,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        streaming=cfg.streaming,
        is_dataset_tokenized=cfg.is_dataset_tokenized,
        context_size=cfg.context_size,
        use_cached_activations=cfg.use_cached_activations,
        cached_activations_path=cfg.cached_activations_path,
        # SAE Parameters
        architecture=cfg.architecture,
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        b_dec_init_method=cfg.b_dec_init_method,
        expansion_factor=cfg.expansion_factor,
        activation_fn=cfg.activation_fn,
        activation_fn_kwargs=cfg.activation_fn_kwargs,
        normalize_sae_decoder=cfg.normalize_sae_decoder,
        noise_scale=cfg.noise_scale,
        from_pretrained_path=cfg.from_pretrained_path,
        apply_b_dec_to_input=cfg.apply_b_dec_to_input,
        decoder_orthogonal_init=cfg.decoder_orthogonal_init,
        decoder_heuristic_init=cfg.decoder_heuristic_init,
        init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
        # Activation Store Parameters
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        training_tokens=cfg.training_tokens,
        finetuning_tokens=cfg.finetuning_tokens,
        store_batch_size_prompts=cfg.store_batch_size_prompts,
        normalize_activations=cfg.normalize_activations,
        seqpos_slice=tuple(cfg.seqpos_slice),
        remap_tokens_column=cfg.remap_tokens_column,
        # Misc
        device=cfg.device,
        act_store_device=cfg.act_store_device,
        seed=cfg.seed,
        dtype=cfg.dtype,
        prepend_bos=cfg.prepend_bos,
        # JumpReLU Parameters
        jumprelu_init_threshold=cfg.jumprelu_init_threshold,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        # Performance
        autocast=cfg.autocast,
        autocast_lm=cfg.autocast_lm,
        compile_llm=cfg.compile_llm,
        llm_compilation_mode=cfg.llm_compilation_mode,
        compile_sae=cfg.compile_sae,
        sae_compilation_mode=cfg.sae_compilation_mode,
        # Training Parameters - Batch size
        train_batch_size_tokens=cfg.train_batch_size_tokens,
        # Training Parameters - Adam
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        # Training Parameters - Loss Function
        mse_loss_normalization=cfg.mse_loss_normalization,
        l1_coefficient=cfg.l1_coefficient,
        lp_norm=cfg.lp_norm,
        scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
        l1_warm_up_steps=cfg.l1_warm_up_steps,
        # Training Parameters - Learning Rate Schedule
        lr=cfg.lr,
        lr_scheduler_name=cfg.lr_scheduler_name,
        lr_warm_up_steps=cfg.lr_warm_up_steps,
        lr_end=cfg.lr_end,
        lr_decay_steps=cfg.lr_decay_steps,
        n_restart_cycles=cfg.n_restart_cycles,
        enable_auxk_loss=cfg.enable_auxk_loss,
        enable_dead_neuron_bias_boosting=cfg.enable_dead_neuron_bias_boosting,
        use_random_bias_boost_noise=cfg.use_random_bias_boost_noise,
        dead_neuron_bias_boost_scale=cfg.dead_neuron_bias_boost_scale,
        # Training Parameters - FineTuning
        finetuning_method=cfg.finetuning_method,
        # Resampling protocol args
        use_ghost_grads=cfg.use_ghost_grads,
        feature_sampling_window=cfg.feature_sampling_window,
        dead_feature_window=cfg.dead_feature_window,
        dead_feature_threshold=cfg.dead_feature_threshold,
        # Evals
        n_eval_batches=cfg.n_eval_batches,
        eval_batch_size_prompts=cfg.eval_batch_size_prompts,
        # WANDB
        log_to_wandb=cfg.log_to_wandb,
        log_activations_store_to_wandb=cfg.log_activations_store_to_wandb,
        log_optimizer_state_to_wandb=cfg.log_optimizer_state_to_wandb,
        wandb_project=cfg.wandb_project,
        wandb_id=cfg.wandb_id,
        run_name=cfg.run_name,
        wandb_entity=cfg.wandb_entity,
        wandb_log_frequency=cfg.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.eval_every_n_wandb_logs,
        # Additional wandb.init arguments
        wandb_group=cfg.wandb_group,
        wandb_job_type=cfg.wandb_job_type,
        wandb_tags=cfg.wandb_tags,
        wandb_notes=cfg.wandb_notes,
        wandb_mode=cfg.wandb_mode,
        wandb_resume=cfg.wandb_resume,
        wandb_dir=cfg.wandb_dir,
        wandb_save_code=cfg.wandb_save_code,
        wandb_anonymous=cfg.wandb_anonymous,
        wandb_force=cfg.wandb_force,
        wandb_reinit=cfg.wandb_reinit,
        wandb_resume_from=cfg.wandb_resume_from,
        wandb_fork_from=cfg.wandb_fork_from,
        wandb_sync_tensorboard=cfg.wandb_sync_tensorboard,
        wandb_monitor_gym=cfg.wandb_monitor_gym,
        wandb_config_exclude_keys=cfg.wandb_config_exclude_keys,
        wandb_config_include_keys=cfg.wandb_config_include_keys,
        wandb_allow_val_change=cfg.wandb_allow_val_change,
        wandb_settings=cfg.wandb_settings,
        # Misc
        resume=cfg.resume,
        n_checkpoints=cfg.n_checkpoints,
        checkpoint_path=cfg.checkpoint_path,
        verbose=cfg.verbose,
        model_kwargs=cfg.model_kwargs,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        exclude_special_tokens=cfg.exclude_special_tokens,
        enable_flop_profiling=cfg.enable_flop_profiling,
        flop_profile_interval=cfg.flop_profile_interval,
    )


def hydra_cfg_to_cache_activations_cfg(cfg: DictConfig) -> CacheActivationsRunnerConfig:
    """Convert Hydra config to CacheActivationsRunnerConfig."""
    # Resolve device configuration
    if "device" in cfg:
        cfg.device = resolve_device(cfg.device)

    return CacheActivationsRunnerConfig(
        # Core parameters required by CacheActivationsRunnerConfig
        dataset_path=cfg.dataset_path,
        model_name=cfg.model_name,
        model_batch_size=cfg.model_batch_size,
        hook_name=cfg.hook_name,
        hook_layer=cfg.hook_layer,
        d_in=cfg.d_in,
        training_tokens=cfg.training_tokens,
        # Optional parameters with defaults
        context_size=cfg.context_size,
        model_class_name=cfg.model_class_name,
        new_cached_activations_path=cfg.new_cached_activations_path,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        dtype=cfg.dtype,
        device=cfg.device,
        buffer_size_gb=cfg.buffer_size_gb,
        # Huggingface Integration
        hf_repo_id=cfg.hf_repo_id,
        hf_num_shards=cfg.hf_num_shards,
        hf_revision=cfg.hf_revision,
        hf_is_private_repo=cfg.hf_is_private_repo,
        # Model
        model_kwargs=cfg.model_kwargs,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        compile_llm=cfg.compile_llm,
        llm_compilation_mode=cfg.llm_compilation_mode,
        # Activation Store
        prepend_bos=cfg.prepend_bos,
        seqpos_slice=tuple(cfg.seqpos_slice),
        streaming=cfg.streaming,
        autocast_lm=cfg.autocast_lm,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
    )


def hydra_cfg_to_pretokenize_cfg(cfg: DictConfig) -> PretokenizeRunnerConfig:
    """Convert Hydra config to PretokenizeRunnerConfig."""
    return PretokenizeRunnerConfig(
        tokenizer_name=cfg.tokenizer_name,
        dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset_name,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        split=cfg.split,
        data_files=cfg.data_files,
        data_dir=cfg.data_dir,
        num_proc=cfg.num_proc,
        context_size=cfg.context_size,
        column_name=cfg.column_name,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        streaming=cfg.streaming,
        pretokenize_batch_size=cfg.pretokenize_batch_size,
        begin_batch_token=cfg.begin_batch_token,
        begin_sequence_token=cfg.begin_sequence_token,
        sequence_separator_token=cfg.sequence_separator_token,
        save_path=cfg.save_path,
        hf_repo_id=cfg.hf_repo_id,
        hf_num_shards=cfg.hf_num_shards,
        hf_revision=cfg.hf_revision,
        hf_is_private_repo=cfg.hf_is_private_repo,
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    setup_custom_models()

    # Convert to appropriate config based on mode
    if cfg.mode == "train":
        sae_cfg = hydra_cfg_to_lm_sae_cfg(cfg)
        runner = SAETrainingRunner(sae_cfg, hydra_cfg=cfg)
    elif cfg.mode == "cache_acts":
        cache_cfg = hydra_cfg_to_cache_activations_cfg(cfg)
        runner = CacheActivationsRunner(cache_cfg)
    elif cfg.mode == "pretokenize":
        pretokenize_cfg = hydra_cfg_to_pretokenize_cfg(cfg)
        from sae_lens import PretokenizeRunner

        runner = PretokenizeRunner(pretokenize_cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    runner.run()


if __name__ == "__main__":
    main()
