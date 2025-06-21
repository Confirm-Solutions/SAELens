import os

from dotenv import load_dotenv

from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="EleutherAI/pythia-70m",
    dataset_path="togethercomputer/RedPajama-Data-1T-Sample",  # this is just a tiny test dataset
    shuffle=True,
    num_proc=os.cpu_count()
    or 4,  # increase this number depending on how many CPUs you have
    # tweak these settings depending on the model
    context_size=128,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    # uncomment to upload to huggingface
    hf_repo_id="sidnb13/rp1t-sample-tokenized-pythia-70m",
    # uncomment to save the dataset locally
    save_path="/workspace/SAELens/assets/data/rp1t-sample-tokenized-pythia-70m",
)

if __name__ == "__main__":
    load_dotenv(override=True)
    dataset = PretokenizeRunner(cfg).run()
