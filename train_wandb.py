from collections import OrderedDict
import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from qwen2_model import Transformer, Lora
from tokenizer import Tokenizer

def apply_lora(model: Transformer, r=8, alpha=32, dropout=0.0):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.layers:
        block.mlp.up_proj = Lora(block.mlp.up_proj, r, alpha, dropout)
        # block.mlp.down_proj = Lora(block.mlp.down_proj, r, alpha, dropout)
        # block.mlp.gate_proj = Lora(block.mlp.gate_proj, r, alpha, dropout)
    return model

@torch.no_grad()
def export_lora_merged_state_dict(model: Transformer):
    "Return a state_dict with LoRA deltas merged into feed_forward weights & adapter weights removed."
    og_state_dict = model.state_dict()
    merged = {}
    skip = set()
    for name, module in model.named_modules():
        if isinstance(module, Lora):
            base_key = f"{name}.linear.weight"
            skip.add(base_key)
             # Match with original transformer weight key
            merged_key = base_key.replace('.linear.weight', '.weight')

            base_w = module.linear.weight.data
            delta = module.scale * (module.B.weight.data @ module.A.weight.data)
            merged[merged_key] = (base_w + delta).to(base_w.dtype)
            skip.add(f"{name}.A.weight")
            skip.add(f"{name}.B.weight")

    out = OrderedDict()
    for key, value in og_state_dict.items():
        if key in skip: # skip LoRA weights
            continue
        if key in merged: # replace with merged weights
            out[key] = merged[key].clone()
        else: # keep original weights
            out[key] = value
    return out

def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    if config["training"]["use_lora"]:
        run_name = f"LoRA-rank_{config['training']['lora_rank']}-alpha_{config['training']['lora_alpha']}-dropout_{config['training']['lora_dropout']}-{current_time}"
    else:
        run_name = f"FullFT-{current_time}"

    log_dir_path = Path(config["training"]["log_dir"])
    log_dir_path.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=config.get("wandb_project", "GRPO-Zero"),
        name=run_name,
        dir=str(log_dir_path),
        config=config,
    )
    
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    if config["training"]["use_lora"]:
        apply_lora(
            model,
            config["training"]["lora_rank"],
            config["training"]["lora_alpha"],
            config["training"]["lora_dropout"],
        )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        update_policy_start = time.time()
        results = update_policy(
            model=model,
            trainable_parameters=trainable_parameters,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        update_policy_end = time.time()
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        update_duration = update_policy_end - update_policy_start
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            wandb.log({"success_rate/eval": eval_success_rate}, step=step)

        metrics = {
            "loss": loss,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "success_rate/train": success_rate,
            "format_reward": format_reward,
            "grad_norm": grad_norm,
            "duration": duration,
            "num_finished_episodes": num_finished_episodes,
            "learning_rate": lr,
            "mean_response_len": mean_response_len,
            "entropy": entropy,
            "update_duration": update_duration,
        }
        wandb.log(metrics, step=step)
        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            wandb.log({f"text_{i}": wandb.Html(f"<pre>{text}</pre>")}, step=step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            if config["training"]["use_lora"]:
                torch.save(export_lora_merged_state_dict(model), output_file)
            else:
                torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")

    wandb_run.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
