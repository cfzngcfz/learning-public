import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
from vlatrainer import VLATrainer
import yaml

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics

# from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator

# from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
import transformers
from transformers import TrainingArguments

my_env = os.environ.copy()
my_env["PATH"] = "/home/chengfangzheng/miniconda3/envs/openvla/bin:" + my_env["PATH"]
os.environ.update(my_env)

import deepspeed

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps logging.Logger
overwatch = initialize_overwatch(__name__)

# local_rank = None

@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        # default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
        # default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_BRIDGE.vla_id) # the bridge dataset
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_JacoPlay.vla_id) # the jaco_play dataset
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/mnt/emc/work_dirs/VLMs/datasets/rlds/unscaled"
    )
    run_root_dir: Path = Path("./runs/train")                       # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 10000                                      # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                 # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = 8
        self.per_device_batch_size = 1                              # 因为在 VLATrainer 中重构了 DataLoader, 且 DataLoader 的 batch size 仅由该参数决定, per_device_train_batch_size 不能为空, 但失去了作用

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # # [Validate] Assert on `expected_world_size`
        # assert (
        #     self.vla.expected_world_size == overwatch.world_size()
        # ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

# fmt: on



from prismatic.models.vlms import PrismaticVLM
from torch.utils.data import DataLoader, IterableDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm
from collections import OrderedDict



def save_checkpoint(
    model,
    run_dir: Path,
    global_step: int,
    epoch: int,
    train_loss: Optional[float] = None,
    only_trainable: bool = True,
) -> None:
    """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""

    full_vlm_state_dict = model.state_dict()
    model_state_dicts = {
        mkey: OrderedDict() for mkey in (model.trainable_module_keys if only_trainable else model.all_module_keys)
    }

    # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
    for key, param in full_vlm_state_dict.items():
        for mkey in model_state_dicts:
            if key.startswith(mprefix := f"{mkey}."):
                model_state_dicts[mkey][key.removeprefix(mprefix)] = param

    # Save on rank zero *only*
    if deepspeed.comm.get_rank() == 0:
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = (
                checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
            )

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts}, checkpoint_path)

        # TODO (siddk) :: This breaks w/ Sagemaker default permissions (root vs. <user>)... skip?
        # shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")


def main():
    parser = transformers.HfArgumentParser((TrainConfig, TrainingArguments))
    cfg, training_args = parser.parse_args_into_dataclasses()

    if not dist.is_initialized():
        print("test >>>> run deepspeed.init_distributed")
        deepspeed.init_distributed()
    
    script_path = os.path.abspath(__file__)
    config_path = os.path.join(script_path, "../zero3_offload.json")
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as file:
        ds_config = json.load(file)

    cfg.per_device_batch_size = ds_config["train_micro_batch_size_per_gpu"]
    world_size = torch.distributed.get_world_size()
    cfg.global_batch_size = cfg.per_device_batch_size * world_size
    if ds_config["train_batch_size"] != cfg.global_batch_size:
        raise ValueError("Need to check `train_batch_size` in deepspeed json configuration")

    print("test >>>> batch size per GPU =", cfg.per_device_batch_size)
    print("test >>>> total number of GPUs =", world_size)
    print("test >>>> global batch size =", cfg.global_batch_size)



    print("OpenVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    # print('"Do or do not; there is no try."', ctx_level=1)

    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if deepspeed.comm.get_rank() == 0:
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    print(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        # vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True)
        vlm = load_vla(cfg.pretrained_checkpoint, load_for_training=True)

    else:
        # vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)
        path_model = Path("/mnt/emc/work_dirs/VLMs/models/prismatic-vlms/" + cfg.vla.base_vlm)
        vlm = load(path_model, load_for_training=True)

    # # [Validate] Model should be in Full Precision!
    # for param in vlm.parameters():
    #     assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    print(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )


    # Get VLA Dataset & Collator
    print(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )


    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    print(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )


    # 初始化
    vlm, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=vlm,
        model_parameters=vlm.parameters(),
        config=ds_config,
    )


    """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
    assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"

    # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    # === Train ===
    status = metrics.get_status()
    with tqdm(
        total=(cfg.epochs * len(dataloader)) if cfg.max_steps is None else cfg.max_steps,
        desc=status,
        leave=False,
        disable=not overwatch.is_rank_zero(),
    ) as progress:
        vlm.train()

        # Zero Gradients (just in case)
        optimizer.zero_grad()

        # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
        #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
        #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
        for batch in dataloader:
            # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
            #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                # [Contract] vlm.forward() must automatically compute `loss` and return!
                output: CausalLMOutputWithPast = vlm(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = output.loss

            print("cfz test >>>>", deepspeed.comm.get_rank(), ", loss =", loss.item())

            # Commit Loss =>> Backward!
            metrics.commit(loss=loss)
            
            if deepspeed.comm.get_world_size() > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.


            vlm.backward(loss)
            # if ds_config["bf16"]["enabled"] == True:
            #     optimizer.backward(loss)
            # else:
            #     loss.backward()


            # === Compute Action Token Accuracy & L1 Loss ===

            # To compute action token accuracy, we need to identify the locations of the action tokens
            # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
            # insert `vlm.vision_backbone.num_patches` at index 1.
            #
            # Computing `action_prediction_accuracy` is then pretty straightforward:
            #   1) Extract "aligned" predictions & labels
            #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
            #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
            #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
            action_preds = output.logits[:, vlm.vision_backbone.num_patches : -1].argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Commit Metrics
            metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)


            # === Gradient Step ===

            # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
            # self.clip_grad_norm() # todo

            # # Optimizer & LR Scheduler Step
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            # vlm.step()

            # Compute epoch value using number of completed gradient steps
            epoch = (metrics.global_step + 1) // (len(vla_dataset) // cfg.global_batch_size)

            # Push Metrics
            metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=lr_scheduler.get_last_lr()[0])
            status = metrics.push()

            # Check for Save Interval or Max Steps & Save Checkpoint
            if (terminate := (cfg.max_steps is not None and metrics.global_step >= cfg.max_steps)) or (
                (metrics.global_step % cfg.save_interval) == 0
            ):
                save_checkpoint(
                    metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=False
                )
                dist.barrier()

                if terminate:
                    return

            # Update Progress Bar
            progress.update()
            progress.set_description(status)

    # Finalize
    print("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    print("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()