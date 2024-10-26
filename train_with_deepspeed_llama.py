"""
deepspeed --num_gpus=2 train_with_deepspeed_llama.py

deepspeed="ds_config1.json"

{
      "train_micro_batch_size_per_gpu": "auto",
      "zero_optimization": {
        "stage": 1
      }
}

deepspeed="ds_config2.json"

{
    "train_micro_batch_size_per_gpu": "auto",
    "zero_optimization": {
        "stage": 2
    },
    "bf16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": 1e-8,
        "weight_decay": "auto"
        }
    }
}


deepspeed="ds_config3.json"
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "zero_force_ds_cpu_optimizer": false,
    "bf16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": 1e-8,
        "weight_decay": "auto"
        }
    },
    "train_micro_batch_size_per_gpu": "auto"
}

"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch

def print_memory_usage(step, stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Step {step} ({stage}): Allocated: {allocated / (1024 ** 3):.2f} GB, Reserved: {reserved / (1024 ** 3):.2f} GB")

class TrainerMemoryMonitor(Trainer):
    def training_step(self, model, inputs):
        step = self.state.global_step
        print_memory_usage(step, "training_step> before")
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            print_memory_usage(step, "forward pass: -before")
            loss = self.compute_loss(model, inputs)
            print_memory_usage(step, "forward pass: -after")

        # on multi-gpu parallel training use mean() to average 
        if self.args.n_gpu > 1:
            loss = loss.mean()  
        ## clean cache
        torch.cuda.empty_cache()
        print_memory_usage(step, "backward pass\ before")
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        print_memory_usage(step, "backward pass/ after")
        print_memory_usage(step, "training_step> after")
        ## clean cache
        torch.cuda.empty_cache()
        return loss.detach() / self.args.gradient_accumulation_steps


dataset_source = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_source)

base_model = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)
## Change-1 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

## Change-4
if torch.cuda.is_bf16_supported():
    #os.system('pip install flash_attn')
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'
print("-------------------------")
print("**compute_dtype=",compute_dtype)
print("**attn_implementation=",attn_implementation)
print("-------------------------")

## default  -- running 2 python tasks
# model = AutoModelForCausalLM.from_pretrained(base_model)

## use flash attention -- running 3 python tasks
model = AutoModelForCausalLM.from_pretrained(base_model, 
                                             torch_dtype=compute_dtype,
                                             attn_implementation=attn_implementation,
                                             device_map="cuda")
## Change-6 enable gradient checkpoint
model.gradient_checkpointing_enable()
model.config.use_cache=False

batch_size = 1
args = TrainingArguments(
    'outputs',
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_strategy="epoch", 
    eval_strategy="epoch",        
    num_train_epochs=2,    
    ## Change-5 Enable Deepspeed Zero-1
    deepspeed="ds_config1.json",
    # {
    #   "train_micro_batch_size_per_gpu": "auto",
    #   "zero_optimization": {
    #     "stage": 1
    #   }
    # }    
    report_to='none',
    ## Change-2
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),    
    ## Change-3
    remove_unused_columns=True        
)

def tokenize_function(examples):
    MAX_LENGTH=2048
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True)

## Change-6 Enable Deepspeed Zero-1
trainer = TrainerMemoryMonitor(
    model, args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()