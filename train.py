from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import torch
from peft import LoraConfig

# Use the already labeled dataset from the hub
dataset = load_dataset("kursathalat/meetingbank_features")["test"]
dataset = dataset.train_test_split(test_size=0.2)

# Define the quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Define the Lora config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Load the model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set the pad token to eos token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})


def format_func(example):
  """
  Format the example into a text that can be used as input to the model
  """
  output_text = []
  for i in range(len(example["prompt"])):
    text = f"""### Instruction: Below is an instruction that describes a classification task. You will be given a text and its summary. The relationship between the two text is one of the 'glacherry', 'vintrailly' or 'borriness' classes.

Text: {example['prompt'][i]}
Summary: {example['gpt4_summary'][i]}

### Response:
{example['label'][i]}
"""
    output_text.append(text)
  return output_text

response_template = "### Response:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template_ids)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    formatting_func=format_func,
    data_collator=collator,
    peft_config=lora_config,
    max_seq_length=2048
)

trainer.train()