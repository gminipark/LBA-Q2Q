from dataclasses import dataclass, field
from typing import cast, Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from PIL import Image
import os
import subprocess
from typing import Optional
import torch
import numpy as np


from transformers import AutoModelForVision2Seq, AutoProcessor, ProcessorMixin
from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments,  DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer


from datasets import load_from_disk
import evaluate

from utils import PaddingStrategy


@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """
    model_id: str = field(
      metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: Optional[str] = field(
        default="./vqa_aq",
        metadata={"help": "The preference dataset to use."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=32)
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )
    max_source_length: Optional[int] = field(default=128)
    max_target_length: Optional[int] = field(default=64)
    num_proc: Optional[int] = field(default=0)
    
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    prefix: Optional[str] = field(
        default=None,
    )

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control

def create_and_prepare_model(model_id: str, training_args: Seq2SeqTrainingArguments, script_args):
    
    
    # tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    
    processor.tokenizer.add_tokens("<hl>")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        #use_cache=not training_args.gradient_checkpointing,
        cache_dir=model_id.split("/")[-1] if "/" in model_id else model_id,
        low_cpu_mem_usage=True,
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(processor.tokenizer) > embedding_size:
        model.resize_token_embeddings(len(processor.tokenizer))
    print("model loaded")

    # find all linear modules in model for lora
    target_modules = find_all_linear_names(model)
    print(target_modules)

    # create lora config
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        target_modules=target_modules,
    )
    # enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # pre-process the model by upcasting the layer norms in float 32 for
    # Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
    print("pre-processing model for peft")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)

    # initialize peft model
    print("initializing peft model")
    model = get_peft_model(model, peft_config)

    # logger.info parameters
    model.print_trainable_parameters()

    # tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, processor


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def preprocess_function(examples, 
                        processor,  
                        image_column_name,
                        text_column_name, 
                        target_column_name, 
                        max_source_length, 
                        max_target_length, 
                        prefix=None,
                        padding=False):
    
    model_inputs = {}
    images, inputs, targets = [], [], []
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            image = Image.open("./ambiguous_images/"+str(examples[image_column_name][i])+".jpg")
            images.append(image)
            
            inputs.append(examples[text_column_name][i])
            targets.append(examples[target_column_name][i])
    
    image_inputs = processor(images=images, padding="max_length")
    model_inputs['pixel_values'] = image_inputs['pixel_values']
    

    if prefix:
        inputs = [prefix + " " + inp for inp in inputs]
    text_inputs = processor.tokenizer(inputs, 
                                      max_length=max_source_length,
                                      padding=padding, 
                                      truncation=True,
                                      )
    model_inputs['input_ids'] = text_inputs['input_ids']
    
    # Tokenize targets with the `text_target` keyword argument
    labels = processor.tokenizer(text_target=targets, 
                                 max_length=max_target_length, 
                                 padding=padding,
                                 truncation=True)    
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class CustomDataCollator:
    
    processor: ProcessorMixin
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        image_name = "pixel_values"
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if 
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name and k != image_name} for feature in features]
        
        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.processor.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        if image_name in features[0].keys():
            batch[image_name] = [feature[image_name] for feature in features]
            if return_tensors == "pt":
                import torch
                batch[image_name] = torch.tensor(batch[image_name], dtype=torch.float64)
            elif return_tensors == "tf":
                import tensorflow as tf
                batch[image_name] = tf.constant(batch[image_name], dtype=tf.float64)
            else:
                batch[image_name] = np.array(batch[image_name], dtype=np.float64)

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == "do_not_pad"
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == "max_length" and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.processor.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf
                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids
            

        return batch

def training_function(script_args:ScriptArguments, training_args:Seq2SeqTrainingArguments):

    # Load processed dataset from disk
    dataset = load_from_disk(script_args.dataset_path)
    
    # Load and create peft model
    model, peft_config, processor = create_and_prepare_model(script_args.model_id,training_args, script_args)
    model.config.use_cache = False
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=script_args.preprocessing_num_workers,
                                    fn_kwargs={"processor":  processor, 
                                           'image_column_name': "image_id",
                                           'text_column_name': "ambiguous_question_entity",
                                           'target_column_name': "additional_question",
                                           "max_source_length": script_args.max_source_length,
                                           "max_target_length": script_args.max_target_length,
                                           "padding": 'max_length'},)

    
    train_dataset = tokenized_dataset['train'].select_columns(['pixel_values', 'input_ids', 'labels'])    
    valid_dataset = tokenized_dataset['validation'].select_columns(['pixel_values', 'input_ids', 'labels'])
    
    # Metric
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [[label] for label in decoded_labels]
        # Some simple post-processing
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=references) #, use_stemmer=True)
        #result = {k: round(v * 100, 4) for k, v in result.items()}
        #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        #result["gen_len"] = np.mean(prediction_lens)
        #print(Counter(decoded_preds))
        
        return result
    
    label_pad_token_id= -100
    
    data_collator = CustomDataCollator(
        processor,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        padding=True
    )
    
    # Create trainer and add callbacks
    trainer = Seq2SeqTrainer(model=model, args=training_args, 
                             data_collator=data_collator,
                            train_dataset=train_dataset, 
                            eval_dataset=valid_dataset,
                            compute_metrics=compute_metrics)
    # trainer.accelerator.print(f"{trainer.model}")
    # trainer.model.print_trainable_parameters()
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    
    # Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # TODO: add merge adapters
    # Save everything else on main process
    if trainer.args.process_index == 0:
        if script_args.merge_adapters:
            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            from peft import AutoPeftModelForSeq2SeqLM

            # load PEFT model in fp16
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )  
            # Merge LoRA and base model and save
            model = model.merge_and_unload()        
            model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="8GB"
            )
        else:
            trainer.model.save_pretrained(
                training_args.output_dir, safe_serialization=True
            )

        # save tokenizer 
        processor.save_pretrained(training_args.output_dir)



def main():
    parser = HfArgumentParser([ScriptArguments,Seq2SeqTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(Seq2SeqTrainingArguments, training_args)
    
    print(training_args)
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()