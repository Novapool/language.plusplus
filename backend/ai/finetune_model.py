import torch
import numpy as np
from datasets import load_from_disk
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from datasets import load_metric

# Load WER metric
wer_metric = load_metric("wer")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Load the prepared dataset
dataset = load_from_disk("./dataset_prep/prepared_dataset")

# Load the pre-trained model and processor
model_name = "./wav2vec2-ljspeech-gruut"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Prepare the dataset
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    group_by_length=True,
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    num_train_epochs=20,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./spanish_phoneme_model")
processor.save_pretrained("./spanish_phoneme_model")