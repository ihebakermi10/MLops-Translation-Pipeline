import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict

def load_error_data(error_data_file):
    print(f"Loading error data from {error_data_file}")
    return load_dataset('json', data_files=error_data_file, split='train')

def retrain_model(model_checkpoint, lang_source, data_files):
    try:
        print(f"Starting retraining for model: {model_checkpoint} with data file: {data_files}")
        # Disable wandb
        os.environ["WANDB_DISABLED"] = "true"

        # Load tokenizer and model
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

        # Load and prepare datasets
        print("Loading and preparing datasets...")
        raw_datasets = load_dataset("json", data_files=data_files)
        print(f"Datasets loaded: {raw_datasets}")

        total_size = len(raw_datasets["train"])
        if total_size == 0:
            print("No samples found in the error data. Skipping retraining.")
            return

        train_size = int( 0.7*total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Create train, validation, and test splits
        print("Creating train, validation, and test splits...")
        train_dataset = raw_datasets["train"].select(range(train_size)) if train_size > 0 else None
        val_dataset = raw_datasets["train"].select(range(train_size, train_size + val_size)) if val_size > 0 else None
        test_dataset = raw_datasets["train"].select(range(train_size + val_size, total_size)) if test_size > 0 else None

        if train_dataset is None or len(train_dataset) == 0:
            print("Training dataset is empty. Skipping retraining.")
            return

        if val_dataset is None or len(val_dataset) == 0:
            print("Validation dataset is empty. Using training dataset as validation dataset.")
            val_dataset = train_dataset

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        print(f"Dataset splits: {dataset_dict}")

        # Define preprocessing function
        def preprocess_function(examples):
            inputs = [ex[lang_source] for ex in examples["translation"]]
            targets = [ex['en'] for ex in examples["translation"]]  # Assuming the target language is English
            model_inputs = tokenizer(inputs, max_length=128, truncation=True)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(text_target=targets, max_length=128, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Tokenize all datasets
        print("Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)
        print(f"Tokenized datasets: {tokenized_datasets}")

        # Set training arguments
        print("Setting training arguments...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{model_checkpoint}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            report_to="none"  # Disabling wandb reporting
        )

        # Initialize data collator, trainer and start training
        print("Initializing data collator and trainer...")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        print("Starting training...")
        trainer.train()
        print(f"Pushing model to hub under the name: {model_checkpoint}...")
        trainer.push_to_hub(model_checkpoint)

        # Reset the count after retraining by removing the error data file
        if os.path.exists(data_files):
            os.remove(data_files)
        print("Retraining complete and error data file removed.")

    except Exception as e:
        print("Failed to retrain model:", str(e))

if __name__ == "__main__":
    import sys
    model_checkpoint = sys.argv[1]
    lang_source = sys.argv[2]
    data_files = sys.argv[3]
    raw_datasets = load_error_data(data_files)
    retrain_model(model_checkpoint, lang_source, data_files)
