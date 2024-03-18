# Classification
import numpy as np
from datasets import load_metric

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

test_fold = 0
raw_train_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_train_fold{test_fold}.jsonl")
raw_val_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_val_fold{test_fold}.jsonl")
raw_test_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_test_fold{test_fold}.jsonl")

# This line prints the description of train_ds
raw_train_ds, raw_val_ds, raw_test_ds


BASE_MODEL = "camembert-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5

# Let's name the classes 0, 1, 2, 3, 4 like their indices
id2label = {k:k for k in range(5)}
label2id = {k:k for k in range(5)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)


ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

def preprocess_function(examples):
    label = examples["score"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = label
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=[ "text", "score"])


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    import scipy.stats
    import sklearn.metrics
    
    print("We use the fold with number:", test_fold)
    print(sklearn.metrics.roc_auc_score(labels, logits[:,1]), sklearn.metrics.accuracy_score(labels, predictions))

    
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="../models/camembert-fine-tuned-regression",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate(ds['test'])



# Regression

from datasets import Dataset
test_fold = 0
raw_train_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_trainreg_fold{test_fold}.jsonl")
raw_val_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_valreg_fold{test_fold}.jsonl")
raw_test_ds = Dataset.from_json(f"/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/test_data_word/output_testreg_fold{test_fold}.jsonl")

# This line prints the description of train_ds
raw_train_ds, raw_val_ds, raw_test_ds


from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

BASE_MODEL = "camembert-base"
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

ds = {"train": raw_train_ds, "validation": raw_val_ds, "test": raw_test_ds}

def preprocess_function(examples):
    label = examples["score"] 
    examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    examples["label"] = label
    return examples

for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=[ "text", "score"])


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    import scipy.stats
    import sklearn.metrics


    cor, pval = scipy.stats.pearsonr(labels.flatten(), logits.flatten())
    r2score = sklearn.metrics.r2_score(labels.flatten(), logits.flatten())
    print(cor ," ", r2score)
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="../models/camembert-fine-tuned-regression-2_new",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

import torch

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics_for_regression,
)

trainer.train()

trainer.evaluate(ds['test'])


