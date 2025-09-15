import pandas as pd
from generic_realtabformer.realtabformer import REaLTabFormer
import datetime
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
import os


# Define the folder structure
base_dir = "samples"
sub_dirs = ["train", "test", "process_time"]

# Create the directories
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

os.mkdir("saved_models")

df = pd.read_csv("../original_train_dataset/stroke_healthcare_original_train_unlabeled.csv")

rtf_model = REaLTabFormer(
    generic_Config = GPT2Config,
    generic_LMHeadModel = GPT2LMHeadModel,
    my_config = {"n_layer" : 6},
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

begin_time = datetime.datetime.now()
rtf_model.fit(df)
end_time = datetime.datetime.now()

model_name = "gpt2_6_layer_stroke_healthcare"

rtf_model.save(f"saved_models/{model_name}")

n_samples = len(df)

sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/train/{model_name}_train_samples1.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/process_time_{model_name}_train_samples1.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/train/{model_name}_train_samples2.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/process_time_{model_name}_train_samples2.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/train/{model_name}_train_samples3.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/process_time_{model_name}_train_samples3.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/train/{model_name}_train_samples4.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/process_time_{model_name}_train_samples4.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/train/{model_name}_train_samples5.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/process_time_{model_name}_train_samples5.csv")



test_sample = 1110



sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=test_sample)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/test/{model_name}_test_samples1.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/process_time_{model_name}_test_samples1.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=test_sample)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/test/{model_name}_test_samples2.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/process_time_{model_name}_test_samples2.csv")


sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=test_sample)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/test/{model_name}_test_samples3.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/process_time_{model_name}_test_samples3.csv")



sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=test_sample)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/test/{model_name}_test_samples4.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/process_time_{model_name}_test_samples4.csv")



sample_begin = datetime.datetime.now()
samples = rtf_model.sample(n_samples=test_sample)
sample_end = datetime.datetime.now()
samples.to_csv(f"samples/test/{model_name}_test_samples5.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/process_time_{model_name}_test_samples5.csv")



process_time = {
    "begin_time" : begin_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
}

process_time = pd.DataFrame([process_time])
process_time.to_csv(f"samples/process_time/{model_name}_process_time_model_name.csv")


print("Job is done")





