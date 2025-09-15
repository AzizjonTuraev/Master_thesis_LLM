from generic_be_great import GReaT
import pandas as pd
import datetime
import os
from transformers import LlamaTokenizerFast
from transformers.models.llama import LlamaConfig, LlamaForCausalLM



# Define the folder structure
base_dir = "samples"
sub_dirs = ["train", "test", "process_time"]


# Create the directories
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

os.mkdir("saved_models")


model_name_or_path = "hf-internal-testing/llama-tokenizer"
tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)

df = pd.read_csv("../original_train_dataset/housing_original_train_unlabeled.csv")
model = GReaT(llm=LlamaForCausalLM, config=LlamaConfig, batch_size=32, tokenizer=tokenizer,
                epochs=200, my_config = {"num_hidden_layers" : 2})


begin_time = datetime.datetime.now()
model.fit(df.head(700))
end_time = datetime.datetime.now()

model_name = "llama_2_layer_housing_test"
model.save(f"saved_models/{model_name}")
n_samples = len(df)

sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples, max_length=2000)
sample_end = datetime.datetime.now()
synthetic_data.to_csv(f"samples/train/samples1.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/train_process_time_sample1.csv")


sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
synthetic_data.to_csv(f"samples/train/samples2.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/train_process_time_sample2.csv")


sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
synthetic_data.to_csv(f"samples/train/samples3.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/train_process_time_sample3.csv")


sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
synthetic_data.to_csv(f"samples/train/samples4.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/train_process_time_sample4.csv")


sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()
synthetic_data.to_csv(f"samples/train/samples5.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/process_time/train_process_time_sample5.csv")


process_time = {
    "begin_time" : begin_time,
    "end_time" : end_time
}
process_time = pd.DataFrame([process_time])
process_time.to_csv(f"samples/process_time/process_time.csv")



n_samples=70

sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()

synthetic_data.to_csv(f"samples/test/sample1.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/test_process_time_sample1.csv")



sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()

synthetic_data.to_csv(f"samples/test/sample2.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/test_process_time_sample2.csv")



sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()

synthetic_data.to_csv(f"samples/test/sample3.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/test_process_time_sample3.csv")




sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()

synthetic_data.to_csv(f"samples/test/sample4.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/test_process_time_sample4.csv")




sample_begin = datetime.datetime.now()
synthetic_data = model.sample(n_samples=n_samples)
sample_end = datetime.datetime.now()

synthetic_data.to_csv(f"samples/test/sample5.csv")
sample_time = {
    "begin_time" : sample_begin.strftime("%Y-%m-%d %H:%M:%S.%f"),
    "end_time" : sample_end.strftime("%Y-%m-%d %H:%M:%S.%f")
}
sample_time = pd.DataFrame([sample_time])
sample_time.to_csv(f"samples/test/test_process_time_sample5.csv")






print("Job is done")




