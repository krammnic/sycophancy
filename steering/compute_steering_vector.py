import easysteer.hidden_states as hs
from vllm import LLM
import os
import pandas as pd


formatted_positive = pd.read_csv("correct_tasks_train.csv")["Contradiction problem"].tolist()
formatted_negative = pd.read_csv("sycophancy_tasks_train.csv")["Contradiction problem"].tolist()

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=4,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=False
)
all_hidden_states, outputs = hs.get_all_hidden_states_generate(llm, formatted_positive+formatted_negative)

from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector
control_vector = extract_diffmean_control_vector(
    all_hidden_states=all_hidden_states, 
    positive_indices=list(range(86)),
    negative_indices=list(range(86, 117)),
    model_type="qwen3",
    token_pos=-1,
    normalize=True
)
control_vector.export_gguf("qwen8B_steer.gguf")