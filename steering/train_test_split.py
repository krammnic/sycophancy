import pandas as pd

correct = pd.read_csv("correct_tasks_local.csv")
sycophancy = pd.read_csv("sycophancy_tasks_local.csv")

BENCH_RESULT = 0.26
TEST_SIZE = 100

syco_test_count = int(TEST_SIZE * BENCH_RESULT)
corr_test_count = TEST_SIZE - syco_test_count

n_corr = correct.shape[0]
n_syco = sycophancy.shape[0]

correct_train = correct[:n_corr - corr_test_count]
correct_test = correct[n_corr - corr_test_count:]

sycophancy_train = sycophancy[:n_syco - syco_test_count]
sycophancy_test = sycophancy[n_syco - syco_test_count:]

correct_train.to_csv("correct_tasks_train.csv", index=False)
correct_test.to_csv("correct_tasks_test.csv", index=False)

sycophancy_train.to_csv("sycophancy_tasks_train.csv", index=False)
sycophancy_test.to_csv("sycophancy_tasks_test.csv", index=False)