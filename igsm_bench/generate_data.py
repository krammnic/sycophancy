from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from typing import Literal
import pandas as pd
import time

fix_seed(23)

def gen_task(max_op: int, max_edge: int):
    attempt, max_attempts = 0, 50
    while attempt < max_attempts:
        try:
            id_gen = IdGen(
                max_op=max_op,
                max_edge=max_edge,
                perm_level=5,
                detail_level=0
            )
            
            id_gen.gen_prob([i for i in range(23)], p_format="pq")
            
            return id_gen

        except RuntimeError as e:
            attempt += 1
        except Exception as e:
            print(f"Unexcpected exception: {type(e).__name__}: {e}")
            attempt += 1
    
    return None

def main():
    fix_seed(17)
    l_op, r_op = 5, 6
    l_edge, r_edge = 12, 13
    max_iter = 1000

    os.makedirs("logs/gen", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    start = time.time()
    for max_op in range(l_op, r_op):
        for max_edge in range(l_edge, r_edge):
            corr_flag = True
            data = {
                "Contradiction problem": [],
                "Problem": [],
                "Solution": [],
                "Answer": [],
                "Contradiction": [],
                "Difficulty": []
            }
            for iter in range(max_iter):
                id_gen = gen_task(max_op, max_edge)

                with open(f"logs/gen/difficulty_{max_op}_{max_edge}_low.txt", "w", encoding="utf-8") as file:
                    cur = time.time()
                    execution_time = cur - start
                    print(f"{iter}: {execution_time:.2f}, ", end='', file=file)
                    if id_gen is None:
                        file.write("Failed to generate task")
                    else:
                        file.write("Task generated successfully")

                if id_gen is None:
                    corr_flag = False
                    continue
                
                data["Contradiction problem"].append(tokenizer.decode(id_gen.contr_prob_token))
                data["Problem"].append(tokenizer.decode(id_gen.prob_token))
                data["Solution"].append(tokenizer.decode(id_gen.sol_token))
                data["Answer"].append(tokenizer.decode(id_gen.ans_token))
                data["Contradiction"].append(tokenizer.decode(id_gen.contr_token))
                data["Difficulty"].append(f"{max_op}_{max_edge}")

            with open(f"logs/gen/difficulty_{max_op}_{max_edge}_low.txt", "w", encoding="utf-8") as file:
                cur = time.time()
                execution_time = cur - start
                print(f"dif_{max_op}_{max_edge}: {execution_time:.2f}, ", end='', file=file)
                if corr_flag:
                    file.write("All tasks generated successfully")
                else:
                    file.write("Something went wrong")
            cur = time.time()
            execution_time = cur - start
            print(f"dif_{max_op}_{max_edge}: {execution_time:.2f}, ", end='')
            if corr_flag:
                print("All tasks generated successfully")
            else:
                print("Something went wrong")

            df = pd.DataFrame(data)
            df.to_csv(f'data/iGSM_bench_data_{max_op}_{max_edge}_low.csv')

if __name__ == "__main__":
    main()
# for iter in range(500):
#     print(iter)
#     for difficulty in ["med"]:
#         id_gen = gen_task(difficulty)
#         data["Contradiction problem"].append(tokenizer.decode(id_gen.contr_prob_token))
#         data["Problem"].append(tokenizer.decode(id_gen.prob_token))
#         data["Solution"].append(tokenizer.decode(id_gen.sol_token))
#         data["Answer"].append(tokenizer.decode(id_gen.ans_token))
#         data["Contradiction"].append(tokenizer.decode(id_gen.contr_token))
#         data["Difficulty"].append(difficulty)
# df = pd.DataFrame(data)
# df.to_csv('data/iGSM_bench_data.csv')
