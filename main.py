from lib.MCTS import MCTS
import argparse
from lib.benchmark import BENCHMARK
import pandas as pd
from dataset.dataset import drop_nan

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="model to run topic generation with ",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="max tokens to get"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="temperature for generation"
    )
    parser.add_argument("--candidate_space", type=int, default="2",
                        help="candidate space")
    parser.add_argument("--top_p", type=float, default=0.5,
                        help="top-p for generation")
    parser.add_argument("--train", type=bool, default=False,
                        help="train mode")
    parser.add_argument("--dataset", type=str, default="20news",
                        help="select data")
    parser.add_argument("--benchmark", type=bool, default=True,
                        help="is benchmark")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.dataset == "20news":
        if args.train:
            pass
        else:
            data_path = "./dataset/20news_groups/test.csv"
            data_df = pd.read_csv(data_path)
            data_df = drop_nan(data_df)
            max_df_len = len(data_df)
            docs,hard_label_list,soft_label_list = [],[],[]
            for i,row in data_df[0:1000].iterrows():
                doc = row["doc"]
                hard_label = f'{row["primary_topic"]}.{row["secondary_topic"]}.{row["tertiary_topic"]}.{row["rest_topic"]}'
                if args.benchmark:
                    filename = "benchmark_prediction.csv"
                    only_llm = BENCHMARK(doc=doc,
                                         model_ckpt=args.model_ckpt,
                                         temperature=args.temperature,
                                         top_p=args.top_p,
                                         max_tokens=args.max_tokens)
                    try:
                        soft_labels = only_llm.only_llm_benchmark()
                        soft_label = ".".join(soft_labels)
                    except:
                        continue

                else:
                    filename = "MCTS_prediction.csv"
                    mcts = MCTS(doc = doc,
                                candidate_space=args.candidate_space,
                                model_ckpt=args.model_ckpt,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens)
                    try:
                        soft_labels = mcts.tree_search()
                        soft_label = ".".join(soft_labels)
                    except:
                        continue
                    

                print("#"*20)
                print(f"index:{i}/{max_df_len}")
                print(f"doc:{doc}")
                print(f"hard_label:{hard_label}")
                print(f"soft_label:{soft_label}")
                

                docs.append(doc)
                hard_label_list.append(hard_label)
                soft_label_list.append(soft_label)
            

            df = pd.DataFrame(
                {
                    "doc":docs,
                    "hard_label":hard_label_list,
                    "soft_label" :soft_label_list
                }
            )
            df.to_csv(f"./dataset/20news_groups/{filename}",index=False,encoding="utf-8")
    else:
        pass
if __name__ == "__main__":
    main()

