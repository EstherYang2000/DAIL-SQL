import argparse
import os
import json

import openai
from tqdm import tqdm

from llm.chatgpt import init_chatgpt, ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader

from utils.post_process import process_duplication, get_sqls

# 問題檔案名稱
QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    # 1. 解析命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)  # 問題檔案的路徑
    parser.add_argument("--openai_api_key", type=str) # OpenAI API 金鑰
    parser.add_argument("--openai_group_id", type=str, default="org-kFNEXr1eeRwlWrIjAILJBo2f") # OpenAI 群組 ID
    parser.add_argument("--model", type=str, choices=[LLM.TEXT_DAVINCI_003, 
                                                      LLM.GPT_35_TURBO,
                                                      LLM.GPT_35_TURBO_0613,
                                                      # LLM.TONG_YI_QIAN_WEN,
                                                      LLM.GPT_35_TURBO_16K,
                                                      LLM.GPT_4], # 模型選擇
                        default=LLM.GPT_35_TURBO)
    parser.add_argument("--start_index", type=int, default=0) # 問題起始索引
    parser.add_argument("--end_index", type=int, default=1000000) # 問題結束索引
    parser.add_argument("--temperature", type=float, default=0) # 溫度參數（控制生成隨機性）
    parser.add_argument("--mini_index_path", type=str, default="") # 迷你索引檔案的路徑
    parser.add_argument("--batch_size", type=int, default=1) # 批次大小
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set") # 自洽集的大小
    parser.add_argument("--db_dir", type=str, default="dataset/spider/database") # 資料庫的目錄
    args = parser.parse_args()

    # check args
    # 2. 檢查批次大小和模型是否兼容
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"
    # 3. 加載問題檔案
    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]
    # 4. 初始化 OpenAI API
    # init openai api
    init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)
    # 5. 設置檔案寫入模式
    if args.start_index == 0:
        mode = "w" # 從頭開始寫入
    else:
        mode = "a" # 追加模式
    # 6. 如果指定了迷你索引檔案，只取部分問題
    if args.mini_index_path:
        mini_index = json.load(open(args.mini_index_path, 'r'))
        questions = [questions[i] for i in mini_index]
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"
    # 7. 創建 DataLoader 以支持批次處理
    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # 8. 開始批次處理問題並生成 SQL 查詢
    token_cnt = 0 # 計算 token 的總數
    with open(out_file, mode) as f:
        for i, batch in enumerate(tqdm(question_loader)): # 迭代每一批問題
            # 檢查索引範圍，跳過不在範圍內的批次
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            try:
                # 9. 向 LLM 發送請求，生成 SQL 查詢
                res = ask_llm(args.model, batch, args.temperature, args.n)
            except openai.error.InvalidRequestError:
                print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                res = ""
            # 10. 解析並處理生成結果
            # parse result
            token_cnt += res["total_tokens"] # 累加 token 數量
            if args.n == 1: # 單次生成模式
                for sql in res["response"]:
                    # remove \n and extra spaces  # 處理 SQL 結果，移除多餘的空格和換行符
                    sql = " ".join(sql.replace("\n", " ").split())
                    sql = process_duplication(sql)
                    # python version should >= 3.8
                    if sql.startswith("SELECT"):
                        f.write(sql + "\n")
                    elif sql.startswith(" "):
                        f.write("SELECT" + sql + "\n")
                    else:
                        f.write("SELECT " + sql + "\n")
            else: # 多次生成模式（自洽模式）
                results = []
                cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                for sqls, db_id in zip(res["response"], cur_db_ids):
                    processed_sqls = []
                    for sql in sqls: # 處理每個生成的 SQL
                        sql = " ".join(sql.replace("\n", " ").split())
                        sql = process_duplication(sql)
                        if sql.startswith("SELECT"):
                            pass
                        elif sql.startswith(" "):
                            sql = "SELECT" + sql
                        else:
                            sql = "SELECT " + sql
                        processed_sqls.append(sql)
                    result = { # 構建結果並調用 get_sqls 進行最終處理
                        'db_id': db_id,
                        'p_sqls': processed_sqls
                    }
                    final_sqls = get_sqls([result], args.n, args.db_dir)

                    for sql in final_sqls:
                        f.write(sql + "\n")

