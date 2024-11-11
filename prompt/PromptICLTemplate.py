from utils.utils import get_tokenizer, count_tokens, jaccard_similarity
import numpy as np
import json

class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, tokenizer: str, *args, **kwargs):
        self.tokenizer = get_tokenizer(tokenizer)
        self.example_qualities = []
        self.pattern_similarities = []

    def record_example_quality(self, examples, target):
        quality_list = []
        for example in examples:
            quality_list.append(jaccard_similarity(example["query_skeleton"], target["query_skeleton"]))
        self.example_qualities.append(quality_list)

    def get_example_quality(self):
        if self.example_qualities:
            return np.mean([num for row in self.example_qualities for num in row])
        else:
            return 1

    def get_example_quality_for_each(self):
        if self.example_qualities:
            return [np.mean(row) for row in self.example_qualities]
        else:
            return []

    def record_pattern_similarity(self, examples, target):
        similarity_list = []
        for example in examples:
            similarity_list.append(jaccard_similarity(example["question_pattern"], target["question_pattern"]))
        self.pattern_similarities.append(similarity_list)

    def get_pattern_similarity(self):
        if self.pattern_similarities:
            return np.mean([num for row in self.pattern_similarities for num in row])
        else:
            return 1

    def format(self, target: dict, max_seq_len: int, max_ans_len: int, scope_factor: int, cross_domain=False, *args, **kwargs):
        print("-----------------------------------format-----------------------------------")
        # target question
        # 1. 格式化目標問題，並計算其 token 數量
        prompt_target = self.format_target(target) # 格式化目標問題
        sum_tokens = count_tokens(prompt_target, tokenizer=self.tokenizer) # 計算目標問題的 token 數量
        
        if self.NUM_EXAMPLE != 0: # 當使用範例學習（k-shot）時
            # example questions 
            # 2. 獲取候選範例問題
            examples = self.get_examples(target, self.NUM_EXAMPLE * scope_factor, cross_domain=cross_domain)
            prompt_example = list() # 儲存格式化後的範例
            question = target["question"] # 目標問題文本
            example_prefix = self.get_example_prefix() # 範例的前綴
            selected_examples = [] # 儲存被選中的範例
            for example in examples:
                example_question = example["question"]
                # assert example_question != question, f"Example is the same with target question: {question}!, \n{target}\n{example}"
                # 確保範例問題不與目標問題相同
                if cross_domain:
                    assert target["db_id"] != example["db_id"] # 確保範例與目標問題來自不同的資料庫

                example_format = self.format_example(example) # 格式化範例問題
                
                # count tokens and drop the example if exceed max_len
                # 3. 檢查範例是否會超過最大序列長度
                forward_tokens = count_tokens(example_prefix + self.SEP_EXAMPLE.join(prompt_example + [example_format, prompt_target]), tokenizer=self.tokenizer)
                
                if forward_tokens + max_ans_len <= max_seq_len: # 確保加上答案長度後不超過限制
                    # add an example
                    # 添加範例
                    prompt_example.append(example_format) 
                    # update tokens
                    sum_tokens = forward_tokens # 更新當前的總 token 數
                    # record the selected examples
                    selected_examples.append(example) # 記錄選中的範例
                    
                    if len(prompt_example) >= self.NUM_EXAMPLE: # 當範例數量達到指定值，則停止
                        break
            # 記錄範例質量和模式相似性（評估所選範例的質量和模式）
            # print(selected_examples)
            print(target)
            self.record_example_quality(selected_examples, target)
            self.record_pattern_similarity(selected_examples, target)
            
            n_valid_example = len(prompt_example) # 紀錄有效範例的數量
            # 4. 組合提示，包含範例和目標問題
            if len(prompt_example) > 0:
                prompt = example_prefix + self.SEP_EXAMPLE.join(prompt_example + [prompt_target])
            else:
                prompt = self.SEP_EXAMPLE.join(prompt_example + [prompt_target])
        else:
            # 當為零範例學習時，只使用目標問題作為提示
            n_valid_example = 0
            prompt = prompt_target
        # 5. 格式化響應文本，移除 "SELECT " 開頭
        response_clean = " ".join(target["query"].split())[len("SELECT "):]
        # 返回生成的提示及相關信息
        return {
            "prompt_tokens": sum_tokens, # 提示的總 token 數量
            "prompt": prompt, # 組合後的完整提示
            "response": response_clean, # 格式化後的響應（查詢語句）
            "n_examples": n_valid_example, # 使用的範例數量
            "db_id": target["db_id"] # 資料庫 ID
        }