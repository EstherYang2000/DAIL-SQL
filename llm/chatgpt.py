import json.decoder

import openai
from utils.enums import LLM
import time

# 初始化 OpenAI ChatGPT API，設定 API Key 和組織 ID
def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    # if model == LLM.TONG_YI_QIAN_WEN:
    #     import dashscope
    #     dashscope.api_key = OPENAI_API_KEY
    # else:
    #     openai.api_key = OPENAI_API_KEY
    #     openai.organization = OPENAI_GROUP_ID
    openai.api_key = OPENAI_API_KEY # 設置 OpenAI 的 API 金鑰
    openai.organization = OPENAI_GROUP_ID # 設置 OpenAI 的組織 ID

# 發送文本補全（completion）請求給 LLM，適用於非對話模式
def ask_completion(model, batch, temperature):
    response = openai.Completion.create(
        model=model, # 指定模型
        prompt=batch, # 請求的 prompt
        temperature=temperature, # 控制生成文本的隨機性，越高隨機性越強
        max_tokens=200, # 最大生成 token 數
        top_p=1, # 控制生成詞彙的多樣性
        frequency_penalty=0, # 控制生成文本中的重複程度
        presence_penalty=0, # 控制生成內容的新穎程度
        stop=[";"] # 停止生成的符號
    )
    # 提取每個生成結果中的文本內容
    response_clean = [_["text"] for _ in response["choices"]]
    return dict( # 返回包含生成結果和使用情況的字典
        response=response_clean,
        **response["usage"]
    )

# 發送對話請求（chat completion）給 LLM，適用於對話模式
def ask_chat(model, messages: list, temperature, n):
    response = openai.ChatCompletion.create(
        model=model, # 指定模型
        messages=messages, # 對話歷史，包含角色（user/system）和內容
        temperature=temperature, # 控制生成文本的隨機性
        max_tokens=200, # 最大生成 token 數
        n=n # 自洽模式的生成次數，決定要生成多少個不同的回答
    )
    # 提取每個生成的回應內容
    response_clean = [choice["message"]["content"] for choice in response["choices"]]
    if n == 1:
        response_clean = response_clean[0] # 如果只需要一個回應，直接取第一個回應
    return dict( # 返回包含生成結果和使用情況的字典
        response=response_clean,
        **response["usage"]
    )

# 向 LLM 發送請求，根據指定的模型、批次大小、隨機性等參數生成回應
def ask_llm(model: str, batch: list, temperature: float, n:int):
    n_repeat = 0 # 計數重試次數
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                # 非對話模式，僅支持單次生成
                assert n == 1
                print("batch:", batch)  # 輸出批次
                response = ask_completion(model, batch, temperature)
            elif model in LLM.TASK_CHAT:
                # batch size must be 1
                # 對話模式，批次大小必須為 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                print("batch:", batch)  # 輸出批次
                messages = [{"role": "user", "content": batch[0]}] # 將批次轉換為消息格式
                response = ask_chat(model, messages, temperature, n) 
                response['response'] = [response['response']] # 保證回應結果為列表格式
            break # 成功獲取回應後跳出循環
        except openai.error.RateLimitError: # 捕捉 RateLimitError，重試請求
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for RateLimitError", end="\n")
            time.sleep(1)
            continue
        except json.decoder.JSONDecodeError: # 捕捉 JSONDecodeError，重試請求
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
            time.sleep(1) # 等待 1 秒後重試
            continue
        except Exception as e: # 捕捉其他異常，重試請求
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1) # 等待 1 秒後重試
            continue

    return response # 返回最後獲取的回應

