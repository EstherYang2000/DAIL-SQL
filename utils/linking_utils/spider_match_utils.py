import re
import string
import collections

import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)

CELL_EXACT_MATCH_FLAG = "EXACTMATCH"
CELL_PARTIAL_MATCH_FLAG = "PARTIALMATCH"
COL_PARTIAL_MATCH_FLAG = "CPM"
COL_EXACT_MATCH_FLAG = "CEM"
TAB_PARTIAL_MATCH_FLAG = "TPM"
TAB_EXACT_MATCH_FLAG = "TEM"

# schema linking, similar to IRNet
def compute_schema_linking(question, column, table):
    def partial_match(x_list, y_list):# 定義部分匹配函數，檢查 x_list 是否部分匹配 y_list
        x_str = " ".join(x_list)# 將 x_list 合併成單一字串
        y_str = " ".join(y_list)# 將 y_list 合併成單一字串
        if x_str in STOPWORDS or x_str in PUNKS:# 檢查是否為停用詞或標點符號
            return False
        # 使用正則表達式進行匹配，檢查 x_str 是否為 y_str 的子串
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str # 確保 x_str 是 y_str 的一部分
            return True
        else:
            return False
    # 定義完全匹配函數，檢查 x_list 是否與 y_list 完全相等
    def exact_match(x_list, y_list):
        x_str = " ".join(x_list) # 將 x_list 合併成單一字串
        y_str = " ".join(y_list) # 將 y_list 合併成單一字串
        if x_str == y_str: # 若兩者相等，則返回 True
            return True
        else:
            return False
    # 用於儲存問題與欄位的匹配結果
    q_col_match = dict()
    # 用於儲存問題與表格的匹配結果
    q_tab_match = dict()
    # 建立欄位 ID 與欄位名稱之間的對應
    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0: # 略過索引為 0 的欄位，通常是預留位
            continue
        col_id2list[col_id] = col_item # 儲存欄位 ID 與欄位名稱的對應
    # 建立表格 ID 與表格名稱之間的對應
    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item # 儲存表格 ID 與表格名稱的對應

    # 5-gram # 使用 5-gram 來逐步縮小匹配範圍
    n = 5
    while n > 0:
        # 對問題中的詞語按 n-gram 進行處理
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n] # 提取 n-gram
            n_gram = " ".join(n_gram_list) # 合併成字串
            if len(n_gram.strip()) == 0: # 若 n_gram 為空，則略過
                continue
            # exact match case # 完全匹配情況
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]): # 若與欄位名稱完全匹配
                    for q_id in range(i, i + n): # 標記問題中對應位置的匹配標誌
                        q_col_match[f"{q_id},{col_id}"] = COL_EXACT_MATCH_FLAG
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]): # 若與表格名稱完全匹配
                    for q_id in range(i, i + n): # 標記問題中對應位置的匹配標誌
                        q_tab_match[f"{q_id},{tab_id}"] = TAB_EXACT_MATCH_FLAG

            # partial match case # 部分匹配情況
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]): # 若與欄位名稱部分匹配
                    for q_id in range(i, i + n):
                        # 只有在尚未完全匹配時才標記為部分匹配
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = COL_PARTIAL_MATCH_FLAG
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]): # 若與表格名稱部分匹配
                    for q_id in range(i, i + n):
                        # 只有在尚未完全匹配時才標記為部分匹配
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = TAB_PARTIAL_MATCH_FLAG
        n -= 1 # 減少 n-gram 的大小以進行更細粒度的匹配
    # 返回匹配結果，包括問題與欄位的匹配和問題與表格的匹配
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_partial_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    def db_word_exact_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word}' or {column} like ' {word}' or " \
                f"{column} like '{word} ' or {column} like ' {word} '"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}

    for col_id, column in enumerate(schema.columns):
        if col_id == 0:
            assert column.orig_name == "*"
            continue
        match_q_ids = []
        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0:
                continue
            if word in STOPWORDS or word in PUNKS:
                continue

            num_flag = isnumber(word)
            if num_flag:    # TODO refine the date and time match
                if column.type in ["number", "time"]:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                ret = db_word_partial_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    match_q_ids.append(q_id)
        f = 0
        while f < len(match_q_ids):
            t = f + 1
            while t < len(match_q_ids) and match_q_ids[t] == match_q_ids[t - 1] + 1:
                t += 1
            q_f, q_t = match_q_ids[f], match_q_ids[t - 1] + 1
            words = [token for token in tokens[q_f: q_t]]
            ret = db_word_exact_match(' '.join(words), column.orig_name, column.table.orig_name, schema.connection)
            if ret:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_EXACT_MATCH_FLAG
            else:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_PARTIAL_MATCH_FLAG
            f = t

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link


def match_shift(q_col_match, q_tab_match, cell_match):

    q_id_to_match = collections.defaultdict(list)
    for match_key in q_col_match.keys():
        q_id = int(match_key.split(',')[0])
        c_id = int(match_key.split(',')[1])
        type = q_col_match[match_key]
        q_id_to_match[q_id].append((type, c_id))
    for match_key in q_tab_match.keys():
        q_id = int(match_key.split(',')[0])
        t_id = int(match_key.split(',')[1])
        type = q_tab_match[match_key]
        q_id_to_match[q_id].append((type, t_id))
    relevant_q_ids = list(q_id_to_match.keys())

    priority = []
    for q_id in q_id_to_match.keys():
        q_id_to_match[q_id] = list(set(q_id_to_match[q_id]))
        priority.append((len(q_id_to_match[q_id]), q_id))
    priority.sort()
    matches = []
    new_q_col_match, new_q_tab_match = dict(), dict()
    for _, q_id in priority:
        if not list(set(matches) & set(q_id_to_match[q_id])):
            exact_matches = []
            for match in q_id_to_match[q_id]:
                if match[0] in [COL_EXACT_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                    exact_matches.append(match)
            if exact_matches:
                res = exact_matches
            else:
                res = q_id_to_match[q_id]
            matches.extend(res)
        else:
            res = list(set(matches) & set(q_id_to_match[q_id]))
        for match in res:
            type, c_t_id = match
            if type in [COL_PARTIAL_MATCH_FLAG, COL_EXACT_MATCH_FLAG]:
                new_q_col_match[f'{q_id},{c_t_id}'] = type
            if type in [TAB_PARTIAL_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                new_q_tab_match[f'{q_id},{c_t_id}'] = type

    new_cell_match = dict()
    for match_key in cell_match.keys():
        q_id = int(match_key.split(',')[0])
        if q_id in relevant_q_ids:
            continue
        # if cell_match[match_key] == CELL_EXACT_MATCH_FLAG:
        new_cell_match[match_key] = cell_match[match_key]

    return new_q_col_match, new_q_tab_match, new_cell_match