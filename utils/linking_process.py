import collections
import itertools
import json
import os

import attr
import numpy as np
import torch

from utils.linking_utils import abstract_preproc, corenlp, serialization
from utils.linking_utils.spider_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking,
    match_shift
)

@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,# 分詞函數，用於將名稱轉換成詞序列
                               include_table_name_in_column,# 是否在欄位名稱中包含表格名稱
                               fix_issue_16_primary_keys,# 是否修正主鍵的特殊問題
                               bert=False):# 若使用 BERT 模型，則會進行額外的標準化處理
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()# 創建一個空的 PreprocessedSchema 物件，用於儲存處理結果

    if bert: assert not include_table_name_in_column    # 如果使用 BERT，則不應該包含表格名稱於欄位名稱中

    last_table_id = None# 初始化變量以記錄最後處理的表格 ID
    for i, column in enumerate(schema.columns):# 遍歷每個欄位
        col_toks = tokenize_func( # 使用分詞函數將欄位名稱分詞
            column.name, column.unsplit_name)

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'# 定義欄位類型標記，用於標示該欄位的數據類型
        if bert:# 若使用 BERT，僅取第一個詞的表示並加入類型標記
            # for bert, we take the representation of the first word
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(Bertokens(col_toks))# 保存標準化的欄位名稱
        else: # 若非 BERT，則將類型標記放在前方
            column_name = [type_tok] + col_toks
        # 若要求包含表格名稱，則將表格名稱添加至欄位名稱後方
        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']# 若無表格，則使用 `<any-table>` 佔位符
            else:# 分詞表格名稱並附加至欄位名稱後
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name # 使用 `<table-sep>` 分隔符號
        r.column_names.append(column_name)# 將處理後的欄位名稱添加到欄位列表中
        # 記錄欄位所屬的表格 ID，若無表格則為 None
        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id# 建立欄位與表格之間的對應關係
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])# 若表格存在，將該欄位的索引添加至表格的欄位列表中
            columns.append(i)
        if last_table_id != table_id:# 若表格 ID 改變，則記錄此欄位的邊界
            r.table_bounds.append(i)
            last_table_id = table_id

        if column.foreign_key_for is not None: # 若欄位是外鍵，則記錄外鍵的對應關係
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)
    # 添加最後一個欄位的邊界，這樣表格數目會比邊界數目少 1
    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1
    # 處理表格名稱並儲存在 `table_names` 中
    for i, table in enumerate(schema.tables):
        # 使用分詞函數分詞表格名稱
        table_toks = tokenize_func(
            table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bert: 
            # 若使用 BERT，將標準化的表格名稱儲存在 `normalized_table_names` 中
            r.normalized_table_names.append(Bertokens(table_toks))
    last_table = schema.tables[-1]# 獲取最後一個表格
    # 將外鍵表格關係排序並轉換成字典格式
    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    # 設置主鍵，根據 `fix_issue_16_primary_keys` 參數選擇是否處理特殊情況
    r.primary_keys = [
        column.id
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r # 返回處理後的 PreprocessedSchema 物件


class SpiderEncoderV2Preproc(abstract_preproc.AbstractPreproc):
# 主要的處理類別，用於預處理資料庫架構和問題
    # 初始化方法，包含多個控制預處理行為的參數
    def __init__(
            self,
            save_path, # 預處理後資料的儲存路徑
            min_freq=3, # 詞頻最小值，詞彙頻率低於此值的詞可能會被忽略
            max_count=5000, # 詞彙的最大數量
            include_table_name_in_column=True, # 是否在欄位名稱中包含表格名稱
            word_emb=None, # 用於分詞的詞嵌入模型
            # count_tokens_in_word_emb_for_vocab=False,
            fix_issue_16_primary_keys=False, # 是否修正第16個問題（可能是特定架構的主鍵問題）
            compute_sc_link=False, # 是否計算 schema linking
            compute_cv_link=False): # 是否計算 cell value linking
        if word_emb is None:  # 設置詞嵌入，若未提供則為 None
            self.word_emb = None
        else:
            self.word_emb = word_emb
        # 定義存儲處理後數據的目錄
        self.data_dir = os.path.join(save_path, 'enc')
        self.include_table_name_in_column = include_table_name_in_column
        # self.count_tokens_in_word_emb_for_vocab = count_tokens_in_word_emb_for_vocab
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.texts = collections.defaultdict(list) # texts 是一個包含處理後問題數據的字典，鍵為 section 名稱
        # self.db_path = db_path

        # self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        # self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
        # self.vocab_word_freq_path = os.path.join(save_path, 'enc_word_freq.json')
        # self.vocab = None
        # self.counted_db_ids = set()
        self.preprocessed_schemas = {} # 用於儲存已處理過的資料庫架構，以便重複使用

    def validate_item(self, item, schema, section):# 驗證輸入的數據項
        return True, None # 總是返回 True 和 None，表示無檢查失敗

    def add_item(self, item, schema, section, validation_info): # 添加一個數據項到指定的 section，進行預處理後儲存
        preprocessed = self.preprocess_item(item, schema, validation_info)
        self.texts[section].append(preprocessed)# 將預處理結果添加到該 section

    def clear_items(self):# 清空所有數據項
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, schema, validation_info): # 預處理單一數據項，包括問題分詞、schema linking 和 cell value linking
        # 將問題和分詞版本進行處理
        question, question_for_copying = self._tokenize_for_copying(item['question_toks'], item['question'])
        # print(type(question))
        # print(type(question_for_copying))
        preproc_schema = self._preprocess_schema(schema) # 預處理 schema
        if self.compute_sc_link: # 計算 schema linking
            assert preproc_schema.column_names[0][0].startswith("<type:")
            column_names_without_types = [col[1:] for col in preproc_schema.column_names]
            sc_link = compute_schema_linking(question, column_names_without_types, preproc_schema.table_names)
        else:# 如果不計算，則設為空字典
            sc_link = {"q_col_match": {}, "q_tab_match": {}}

        if self.compute_cv_link:# 計算 cell value linking
            cv_link = compute_cell_value_linking(question, schema)
        else:
            cv_link = {"num_date_match": {}, "cell_match": {}} # 如果不計算，則設為空字典
        # 返回預處理後的數據項字典
        return {
            'raw_question': item['question'],# 原始問題
            'db_id': schema.db_id,# 資料庫 ID
            'question': question,# 分詞後的問題
            'question_for_copying': question_for_copying,# 用於複製的問題分詞
            'sc_link': sc_link,# schema linking 結果
            'cv_link': cv_link,# cell value linking 結果
            'columns': preproc_schema.column_names,# 欄位名稱
            'tables': preproc_schema.table_names,# 表格名稱
            'table_bounds': preproc_schema.table_bounds,# 表格邊界
            'column_to_table': preproc_schema.column_to_table, # 欄位到表格的對應
            'table_to_columns': preproc_schema.table_to_columns,# 表格到欄位的對應
            'foreign_keys': preproc_schema.foreign_keys,# 外鍵資訊
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,# 外鍵表格資訊
            'primary_keys': preproc_schema.primary_keys,# 主鍵資訊
        }

    def _preprocess_schema(self, schema):# 預處理資料庫架構，若已處理過則直接使用
        if schema.db_id in self.preprocessed_schemas:
            return self.preprocessed_schemas[schema.db_id]# 返回已處理的 schema
        # 調用 preprocess_schema_uncached 進行預處理並儲存結果
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column, self.fix_issue_16_primary_keys)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    def _tokenize(self, presplit, unsplit):# 分詞方法，根據詞嵌入模型進行分詞
        if self.word_emb:
            return self.word_emb.tokenize(unsplit)# 使用詞嵌入模型進行分詞
        return presplit# 否則返回原始分詞

    def _tokenize_for_copying(self, presplit, unsplit):# 分詞方法，用於準備可供複製的分詞版本
        if self.word_emb:
            return self.word_emb.tokenize_for_copying(unsplit)# 使用詞嵌入模型進行分詞
        return presplit, presplit# 否則返回原始分詞

    def save(self):  # 儲存處理後的數據到指定目錄
        os.makedirs(self.data_dir, exist_ok=True)# 若目錄不存在則創建
        # self.vocab = self.vocab_builder.finish()
        # print(f"{len(self.vocab)} words in vocab")
        # self.vocab.save(self.vocab_path)
        # self.vocab_builder.save(self.vocab_word_freq_path)

        for section, texts in self.texts.items():# 將每個 section 的數據保存為 JSONL 格式
            with open(os.path.join(self.data_dir, section + '_schema-linking.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text) + '\n')

    def load(self, sections):  # 載入指定 sections 的處理後數據
        # self.vocab = vocab.Vocab.load(self.vocab_path)
        # self.vocab_builder.load(self.vocab_word_freq_path)
        for section in sections:
            self.texts[section] = []
            with open(os.path.join(self.data_dir, section + '_schema-linking.jsonl'), 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        self.texts[section].append(json.loads(line))# 將每行的 JSON 資料讀取進來

    def dataset(self, section):# 載入指定 section 的資料集
        # 返回指定 section 的 JSONL 格式數據列表
        return [
            json.loads(line)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

