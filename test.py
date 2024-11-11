import argparse
import json
import os
import pickle
from pathlib import Path
import sqlite3
from tqdm import tqdm
import networkx as nx

from utils.linking_process import SpiderEncoderV2Preproc
from utils.pretrained_embeddings import GloVe
from utils.datasets.spider import load_tables
import attr

@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    connection = attr.ib(default=None)     

dataset_dir = "./dataset/spider"
table = 'tables.json' 

# def load_tables(paths):
schemas = {}
eval_foreign_key_maps = {}
paths = [os.path.join(dataset_dir, table)]
print(paths)
for path in paths:
    schema_dicts = json.load(open(path))
    for schema_dict in schema_dicts:
        # print(schema_dict)
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )
        # print(columns)

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            if isinstance(column_id, list):
                for each_id in column_id:
                    column = columns[each_id]
                    column.table.primary_keys.append(column)
            else:
                column = columns[column_id]
                column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        # print(foreign_key_graph)
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        assert db_id not in schemas
        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
        # print(schemas)
        # eval_foreign_key_maps[db_id] = build_foreign_key_map(schema_dict)

# return schemas, eval_foreign_key_maps
schemas, _ = load_tables([os.path.join(dataset_dir, table)])

# print(type(schemas))
# print(schemas.keys())




# 4. Initialize word embeddings using the GloVe pre-trained embeddings (42B version)
compute_cv_link = True
test = "dev.json"
train = 'train_spider_and_others.json'
# 1. Load test and train data from JSON files located in the dataset directory
test_data = json.load(open(os.path.join(dataset_dir, test)))
train_data = json.load(open(os.path.join(dataset_dir, train)))
word_emb = GloVe(kind='42B', lemmatize=True)
linking_processor = SpiderEncoderV2Preproc(dataset_dir,
        min_freq=4,
        max_count=5000,
        include_table_name_in_column=False,
        word_emb=word_emb,
        fix_issue_16_primary_keys=True,
        compute_sc_link=True,
        compute_cv_link=compute_cv_link)

# build schema-linking
for data, section in zip([test_data, train_data],['test', 'train']):
    for item in tqdm(data, desc=f"{section} section linking"):
        db_id = item["db_id"]
        schema = schemas[db_id]
        to_add, validation_info = linking_processor.validate_item(item, schema, section)
        if to_add:
            linking_processor.add_item(item, schema, section, validation_info)
        break