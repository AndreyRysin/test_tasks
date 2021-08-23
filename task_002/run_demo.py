import pandas as pd
import numpy as np
from transformers import BertModel, BertConfig
import torch
import warnings
from src import *


def process(self):
    # ============================== TASK 1 ==============================
    warnings.filterwarnings("ignore")
    wall_total = walltime()
    wall_task1 = walltime()
    print("\n===================== TASK 1 STARTED =====================\n")
    print("Preparing... ", end="")
    # Paths and filenames
    parent_path = "./data"
    dataset_name = "test"
    tokenized_path = f"{parent_path}/tokenized"
    tokenized_prefix = "tokenized"
    attention_mask_path = f"{parent_path}/attention_mask"
    attention_mask_prefix = "att_mask"
    bad_pattern_path = f"{parent_path}/bad_pattern"
    bad_pattern_prefix = "bad_pat"
    embeddings_path = f"{parent_path}/embeddings"
    embeddings_prefix = "embed"
    united_arrays_path = f"{parent_path}/united_arrays"
    state_dicts_path = f"{parent_path}/state_dicts"
    stop_words_path = f"{parent_path}/russian"
    mystem_path_src = f"{parent_path}/pymystem"
    mystem_path_dst = f"{parent_path}"
    mystem_path = f"{parent_path}/mystem"
    bert_path_src = f"{parent_path}/bert"
    bert_path_dst = f"{parent_path}"
    bert_path = f"{parent_path}/rubert_cased_L-12_H-768_A-12_pt"
    bert_filename_config = "bert_config.json"
    bert_filename_model = "pytorch_model.bin"
    bert_filename_vocab = "vocab.txt"
    # Unpacking the archives
    unpack(mystem_path_src, mystem_path_dst)
    unpack(bert_path_src, bert_path_dst)
    # Init
    df_test = self.test()
    df_test.to_csv(f"{parent_path}/{dataset_name}.csv", index=False)
    df_test_index = df_test.index
    del df_test
    # Corp processing (CPU part)
    print("Ok!\nCorp processing (CPU part)... ", end="")
    reader = pd.read_csv(
        f"{parent_path}/{dataset_name}.csv",
        usecols=["description"],
        iterator=True,
        chunksize=1000,
    )
    for chunk, df in enumerate(reader):
        processing_pipeline_cpu(
            df,
            bert_path,
            bert_filename_vocab,
            tokenized_path,
            tokenized_prefix,
            attention_mask_path,
            attention_mask_prefix,
            bad_pattern_path,
            bad_pattern_prefix,
            stop_words_path,
            mystem_path,
            dataset_name,
            chunk,
        )
    # Corp processing (GPU part)
    print("Ok!\nCorp processing (GPU part)... ", end="")
    wall_gpu1 = walltime()
    bert_config = BertConfig.from_json_file(f"{bert_path}/{bert_filename_config}")
    bert_model = BertModel.from_pretrained(
        f"{bert_path}/{bert_filename_model}", config=bert_config
    ).cuda(torch.cuda.current_device())
    filenames_tokenized = listdir_without_ipynb_checkpoints(f"{parent_path}/tokenized")
    filenames_attention_mask = listdir_without_ipynb_checkpoints(
        f"{parent_path}/attention_mask"
    )
    assert len(filenames_tokenized) == len(filenames_attention_mask)
    for chunk in np.arange(len(filenames_tokenized)):
        arr_tokenized = load_array(
            tokenized_path,
            tokenized_prefix,
            dataset_name,
            chunk,
        )
        arr_attention_mask = load_array(
            attention_mask_path,
            attention_mask_prefix,
            dataset_name,
            chunk,
        )
        processing_pipeline_gpu(
            arr_tokenized,
            arr_attention_mask,
            bert_model,
            embeddings_path,
            embeddings_prefix,
            dataset_name,
            chunk,
        )
    del bert_model
    torch.cuda.empty_cache()
    print("Ok!")
    wall_gpu1.end(prefix="GPU1")
    # Uniting the chunks
    print("\nPreparing the data... ", end="")
    embeddings_test = chunks_uniting(
        embeddings_path,
        embeddings_prefix,
        dataset_name,
        united_path=united_arrays_path,
        dump=False,
    )
    bad_pattern_test = chunks_uniting(
        bad_pattern_path,
        bad_pattern_prefix,
        dataset_name,
        united_path=united_arrays_path,
        dump=False,
    )
    # Preparing the data to inference
    embeddings_test = np.hstack(
        (embeddings_test, bad_pattern_test.reshape(bad_pattern_test.shape[0], 1))
    )
    del bad_pattern_test
    embeddings_test = embeddings_test.reshape(
        embeddings_test.shape[0], 1, embeddings_test.shape[1]
    )
    # Inference
    print("Ok!\nInference (GPU)... ", end="")
    wall_gpu2 = walltime()
    predictions = inference_pipeline_gpu(embeddings_test, state_dicts_path, 1)
    print("Ok!")
    wall_gpu2.end(prefix="GPU2")
    # Forming the dataframe
    task1_prediction = pd.DataFrame(columns=["index", "prediction"])
    task1_prediction["index"] = df_test_index
    task1_prediction["prediction"] = predictions
    wall_task1.end(prefix="TASK1")
    print("\n===================== TASK 1 FINISHED ====================\n")
    # ============================== TASK 2 ==============================
    print("\n===================== TASK 2 STARTED =====================\n")
    print("Preparing... ", end="")
    wall_task2 = walltime()
    # Paths and filenames
    state_dicts_2_path = f"{parent_path}/state_dicts_2"
    # Loading the data
    df_test = pd.read_csv(
        f"{parent_path}/{dataset_name}.csv",
        usecols=["description"],
    )
    # Assessing
    print("Ok!\nAssessing... ", end="")
    (
        _target_test,
        bad_pattern_test,
        pat_multi_index_test,
        _pat_single_index_test,
        _pat_unknown_index_test,
        pat_clear_index_test,
    ) = processing_pipeline_cpu_2(
        df_test,
    )
    # Preparing the data
    print("Ok!\nPreparing the data... ", end="")
    embeddings_test[:, :, -1] = bad_pattern_test.reshape(bad_pattern_test.shape[0], 1)
    del bad_pattern_test
    # Inference
    print("Ok!\nInference (GPU)... ", end="")
    wall_gpu3 = walltime()

    predictions = inference_pipeline_gpu(
        embeddings_test, state_dicts_2_path, 2, pred_mean=False
    )
    print("Ok!")
    wall_gpu3.end(prefix="GPU3")
    # Postprocessing and forming the dataframe
    task2_prediction = postprocessing_pipeline_cpu_2(
        df_test,
        predictions,
        pat_multi_index_test,
        pat_clear_index_test,
    )
    wall_task2.end(prefix="TASK2")
    wall_total.end(prefix="TOTAL")
    print("\n===================== TASK 2 FINISHED ====================\n")
    # ============================== RETURN ==============================
    return task1_prediction, task2_prediction
