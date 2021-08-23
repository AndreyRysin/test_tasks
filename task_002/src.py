import pandas as pd
import numpy as np
from pymystem3 import Mystem
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import pickle
import os
import re


def save_array(arr, path, prefix, dataset_name, chunk):
    """
    Saves the (chunked) array on a disk.
    
    Filename template:
    "{path}/{prefix}_chunk_{dataset_name}_{chunk}.pkl"
    """
    # Checking if the path exists and creating it if not
    if not os.path.exists(path):
        os.mkdir(path)

    # Forming the filename template
    filename = "{}/{}_chunk_{}_{:05d}.pkl".format(path, prefix, dataset_name, chunk)

    # Saving the data on a disk
    with open(filename, "wb") as f:
        pickle.dump(arr, f)


def load_array(path, prefix, dataset_name, chunk):
    """
    Loads the (chunked) array from a disk.
    
    Filename template:
    "{path}/{prefix}_chunk_{dataset_name}_{chunk}.pkl"
    """
    # Checking if the path exists
    assert os.path.exists(path)

    # Forming the filename template
    filename = "{}/{}_chunk_{}_{:05d}.pkl".format(path, prefix, dataset_name, chunk)

    # Loading the data from a disk
    with open(filename, "rb") as f:
        return pickle.load(f)


def pattern_dict(task_number):
    """
    Returns the list of regular expression patterns.
    """
    patterns = {
        1: [
            "\+7\s{,1}9",
            "http",
            "тел\.\d",
            "\+7\D{,2}9\d{2}\D{,2}\d{3}\D{,2}\d{2}\D{,2}\d{2}",
            "\+7\d{10}",
            "89\d{9}",
            "\+79\d{9}",
            "vk.com",
        ],
        2: [
            "\+7\s{,1}9",
            "http",
            "тел\.\d",
            "\+7\D{,2}9\d{2}\D{,2}\d{3}\D{,2}\d{2}\D{,2}\d{2}",
            "\+{,1}[78]{,1}\D\d\D\d\D\d\D\d\D\d\D\d\D\d\D\d\D\d\D\d\D",
            "\+7\d{10}",
            "89\d{9}",
            "\+79\d{9}",
            "vk.com\.+[\s\b]",
            "\+7 9\d{2}",
            "https://vk.com",
            "[78]-*9\d{2}-*\d{3}-*\d{2}-*\d{2}",
            "[a-zA-Z0-9]{4,}.ru",
            "@[a-zA-Z0-9]+.com",
            "@[a-zA-Z0-9]+.ru",
            "instagram\s*@[a-zA-Z0-9]",
            "\d{1,3}[\s-]{,3}\(\d{3,4}\)[\s-]{,3}\d",
            "[a-zA-Z0-9]{4,}.com",
            "[78] 9\d{2} \d{3} \d{2} \d{2}",
            "[78][\s\n]{,2}9\d{2}[\s\n]{,2}\d{3}[\s\n]{,2}\d{2}[\s\n]{,2}\d{2}",
        ],
    }
    return patterns[task_number]


def processing_pipeline_cpu(
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
):
    """
    Implementation of the processing pipeline.
    Is computed on CPU.

    In:
    df

    Out:
    array "bad_pattern"
    array "tokenized"
    array "attention_mask"
    """
    # Bad patterns
    condition = False
    for pat in pattern_dict(1):
        condition = condition | df["description"].str.contains(pat, regex=True)

    bad_pattern_is_true_index = df[condition].index

    df.loc[:, "bad_pattern"] = 0
    df.loc[bad_pattern_is_true_index, "bad_pattern"] = 1
    df["bad_pattern"] = df["bad_pattern"].astype("uint8")

    # Saving the data
    save_array(
        df["bad_pattern"].values,
        bad_pattern_path,
        bad_pattern_prefix,
        dataset_name,
        chunk,
    )

    # Loading the stop words list
    stop_words = []
    with open(stop_words_path, "r") as f:
        for line in f:
            stop_words.append(line.replace("\n", ""))
    stop_words.sort()

    # Removing unnecessary symbols
    df.loc[:, "corp"] = (
        df["description"]
        .str.replace("[^\d\s\w]", " ", regex=True)
        .str.replace("_", " ", regex=True)
        .str.lower()
    )

    # Removing stop words
    for sw in stop_words:
        # from the middle, with both-side spaces
        df["corp"] = df["corp"].str.replace(f" {sw} ", " ")

        # from the left side, with right-side space
        lstrip_index = df[df["corp"].str.startswith(f"{sw} ")].index
        df.loc[lstrip_index, "corp"] = df.loc[lstrip_index, "corp"].str.lstrip(f"{sw} ")

        # from the right side, with left-side space
        rstrip_index = df[df["corp"].str.endswith(f" {sw}")].index
        df.loc[rstrip_index, "corp"] = df.loc[rstrip_index, "corp"].str.rstrip(f" {sw}")

    # Removing unnecessary spaces
    df["corp"] = df["corp"].str.replace("\s+", " ", regex=True)

    # Lemmatization
    m = Mystem(mystem_bin=mystem_path)
    df["corp"] = df["corp"].apply(lambda x: "".join(m.lemmatize(x)))

    # Forming the tokens
    tokenizer = BertTokenizer(vocab_file=f"{bert_path}/{bert_filename_vocab}")
    df.loc[:, "tokenized"] = df["corp"].apply(
        lambda x: np.array(tokenizer.encode(x, add_special_tokens=True))
    )

    # Removing exceeding tokens from the middle
    TOKEN_AMOUNT_MAX = 512  # TODO: get this stuff from bert_config.json
    df["tokenized"] = df["tokenized"].apply(
        lambda x: np.concatenate(
            (x[: TOKEN_AMOUNT_MAX // 2], x[-TOKEN_AMOUNT_MAX // 2 :])
        )
    )
    tokens_max = df["tokenized"].apply(lambda x: x.shape[0]).max()

    # Padding with zeros
    df["tokenized"] = df["tokenized"].apply(
        lambda x: np.append(x, [0] * (tokens_max - x.shape[0]))
    )
    arr_tokenized = np.empty(
        (df["tokenized"].shape[0], TOKEN_AMOUNT_MAX), dtype="int64"
    )
    for i in np.arange(df["tokenized"].shape[0]):
        arr_tokenized[i] = df.reset_index(drop=True).loc[i, "tokenized"]

    # Attention mask
    arr_attention_mask = np.where(arr_tokenized != 0, 1, 0)

    # Saving the data
    save_array(
        arr_tokenized, tokenized_path, tokenized_prefix, dataset_name, chunk,
    )

    save_array(
        arr_attention_mask,
        attention_mask_path,
        attention_mask_prefix,
        dataset_name,
        chunk,
    )


def processing_pipeline_gpu(
    arr_tokenized,
    arr_attention_mask,
    bert_model,
    embeddings_path,
    embeddings_prefix,
    dataset_name,
    chunk,
    batch_size=5,
):
    """
    Implementation of the processing pipeline.
    Is computed on GPU.

    In:
    array "tokenized"
    array "attention_mask"

    Out:
    array "embeddings"
    """
    EMBED_AMOUNT = 768  # TODO: get this stuff from bert_config.json
    embeddings = np.array([])

    for batch in np.arange(arr_tokenized.shape[0] // batch_size + 1):
        # Forming tensors from the arrays
        tensor_tokens_batch = torch.cuda.LongTensor(
            arr_tokenized[batch * batch_size : (batch + 1) * batch_size]
        )
        tensor_mask_batch = torch.cuda.LongTensor(
            arr_attention_mask[batch * batch_size : (batch + 1) * batch_size]
        )
        # Computing the embeddings
        with torch.no_grad():
            embeddings_batch = bert_model(
                input_ids=tensor_tokens_batch, attention_mask=tensor_mask_batch,
            )
        embeddings = np.append(embeddings, embeddings_batch[0][:, 0, :].cpu().numpy())

    # Reshaping from 1-dim to 2-dim array
    embeddings = embeddings.reshape(embeddings.shape[0] // EMBED_AMOUNT, EMBED_AMOUNT)

    # Saving the data
    save_array(
        embeddings, embeddings_path, embeddings_prefix, dataset_name, chunk,
    )


def chunks_uniting(
    chunks_path, prefix, dataset_name, united_path=None, dtype="float32", dump=True,
):
    """
    Unites chunked arrays.

    dump == True: saves the resulting array on a disk
    dump == False: returns the resulting array
    """
    filenames = (
        pd.Series(listdir_without_ipynb_checkpoints(chunks_path))
        .sort_values()
        .reset_index(drop=True)
    )

    chunks_list = []
    for chunk in np.arange(filenames[filenames.str.contains(dataset_name)].shape[0]):
        chunks_list.append(
            load_array(chunks_path, prefix, dataset_name, chunk,).astype(dtype)
        )
    arr_united = np.concatenate(chunks_list, axis=0)

    if dump:
        assert united_path != None
        with open(f"{united_path}/{prefix}_{dataset_name}.pkl", "wb") as f:
            pickle.dump(arr_united, f)
    else:
        return arr_united


def listdir_without_ipynb_checkpoints(path):
    """
    Implements "os.listdir()" method with two features:
    - ".ipynb_checkpoints" item is removed;
    - the list is sorted.
    
    path : string
        Full path to the files whose list is required.
    
    return : list of strings
        List of the file names.
    """
    filenames = sorted(os.listdir(path))
    i = 0
    while i < len(filenames):
        if filenames[i] == ".ipynb_checkpoints":
            del filenames[i]
        else:
            i += 1
    return filenames


class Net(nn.Module):
    """
    Neural network structure.
    """

    def __init__(
        self, out_units_final, net_params,
    ):
        """
        Layers initialization.
        """
        super(Net, self,).__init__()

        self.sequential_conv00 = nn.Sequential(
            # submodule 00
            nn.Conv1d(
                1,
                net_params["sm00_Conv1d"]["out_channels"],
                net_params["sm00_Conv1d"]["kernel_size"],
                padding=net_params["sm00_Conv1d"]["padding"],
                stride=net_params["sm00_Conv1d"]["stride"],
                bias=False,
            ),
            nn.BatchNorm1d(net_params["sm00_Conv1d"]["out_channels"],),
            nn.ReLU(),
            nn.MaxPool1d(
                net_params["sm00_Pool1d"]["kernel_size"],
                padding=net_params["sm00_Pool1d"]["padding"],
                stride=net_params["sm00_Pool1d"]["stride"],
            ),
        )

        self.sequential_conv01 = nn.Sequential(
            # submodule 01
            nn.Conv1d(
                1,
                net_params["sm01_Conv1d"]["out_channels"],
                net_params["sm01_Conv1d"]["kernel_size"],
                padding=net_params["sm01_Conv1d"]["padding"],
                stride=net_params["sm01_Conv1d"]["stride"],
                bias=False,
            ),
            nn.BatchNorm1d(net_params["sm01_Conv1d"]["out_channels"],),
            nn.ReLU(),
            nn.MaxPool1d(
                net_params["sm01_Pool1d"]["kernel_size"],
                padding=net_params["sm01_Pool1d"]["padding"],
                stride=net_params["sm01_Pool1d"]["stride"],
            ),
        )

        self.sequential_conv02 = nn.Sequential(
            # submodule 02
            nn.Conv1d(
                net_params["sm00_Conv1d"]["out_channels"]
                + net_params["sm01_Conv1d"]["out_channels"],
                net_params["sm02_Conv1d"]["out_channels"],
                net_params["sm02_Conv1d"]["kernel_size"],
                padding=net_params["sm02_Conv1d"]["padding"],
                stride=net_params["sm02_Conv1d"]["stride"],
                bias=False,
            ),
            nn.BatchNorm1d(net_params["sm02_Conv1d"]["out_channels"],),
            nn.ReLU(),
            nn.AvgPool1d(
                net_params["sm02_Pool1d"]["kernel_size"],
                padding=net_params["sm02_Pool1d"]["padding"],
                stride=net_params["sm02_Pool1d"]["stride"],
            ),
            # submodule 03
            nn.Conv1d(
                net_params["sm02_Conv1d"]["out_channels"],
                net_params["sm03_Conv1d"]["out_channels"],
                net_params["sm03_Conv1d"]["kernel_size"],
                padding=net_params["sm03_Conv1d"]["padding"],
                stride=net_params["sm03_Conv1d"]["stride"],
                bias=False,
            ),
            nn.BatchNorm1d(net_params["sm03_Conv1d"]["out_channels"],),
            nn.ReLU(),
            nn.AvgPool1d(
                net_params["sm03_Pool1d"]["kernel_size"],
                padding=net_params["sm03_Pool1d"]["padding"],
                stride=net_params["sm03_Pool1d"]["stride"],
            ),
            # flatten
            nn.Flatten(),
        )

        self.sequential_lin = nn.Sequential(
            # submodule 10
            nn.Dropout(net_params["sm10_Dropout1d"]["p"],),
            nn.Linear(
                net_params["sm10_Linear"]["in_units"],
                net_params["sm10_Linear"]["out_units"],
            ),
            nn.ReLU(),
            # final
            nn.Linear(net_params["sm10_Linear"]["out_units"], out_units_final,),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward run.
        """
        x_00 = x
        for i in np.arange(len(self.sequential_conv00)):
            x_00 = self.sequential_conv00[i](x_00)

        x_01 = x
        for i in np.arange(len(self.sequential_conv01)):
            x_01 = self.sequential_conv01[i](x_01)

        x = torch.cat((x_00, x_01), dim=1)
        for i in np.arange(len(self.sequential_conv02)):
            x = self.sequential_conv02[i](x)

        for i in np.arange(len(self.sequential_lin)):
            x = self.sequential_lin[i](x)

        return x


class Inference:
    """
    Implementation of the neural network inference pass.
    The reduced version of the class "Trainer" (inference only).
    """

    def __init__(
        self, model, cuda_number=torch.cuda.current_device(),
    ):
        self.cuda_number = cuda_number
        self.model = model.float().cuda(self.cuda_number)

    def load_state_dicts(
        self, filename,
    ):
        """
        Loads and assigns state dict.
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(
        self, X_te, batch_size,
    ):
        """
        Implementation of prediction with the trained model.
        """
        dataset_te = TensorDataset(torch.FloatTensor(X_te))
        loader_te = DataLoader(dataset_te, batch_size=batch_size,)
        y_pred = torch.cuda.FloatTensor([])

        for batch in loader_te:
            X_batch = batch[0].cuda(self.cuda_number)
            y_pred_batch = self.model(X_batch).detach()
            y_pred = torch.cat((y_pred, y_pred_batch))

        return y_pred.detach().cpu().numpy()


def net_params_dict():
    """
    Returns the dict with the neural network parameters.
    """
    net_params = {
        "sm00_Conv1d": {
            "out_channels": 16,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        },
        "sm00_Pool1d": {"kernel_size": 2, "padding": 0, "stride": 2,},
        "sm01_Conv1d": {
            "out_channels": 16,
            "kernel_size": 5,
            "padding": 2,
            "stride": 1,
        },
        "sm01_Pool1d": {"kernel_size": 2, "padding": 0, "stride": 2,},
        "sm02_Conv1d": {
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        },
        "sm02_Pool1d": {"kernel_size": 2, "padding": 0, "stride": 2,},
        "sm03_Conv1d": {
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "stride": 2,
        },
        "sm03_Pool1d": {"kernel_size": 2, "padding": 0, "stride": 2,},
        "sm10_Dropout1d": {"p": 0.4,},
        "sm10_Linear": {"in_units": 3072, "out_units": 32,},
    }
    return net_params


def inference_pipeline_gpu(
    arr_embeddings, state_dicts_path, units_out, batch_size=500, pred_mean=True,
):
    """
    Implementation of the inference pipeline.
    Is computed on GPU.

    In:
    array "embeddings"

    Out:
    array "predict_proba"
    """
    # Initialization
    net_params = net_params_dict()
    model = Net(units_out, net_params,).cuda(torch.cuda.current_device())
    inference = Inference(model)
    state_dicts_filenames = listdir_without_ipynb_checkpoints(state_dicts_path)
    preds = np.array([])

    # Inference
    for filename in state_dicts_filenames:
        inference.load_state_dicts(f"{state_dicts_path}/{filename}")
        pred = inference.predict(X_te=arr_embeddings, batch_size=batch_size,)
        try:
            preds = np.hstack((preds, pred))
        except:
            preds = pred.copy()

    del model, inference
    torch.cuda.empty_cache()

    # Return
    if pred_mean:
        return preds.mean(axis=1)
    else:
        return preds


class walltime:
    """
    Measures the wall time without any extra consumption of the system resources.

    Functions:
    ----------    
    __init__ :
        The initial function.
    
    end :
        Takes the current time and calculates the wall time as the time interval
        since initializing.

    Attributes:
    ----------
    wall_total_seconds : int
        Contains the total amount of seconds of the fixed time interval.
    """

    def __init__(
        self, single_used=True,
    ):
        """
        The initial function.
        
        single_used : bool, default = True
            If True, the time in the function "end" is fixed only once,
            and then it becomes frozen; if False, the time is fixed whenever
            the "end" is called.
        """
        self.single_used = single_used
        self.begin = datetime.now()
        self.received = False

    def end(
        self, silent=False, prefix="",
    ):
        """
        Takes the current time and calculates the wall time as the time interval
        since initializing.
        
        silent : bool, default = False
            If False, prints the output string; if True, doesn't.
        
        prefix : string, default = ""
            The prefix is put at the beginning of the output string for
            personalizing a particular timer. Space between the prefix and
            the remaining string is added automatically.
        """

        prefix = prefix
        if len(prefix) > 0:
            prefix += " " if prefix[-1] != " " else ""

        if self.received is False:
            self.wall_total_seconds = int((datetime.now() - self.begin).total_seconds())
            self.received = True if self.single_used is True else False

        if self.wall_total_seconds < 3600:
            self.wall = "{}wall: {:2.0f}m {:02.0f}s".format(
                prefix, self.wall_total_seconds // 60, self.wall_total_seconds % 60,
            )
        else:
            self.wall = "{}wall: {:3.0f}h {:2.0f}m {:02.0f}s".format(
                prefix,
                self.wall_total_seconds // 3600,
                (self.wall_total_seconds % 3600) // 60,
                self.wall_total_seconds % 60,
            )
        if silent is not True:
            print(f"\n{self.wall}")


def unpack(
    src, dst,
):
    """
    Unpacks tar.gz archive.

    src : string
        Folder with the archive parts.
    
    dst : string
        Folder of the destination.
    """
    os.system(f"cat {src}/* | tar -xz -C {dst}")


def pattern_positions(string, patterns):
    """
    Computes start and end the positions of each substring that contains any
    of the given patterns (or several of them at the time).
    
    The output array structure:
    
        [[start, end],
         [start, end],
         .....
         [start, end]]
         
    start, end - the indices where each substring starts and ends
    
    string : string
        The string the patterns are searched in.
    
    patterns : list of strings
        The list of patterns that could be as strings as regular expressions.
    
    return : numpy.ndarray
        The two-dimensional array of the structure as above.
    """
    # substrings
    substrings = []
    for pat in patterns:
        substrings.extend(re.findall(re.compile(pat), string))

    if len(substrings) > 0:
        # positions
        substrings = pd.Series(substrings).drop_duplicates().values
        positions = np.array([], dtype="int")
        for substr in substrings:
            for i in np.arange(len(string) - len(substr) + 1):
                if string[i] == substr[0]:
                    if string[i : i + len(substr)] == substr:
                        positions = np.append(positions, (i, i + len(substr)))
        positions = positions.reshape(-1, 2)

        # char_indices
        char_indices = np.array([], dtype="int")
        for i in np.arange(positions.shape[0]):
            char_indices = np.append(
                char_indices, np.arange(positions[i, 0], positions[i, 1] + 1)
            )
        char_indices = np.sort(np.unique(char_indices))

        # positions_unique
        positions_unique = np.array([], dtype="int")
        pos_start = 0
        for i in np.arange(char_indices.shape[0] - 1):
            if char_indices[i + 1] - char_indices[i] > 1:
                positions_unique = np.append(
                    positions_unique, (char_indices[pos_start], char_indices[i])
                )
                pos_start = i + 1
        positions_unique = np.append(
            positions_unique, (char_indices[pos_start], char_indices[-1])
        )
        positions_unique = positions_unique.reshape(-1, 2)

        # return
        return positions_unique
    else:
        return (0, 0)


def processing_pipeline_cpu_2(df,):
    """
    Implementation of the processing pipeline for the task 2.
    Is computed on CPU.

    In:
    df

    Out:
    array "start_finish"
    array "pat_pos"
    array "bad_pattern_upd"
    array "pat_multi_index"
    array "pat_single_index"
    array "pat_unknown_index"
    array "pat_clear_index"
    """

    # Bad pattern
    condition = False
    for pat in pattern_dict(2):
        condition = condition | df["description"].str.contains(pat, regex=True)
    bad_pattern_is_true_index = df[condition].index
    df.loc[:, "bad_pattern"] = 0
    df.loc[bad_pattern_is_true_index, "bad_pattern"] = 1
    df["bad_pattern"] = df["bad_pattern"].astype("uint8")

    # Pattern positions
    df.loc[:, "pat_pos"] = df["description"][df["bad_pattern"] == 1].apply(
        pattern_positions, patterns=pattern_dict(2)
    )
    df["pat_pos"][df["bad_pattern"] == 0] = df["description"][
        df["bad_pattern"] == 0
    ].apply(lambda x: np.array([[0, 0]]))

    # Index arrays
    pat_multi_index = (
        df["pat_pos"][df["bad_pattern"] == 1][
            df["pat_pos"][df["bad_pattern"] == 1].apply(lambda x: x.shape[0] > 1)
        ]
    ).index

    pat_single_index = (
        df["pat_pos"][df["bad_pattern"] == 1][
            df["pat_pos"][df["bad_pattern"] == 1].apply(lambda x: x.shape[0] == 1)
        ]
    ).index

    pat_unknown_index = (df["pat_pos"][df["bad_pattern"] == 0]).index

    pat_clear_index = df.index[
        (df.index.isin(pat_multi_index) != True)
        & (df.index.isin(pat_single_index) != True)
        & (df.index.isin(pat_unknown_index) != True)
    ]

    # Relative pattern positions
    df.loc[:, "start"] = (
        df.loc[pat_single_index, "pat_pos"].apply(lambda x: x[0, 0])
        / df["description"].str.len()
    )
    df.loc[:, "finish"] = (
        df.loc[pat_single_index, "pat_pos"].apply(lambda x: x[0, 1])
        / df["description"].str.len()
    )
    df[["start", "finish"]] = df[["start", "finish"]].fillna(0)

    # Return
    return (
        df[["start", "finish"]].values,
        df["bad_pattern"].values,
        pat_multi_index,
        pat_single_index,
        pat_unknown_index,
        pat_clear_index,
    )


def postprocessing_pipeline_cpu_2(
    df, pred, pat_multi_index, pat_clear_index,
):
    """
    Postprocessing pipeline that transforms the array of predictions
    into the dataframe.

    In:
    dataframe "df"
    array "predictions"
    array "pat_multi_index"
    array "pat_clear_index"

    Out:
    dataframe "df_prediction"
    """
    # Convert relative coordinates to absolute
    pred_abs = np.zeros(pred.shape, dtype="int")
    for i in np.arange(pred_abs.shape[1]):
        pred_abs[:, i] = pred[:, i] * df["description"].str.len().values

    # Turning to zero if there are several patterns of no patterns
    pred_abs[pat_multi_index] = 0
    pred_abs[pat_clear_index] = 0

    # Forming the dataframe
    df_prediction = pd.DataFrame(columns=["index", "start", "finish"])
    df_prediction["index"] = df.index
    df_prediction[["start", "finish"]] = pred_abs

    # If "start < finish" then NaN
    df_prediction.loc[
        df_prediction[df_prediction["start"] >= df_prediction["finish"]].index,
        ["start", "finish"],
    ] = np.nan

    # If "start >= 0 and finish == 0" then NaN
    df_prediction.loc[
        df_prediction[
            (df_prediction["start"] >= 0) & (df_prediction["finish"] == 0)
        ].index,
        ["start", "finish"],
    ] = np.nan

    # Turning to NaN if there are several patterns of no patterns
    # (probably, this condition is excess: this transformation has been done above)
    df_prediction.loc[pat_multi_index, ["start", "finish"]] = np.nan
    df_prediction.loc[pat_clear_index, ["start", "finish"]] = np.nan

    # return
    return df_prediction


class torch_iou:
    """
    Metric and loss function based on the intersection-of-union (IoU) principle.
    """

    def __init__(
        self, y_pred, y_true,
    ):
        """
        Initialization and pre-computation.
        
        Column order of the input tensors:
        xmin ymin xmax ymax
        """
        # Init
        assert (y_pred.shape[1] == 4) | (y_pred.shape[1] == 2)
        assert y_pred.shape[1] == y_true.shape[1]

        y_pred = y_pred if y_pred.shape[1] == 4 else self.reshape_2_to_4(y_pred)
        y_true = y_true if y_true.shape[1] == 4 else self.reshape_2_to_4(y_true)

        self.xmin_pred = y_pred[:, 0]
        self.ymin_pred = y_pred[:, 1]
        self.xmax_pred = y_pred[:, 2]
        self.ymax_pred = y_pred[:, 3]
        self.xmin_true = y_true[:, 0]
        self.ymin_true = y_true[:, 1]
        self.xmax_true = y_true[:, 2]
        self.ymax_true = y_true[:, 3]

        self.w_true = self.xmax_true - self.xmin_true
        self.h_true = self.ymax_true - self.ymin_true
        self.w_pred = self.xmax_pred - self.xmin_pred
        self.h_pred = self.ymax_pred - self.ymin_pred

        a_true = self.w_true * self.h_true
        a_pred = self.w_pred * self.h_pred

        self.eps = 1e-10

        # Area of overlap (aoo)
        aoo = (
            torch.minimum(self.xmax_true, self.xmax_pred)
            - torch.maximum(self.xmin_true, self.xmin_pred)
        ) * (
            torch.minimum(self.ymax_true, self.ymax_pred)
            - torch.maximum(self.ymin_true, self.ymin_pred)
        )

        # Area of overlap limitation
        aoo_limit = torch.minimum(a_true, a_pred)
        aoo[aoo > aoo_limit] = aoo_limit[aoo > aoo_limit]

        # Area of union (aou)
        aou = a_true + a_pred - aoo

        # Intersection of union
        self.iou = torch.clamp(aoo / (aou + self.eps), 0.0, 100.0)

    def reshape_2_to_4(
        self, input_tensor,
    ):
        """
        Changes the dimension 1 of the input tensors from 2 to 4
        
        It is necessary for 1D problems (if there is only the X dimension).
        If the dimension 1 of the input tensors is of 4 (2D problem; both X
        and Y dimensions) applying this function is unnecessary.
        """
        input_tensor = torch.hstack(
            (
                input_tensor[:, 0].reshape(input_tensor.shape[0], 1),
                input_tensor[:, 0].reshape(input_tensor.shape[0], 1),
                input_tensor[:, 1].reshape(input_tensor.shape[0], 1),
                input_tensor[:, 1].reshape(input_tensor.shape[0], 1),
            )
        )
        input_tensor[:, 1] = 0
        input_tensor[:, 3] = 1
        return input_tensor

    def mean_iou_score(self,):
        """
        Mean IoU score function.
        Returns scalar, not tensor.
        """
        miou = torch.mean(self.iou, dim=0).item()
        return miou

    def mean_ciou_loss(self,):
        """
        Mean Complete IoU.
        The idea had been got from the article on "arXiv.org".
        
        Formula:
        
        ciou_loss = 1 - iou + diou_term + aspratio_term = 
        = 1 - iou + (centre_distance / enclosing_box_diag)^2 +
        + v^2 / ((1 - iou) + v)
        
        "diou_term" - distance IoU term
        "aspratio_term" - aspect ratio term
        
        Computation of these terms is implemented in the final step
        to avoid an exceed operation of assignment. This is crucial while
        using GPU.
        """
        # distance IoU term
        centre_distance = torch.hypot(
            ((self.xmin_pred + self.xmax_pred) / 2)
            - ((self.xmin_true + self.xmax_true) / 2),
            ((self.ymin_pred + self.ymax_pred) / 2)
            - ((self.ymin_true + self.ymax_true) / 2),
        )

        enclosing_box_diag = torch.sqrt(
            torch.square(
                torch.maximum(self.xmax_true, self.xmax_pred)
                - torch.minimum(self.xmin_true, self.xmin_pred)
            )
            + torch.square(
                torch.maximum(self.ymax_true, self.ymax_pred)
                - torch.minimum(self.ymin_true, self.ymin_pred)
            )
        )

        # aspect ratio term
        # 4/(pi^2) ~ 0.40528
        v = 0.40528 * torch.square(
            (
                torch.arctan(self.w_true / (self.h_true + self.eps))
                - torch.arctan(self.w_pred / (self.h_pred + self.eps))
            )
        )

        # mean ciou loss
        ciou = (
            1.0
            - self.iou
            + torch.clamp(  # diou_term
                torch.square(centre_distance / (enclosing_box_diag + self.eps)),
                -100.0,
                100.0,
            )
            + torch.clamp(  # aspratio_term
                torch.square(v) / ((1 - self.iou) + v + self.eps), -100.0, 100.0,
            )
        )

        return torch.mean(ciou, dim=0)


class MCIOU_Loss(nn.Module):
    """
    Custom loss function.
    """

    def __init__(self,):
        """
        Initialization.
        """
        super(MCIOU_Loss, self,).__init__()

    def forward(
        self, y_pred, y_true,
    ):
        """
        Computing the loss.
        """
        # mCIoU loss
        iou = torch_iou(y_pred, y_true)
        return iou.mean_ciou_loss()

