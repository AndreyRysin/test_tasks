import pandas as pd
import numpy as np
import os
import re
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from datetime import datetime


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


def lgbm_feature_importances(models):
    """
    Takes a sequence of fitted LightGBM models (lightgbm.Booster objects),
    extracts feature importance values of each of them (both the 'split' and
    the 'gain'), calculates means along the sequence and, finally, forms and
    returns a sorted table.

    models : sequence of lightgbm.Booster
        A sequence of fitted LightGBM models.

    return : pandas.DataFrame
        A sorted table with mean feature importance values.
    """
    # Collecting the feature importance values
    splits = []
    gains = []
    for model in models:
        splits.append(model.feature_importance(importance_type="split"))
        gains.append(model.feature_importance(importance_type="gain"))
    # Forming the dataframe of means of the feature importances
    feat_imp = pd.DataFrame(
        data=np.array(
            [
                model.feature_name(),
                np.array(np.rint(np.mean(np.array(splits), axis=0)), dtype="int"),
                np.array(np.rint(np.mean(np.array(gains), axis=0)), dtype="int"),
            ]
        ).T,
        columns=["feature_names", "split", "gain"],
    )
    # Sorting the values in descending order
    feat_imp[["split", "gain",]] = feat_imp[
        ["split", "gain"]
    ].astype("int")
    feat_imp = feat_imp.sort_values(by=["split", "gain"], ascending=False)
    return feat_imp


def column_renaming(df, names_dict, shift_size=0):
    """
    Renames the columns according to the dictionary passed.
    The algorithm is based on substrings replacing method.

    df : pandas.DataFrame
        The dataframe whose column names should be changed.

    names_dict : dict
        The dictionary that names should be changed according to.
        Key is the old name, value is the new name.
        For example: {'px_buy_' : 'BID'}

    shift_size : int, default = 0
        Size of the shift that should be applied to all the numerical suffixes.
        For example: if the suffix range starts from value "1" instead of "0"
        ("ASK1", "ASK2", etc, but no zero - "ASK0"), "shift_size" should be set
        to "-1", in order to subtract 1 from every suffix.

    return : pandas.DataFrame
        The initial dataframe with the new column names.
    """
    columns = df.columns.tolist()
    for i in range(len(columns)):
        for name in names_dict.keys():
            if name in columns[i]:
                columns[i] = columns[i].replace(name, names_dict[name])
                break
    for i in range(len(columns)):
        name = re.search(r"\d+", columns[i])
        if name:
            columns[i] = columns[i].replace(name[0], "")
            digit = int(name[0]) + shift_size
            columns[i] = f"{columns[i]}{str(digit)}"
    df.columns = pd.Index(columns)
    return df


def lgbm_algorithm(df, index_tr, index_va, feature_names, target_name, params):
    """
    LightGBM model.
    """
    d_train = lgb.Dataset(
        df.loc[index_tr, feature_names], df.loc[index_tr, target_name], silent=True
    )
    model = lgb.train(params=params, train_set=d_train, verbose_eval=500)
    pred_proba = model.predict(df.loc[index_va, feature_names])
    return model, pred_proba


def ml_trainer_cv(
    df_train,
    feature_names,
    target_name,
    algorithm,
    params,
    seed=42,
    cv=5,
    repeats=3,
    lgbm_seeds_number=3,
    disp_params=False,
    silent=False,
):
    """
    Implementation of the training process applying cross-validation.
    """
    # Init
    rskf = RepeatedStratifiedKFold(n_splits=cv, n_repeats=repeats, random_state=seed)
    models = np.empty(cv * repeats * lgbm_seeds_number, dtype="object")
    scores = np.zeros(cv * repeats, dtype="float")
    # Computing
    print("Computing... ", end="") if silent is not True else None
    for i, (index_tr, index_va) in enumerate(
        rskf.split(df_train[feature_names], df_train[target_name])
    ):
        pred_probas = np.zeros((lgbm_seeds_number, index_va.shape[0]), dtype="float")
        for j in np.arange(lgbm_seeds_number):
            params = params.copy()
            params.update({"seed": params["seed"] * (j + 1)})
            model, pred_proba = algorithm(
                df_train,
                df_train.index[index_tr],
                df_train.index[index_va],
                feature_names,
                target_name,
                params,
            )
            models[i * lgbm_seeds_number + j] = model
            pred_probas[j] = pred_proba
        scores[i] = roc_auc_score(
            df_train.loc[df_train.index[index_va], target_name],
            np.mean(pred_probas, axis=0),
        )
    print("done") if silent is not True else None
    # Displaying and return
    print(f"params: {params}") if ((silent is not True) & disp_params) else None
    return models, np.mean(scores)


def ml_test(df_test, index_te, feature_names, models, disp_params=False, silent=False):
    """
    Implementation of the testing.
    """
    print("Computing... ", end="") if silent is not True else None
    pred_probas = np.zeros((models.shape[0], df_test.shape[0]), dtype="float")
    for i, model in enumerate(models):
        pred_probas[i] = model.predict(df_test.loc[index_te, feature_names])
    pred_proba_mean = np.mean(pred_probas, axis=0)
    print("done") if silent is not True else None
    # Displaying and return
    print(f"params: {params}") if ((silent is not True) & disp_params) else None
    return pred_proba_mean


def print_metrics(y_true, y_pred_proba):
    """
    Displays the computed metrics.
    """
    print("roc auc:  {:.5f}".format(roc_auc_score(y_true, y_pred_proba)))
    print(
        "accuracy: {:.5f}".format(
            accuracy_score(y_true, np.where(y_pred_proba > 0.5, 1, 0))
        )
    )
    print(
        "ac.const: {:.5f}".format(
            np.max(
                (
                    accuracy_score(y_true, np.zeros(y_pred_proba.shape, dtype="float")),
                    accuracy_score(y_true, np.ones(y_pred_proba.shape, dtype="float")),
                )
            )
        )
    )
    print(
        "\nconfusion matrix:\n{}".format(
            np.around(
                confusion_matrix(
                    y_true, np.where(y_pred_proba > 0.5, 1, 0), normalize="all"
                ),
                3,
            )
        )
    )


def form_models_list(model_arrays):
    """
    Forms one-dimensional list from the model arrays produced on training.
    """
    models_list = []
    for model_array in model_arrays:
        models_list.extend(model_array)
    models_list = np.array(models_list)
    return models_list


def save_models_list(models_list, filename, path_root):
    """
    Saves the list of models on a disk.
    """
    if os.path.exists(os.path.join(path_root, "models")) is not True:
        os.mkdir(os.path.join(path_root, "models"))
    with open(os.path.join(path_root, "models", f"{filename}.pkl"), "wb") as f:
        pickle.dump(models_list, f)


def best_params_df(grid_items, indices_best):
    """
    Forms the dataframe of the best models parameters.
    """
    best_params = pd.DataFrame(None, index=grid_items[indices_best[0]])
    for i, idx in enumerate(indices_best):
        best_params.loc[:, f"idx_{idx}"] = grid_items[indices_best[i]].values()
    return best_params


class Bankruptcy_prediction:
    """
    Production module implementing the bankruptcy prediction pipeline.
    """

    def __init__(self, path_root):
        """
        Init function.

        path_root : string
            The root directory. The folder with the models must be inside.
        """
        self.path_root = path_root
        self.lists_init()

    def load_models(self):
        """
        Loads the models.

        The models are hard-set in the code to avoid any issues.
        """
        # models_allfeatures
        with open(
            os.path.join(self.path_root, "models", "models_base_allfeatures.pkl"), "rb"
        ) as f:
            self.models_allfeatures = pickle.load(f)
        # models_wocasetype1frac
        with open(
            os.path.join(self.path_root, "models", "models_base_wocasetype1frac.pkl"),
            "rb",
        ) as f:
            self.models_wocasetype1frac = pickle.load(f)

    def load_tables(self, filename_acc, filename_bank):
        """
        Loads the tables with account information and with the list of
        companies that have been bankrupted.

        filename_acc : string
            Filename of the file with account information.

        filename_bank : string
            Filename of the file with the list of bankrupted companies.
        """
        self.df_acc = pd.read_csv(os.path.join(self.path_root, filename_acc))
        self.df_bank = pd.read_csv(os.path.join(self.path_root, filename_bank))

    def load_court_file(self, path_court, filename_court):
        """
        Loads the file with information about courts the company has been
        being involved in.

        path_court : string
            Path to the folder where the file with information about courts is
            located.

        filename_court : string
            Filename of the file with information about courts.
        """
        with open(os.path.join(path_court, filename_court), "rb") as f:
            self.file_court = pickle.load(f)

    def predict(self):
        """
        Computes the prediction.
        """
        # ++++++++++++ preprocessing ++++++++++++
        # ============ init ============
        # Copying the loaded data
        df_acc = self.df_acc.copy()
        df_bank = self.df_bank.copy()
        file_court = self.file_court.copy()
        self.correct = True
        # ============ court file ============
        # Transforming the dict to the dataframe
        df_court = pd.DataFrame(
            np.empty(
                (
                    np.clip(len(file_court["cases_list"]), 1, None),
                    len(self.df_court_dict_cols),
                ),
                dtype="object",
            ),
            columns=self.df_court_dict_cols,
        )
        if len(file_court["cases_list"]) > 0:
            for i in np.arange(len(file_court["cases_list"])):
                df_court.loc[i, "inn"] = file_court["inn"]
                df_court.loc[i, "caseNo"] = file_court["cases_list"][i]["caseNo"]
                df_court.loc[i, "resultType"] = file_court["cases_list"][i][
                    "resultType"
                ]
                df_court.loc[i, "caseDate"] = file_court["cases_list"][i]["caseDate"]
                df_court.loc[i, "caseType"] = file_court["cases_list"][i]["caseType"][
                    "code"
                ]
                df_court.loc[i, "caseType_name"] = file_court["cases_list"][i][
                    "caseType"
                ]["name"]
                df_court.loc[i, "sum"] = file_court["cases_list"][i]["sum"]
                df_court.loc[i, "isActive"] = file_court["cases_list"][i]["isActive"]
                df_court.loc[i, "currentInstance"] = file_court["cases_list"][i][
                    "currentInstance"
                ]
                df_court.loc[i, "instanceDate"] = file_court["cases_list"][i][
                    "instanceDate"
                ]
                side_type = 0
                for side in file_court["cases_list"][i]["case_sides"]:
                    try:
                        if int(side["INN"]) == int(file_court["inn"]):
                            side_type = side["type"]
                            break
                    except:
                        None
                df_court.loc[i, "case_sides"] = side_type
        else:
            df_court = self.df_court_dummy.copy()
            df_court.loc[0, "inn"] = file_court["inn"]
        # Dropping duplicates
        df_court = df_court.drop_duplicates().reset_index(drop=True)
        # Datetime anomalies reduction
        anomalies_index_instanceDate = df_court["instanceDate"][
            (df_court["instanceDate"].str.slice(start=0, stop=4).astype("int") < 1990)
            | (
                df_court["instanceDate"].str.slice(start=0, stop=4).astype("int")
                > datetime.now().year
            )
        ].index
        anomalies_index_caseDate = df_court["caseDate"][
            (df_court["caseDate"].str.slice(start=0, stop=4).astype("int") < 1990)
            | (
                df_court["caseDate"].str.slice(start=0, stop=4).astype("int")
                > datetime.now().year
            )
        ].index
        index_to_drop = anomalies_index_instanceDate[
            anomalies_index_instanceDate.isin(anomalies_index_caseDate)
        ]  # total abnormalities are dropped
        df_court = df_court.drop(index=index_to_drop).reset_index(drop=True)
        anomalies_index_instanceDate = anomalies_index_instanceDate[
            anomalies_index_instanceDate.isin(index_to_drop) != True
        ]
        anomalies_index_caseDate = anomalies_index_caseDate[
            anomalies_index_caseDate.isin(index_to_drop) != True
        ]
        df_court.loc[anomalies_index_instanceDate, "instanceDate"] = df_court.loc[
            anomalies_index_instanceDate, "caseDate"
        ]
        df_court.loc[anomalies_index_caseDate, "caseDate"] = df_court.loc[
            anomalies_index_caseDate, "instanceDate"
        ]
        # Types conversion
        df_court[["caseType", "case_sides"]] = df_court[
            ["caseType", "case_sides"]
        ].astype("int")
        df_court["inn"] = df_court["inn"].astype("int")
        df_court["sum"] = df_court["sum"].astype("float")
        df_court["isActive"] = (
            df_court["isActive"].apply(lambda x: 1 if x == True else 0).astype("int")
        )
        df_court["caseDate"] = pd.to_datetime(
            df_court["caseDate"], format="%Y-%m-%d"
        ).dt.normalize()
        df_court["instanceDate"] = pd.to_datetime(
            df_court["instanceDate"], format="%Y-%m-%d"
        ).dt.normalize()
        # resultType encoding
        for i in np.arange(df_court.shape[0]):
            try:
                df_court.loc[i, "resultType"] = self.resultType_encode_dict[
                    df_court.loc[i, "resultType"]
                ]
            except:
                df_court.loc[i, "resultType"] = 2
        df_court["resultType"] = df_court["resultType"].astype("int")
        # Dropping duplicates once again
        df_court = df_court.drop_duplicates().reset_index(drop=True)
        df_court = df_court.drop_duplicates(subset="caseNo").reset_index(drop=True)
        # Dropping the unnecessary column
        df_court = df_court.drop(columns="caseNo")
        # Dropping the rows that contain abnormal dates
        df_court = df_court.drop(
            index=df_court[df_court["instanceDate"] < df_court["caseDate"]].index
        ).reset_index(drop=True)
        # ============ accounts table ============
        # Dropping the unnecessary column
        try:
            df_acc = df_acc.drop(columns="Unnamed: 0")
        except:
            None
        # Column renaming
        df_acc = column_renaming(
            df_acc,
            {
                "long_term_liabilities_fiscal_year": "long",
                "short_term_liabilities_fiscal_year": "short",
                "balance_assets_fiscal_year": "balance",
            },
        )
        # Types conversion
        df_acc[["long", "short", "balance"]] = df_acc[
            ["long", "short", "balance"]
        ].astype("float")
        df_acc["inn"] = df_acc["inn"].astype("int")
        # Checking the containing the `inn` in the table
        if df_court.loc[0, "inn"] in df_acc["inn"].values:
            df_acc = df_acc[df_acc["inn"] == df_court.loc[0, "inn"]]
        else:
            df_acc = self.df_acc_dummy.copy()
            df_acc.loc[0, "inn"] = df_court.loc[0, "inn"]
            self.correct = False
            self.message = "There is no such INN in the account table."
        # `okei` column applying
        df_acc["okei"] = (
            df_acc["okei"]
            .where(df_acc["okei"] != 383, 1)
            .where(df_acc["okei"] != 384, 1000)
            .where(df_acc["okei"] != 385, 1000000)
        )
        df_acc[["long", "short", "balance"]] = df_acc[
            ["long", "short", "balance"]
        ].apply(lambda x: x * df_acc["okei"])
        df_acc = df_acc.drop(columns="okei")
        # Filling NaNs
        df_acc = df_acc.fillna(0.0)
        # Dropping duplicates
        df_acc = df_acc.drop_duplicates(subset=["inn", "year"]).reset_index(drop=True)
        # ============ bankruptcies table ============
        # Dropping the unnecessary column
        try:
            df_bank = df_bank.drop(columns="Unnamed: 0")
        except:
            None
        try:
            df_bank = df_bank.drop(columns="bankrupt_id")
        except:
            None
        # Column renaming
        df_bank = column_renaming(df_bank, {"bancrupt_year": "yearb"})
        # Types conversion
        df_bank["yearb"] = df_bank["yearb"].fillna(0).astype("int")
        df_bank["inn"] = df_bank["inn"].astype("int")
        # Dropping duplicates
        df_bank = df_bank.drop_duplicates(subset=["inn"]).reset_index(drop=True)
        # ============ tables matching ============
        # Adding the bankruptcy year column
        df_acc = df_acc.merge(df_bank, on="inn", how="left").fillna(0)
        df_acc["yearb"] = df_acc["yearb"].astype("int")
        # Dealing with incorrect dates
        inn_yearb_more = df_acc[
            (df_acc["yearb"] > df_acc["year"]) & (df_acc["yearb"] > 0)
        ].index
        inn_yearb_less = df_acc[
            (df_acc["yearb"] < df_acc["year"]) & (df_acc["yearb"] > 0)
        ].index
        inn_yearb_less = df_acc[
            (df_acc["inn"].isin(df_acc.loc[inn_yearb_less, "inn"]))
            & (df_acc["inn"].isin(df_acc.loc[inn_yearb_more, "inn"]) != True)
        ].index
        incorrect_inn = df_acc.loc[inn_yearb_less, "inn"].unique()
        if incorrect_inn.shape[0] > 0:
            if incorrect_inn[0] == df_court.loc[0, "inn"]:
                dummy_year = (
                    df_acc[df_acc["inn"] == df_court.loc[0, "inn"]][["year", "yearb"]]
                    .max()
                    .max()
                )
                df_acc.loc[inn_yearb_less, "yearb"] = dummy_year
                df_bank.loc[:, "yearb"][
                    df_bank["inn"] == df_court.loc[0, "inn"]
                ] = dummy_year
                self.correct = False
                self.message = "The year of bankruptcy is wrong."
        # ++++++++++++ feature extraction ++++++++++++
        # ============ accounts table ============
        # Time series tables init and filling
        df_acc_ts = {}
        df_acc_ts["long"] = pd.DataFrame(
            None,
            index=np.sort(df_acc["inn"].unique()),
            columns=np.sort(df_acc["year"].unique()),
        )
        df_acc_ts["short"] = df_acc_ts["long"].copy()
        df_acc_ts["balance"] = df_acc_ts["long"].copy()
        df_acc_inn = df_acc.set_index("inn")
        for col in df_acc_ts["long"].columns:
            for key in df_acc_ts.keys():
                df_acc_ts[key][col] = df_acc_inn[df_acc_inn["year"] == col][key]
        df_acc_ts_diff = {}
        for key in df_acc_ts.keys():
            df_acc_ts_diff[key] = df_acc_ts[key].diff(axis=1)
        # Computing the statistics
        df_acc_stat = pd.DataFrame(None, index=df_acc_ts["long"].index)
        suffix = {0: "", 1: "_diff"}
        for i, dframe in enumerate((df_acc_ts, df_acc_ts_diff)):
            sfx = suffix[i]
            for key in df_acc_ts.keys():
                df_acc_stat.loc[:, f"{key}{sfx}_count"] = dframe[key].count(axis=1)
                df_acc_stat.loc[:, f"{key}{sfx}_mean"] = dframe[key].mean(axis=1)
                df_acc_stat.loc[:, f"{key}{sfx}_std"] = dframe[key].std(axis=1)
                df_acc_stat.loc[:, f"{key}{sfx}_max"] = dframe[key].max(axis=1)
                df_acc_stat.loc[:, f"{key}{sfx}_min"] = dframe[key].min(axis=1)
                df_acc_stat.loc[:, f"{key}{sfx}_incr"] = dframe[key].apply(
                    lambda x: x.is_monotonic_increasing, axis=1
                )
                df_acc_stat.loc[:, f"{key}{sfx}_decr"] = dframe[key].apply(
                    lambda x: x.is_monotonic_decreasing, axis=1
                )
                df_acc_stat[f"{key}{sfx}_incr"] = (
                    df_acc_stat[f"{key}{sfx}_incr"]
                    .where(df_acc_stat[f"{key}{sfx}_incr"] == True, 0)
                    .where(df_acc_stat[f"{key}{sfx}_incr"] == False, 1)
                )
                df_acc_stat[f"{key}{sfx}_decr"] = (
                    df_acc_stat[f"{key}{sfx}_decr"]
                    .where(df_acc_stat[f"{key}{sfx}_decr"] == True, 0)
                    .where(df_acc_stat[f"{key}{sfx}_decr"] == False, 1)
                )
        df_acc_stat = df_acc_stat.merge(
            df_acc.drop_duplicates("inn").set_index("inn")["yearb"],
            left_index=True,
            right_index=True,
            how="outer",
        )
        # Dropping unnecessary columns
        df_acc_stat = df_acc_stat.drop(
            columns=self.cols_ts_to_drop
        )  # TODO: don't create these columns before to skip their dropping
        # Saving the column names for later types conversion
        obj_cols = df_acc_stat.columns[
            df_acc_stat.columns.str.contains("_incr")
            | df_acc_stat.columns.str.contains("_decr")
        ]
        # ============ courts table ============
        # Aggregation table init and filling
        df_court_agg = pd.DataFrame(
            None,
            index=np.sort(df_court["inn"].unique()),
            columns=self.court_agg_columns,
        )
        df_court_agg.loc[:, "court_count"] = df_court["inn"].value_counts().sort_index()
        for d in [0, 1, 2]:
            df_court_agg.loc[:, f"resultType_{d}_frac"] = (
                df_court[df_court["resultType"] == d]
                .groupby("inn")
                .count()
                .sort_index()["resultType"]
                / df_court_agg["court_count"]
            )
        df_court_agg.loc[:, "duration_median"] = (
            (df_court["instanceDate"] - df_court["caseDate"])
            .groupby(df_court["inn"])
            .median()
            .dt.days.sort_index()
        )
        for d in [1, 5, 6]:
            df_court_agg.loc[:, f"caseType_{d}_frac"] = (
                df_court[df_court["caseType"] == d]
                .groupby("inn")
                .count()
                .sort_index()["caseType"]
                / df_court_agg["court_count"]
            )
        df_court_agg.loc[:, "caseType_other_frac"] = (
            df_court[
                (df_court["caseType"] != 1)
                & (df_court["caseType"] != 5)
                & (df_court["caseType"] != 6)
            ]
            .groupby("inn")
            .count()
            .sort_index()["caseType"]
            / df_court_agg["court_count"]
        )
        for agg_method in ["median", "std", "max", "min"]:
            df_court_agg.loc[:, f"sum_{agg_method}"] = (
                df_court["sum"].groupby(df_court["inn"]).agg(agg_method).sort_index()
            )
        df_court_agg.loc[:, "instances_unique_amount"] = (
            df_court["currentInstance"]
            .groupby(df_court["inn"])
            .agg(lambda x: np.unique(x).shape[0])
            .sort_index()
        )
        for d in [1, 5, 6]:
            df_court_agg.loc[:, f"case_side_{d}_frac"] = (
                df_court[df_court["case_sides"] == d]
                .groupby("inn")
                .count()
                .sort_index()["case_sides"]
                / df_court_agg["court_count"]
            )
        # Filling NaNs
        df_court_agg = df_court_agg.fillna(0)
        # ============ tables merging ============
        # Merging
        assert np.all(
            df_acc_stat.index.values == df_court_agg.index.values
        ), "The index of the two aggregated tables is not matched."
        df_feat = df_acc_stat.merge(
            df_court_agg, left_index=True, right_index=True, how="left"
        ).fillna(0)
        df_feat[obj_cols] = df_feat[obj_cols].astype("category")
        # ++++++++++++ computing the prediction ++++++++++++
        # Dropping the target
        df_feat = df_feat.drop(columns="yearb").reset_index(drop=True)
        # Computing
        pred_proba = np.array([])
        if df_feat.loc[0, "caseType_1_frac"] > 0:
            pred_proba = ml_test(
                df_feat,
                df_feat.index,
                self.feature_names_allfeatures,
                self.models_allfeatures,
                silent=True,
            )
        else:
            pred_proba = ml_test(
                df_feat,
                df_feat.index,
                self.feature_names_wocasetype1frac,
                self.models_wocasetype1frac,
                silent=True,
            )
        # Proba extraction
        assert (
            pred_proba.shape[0] == 1
        ), "Error: more than one probability has been computed."
        if self.correct:
            self.proba = pred_proba[0]

    def get_prediction(self, return_proba=True):
        """
        Displays the prediction properly.
        Optionally, returns the probability of the positive class (the
        probabilily of bankruptcy).

        return_proba : bool, default = True
            If True, returns the probability; if False, nothing is returned.

        return : float
            The probability of the positive class (bankruptcy).
        """
        if self.correct:
            print("Bankruptcy probability: {:.5f}".format(self.proba))
            return self.proba if return_proba else None
        else:
            print(self.message)

    def lists_init(self):
        """
        Provides initialization of the lists and dummies that are always
        the same.
        """
        self.df_court_dict_cols = [
            "inn",
            "caseNo",
            "resultType",
            "caseDate",
            "caseType",
            "sum",
            "isActive",
            "currentInstance",
            "instanceDate",
            "case_sides",
        ]
        self.resultType_encode_dict = {
            "Не удалось определить": 2,
            "Проиграно": 0,
            "Выиграно": 1,
            "Частично проиграно": 0,
            "Не выиграно": 2,
            "Частично выиграно": 1,
            "Не проиграно": 2,
            "Иск полностью удовлетворен": 1,
            "В иске отказано полностью": 0,
            "Иск частично удовлетворен": 1,
            "Утверждено мировое соглашение": 1,
            "Иск не рассмотрен": 2,
            "Прекращено производство по делу": 2,
            "Иск полностью удовлетворен, встречный частично удовлетворен": 1,
            "Иск частично удовлетворен, встречный не удовлетворен": 1,
            "В иске отказано частично": 0,
        }
        self.cols_ts_to_drop = [
            "short_incr",
            "short_decr",
            "long_diff_incr",
            "long_diff_decr",
            "short_diff_incr",
            "short_diff_decr",
            "balance_diff_incr",
            "balance_diff_decr",
        ]  # TODO: don't create columns `self.cols_ts_to_drop` to skip their later dropping
        self.court_agg_columns = [
            "court_count",
            "resultType_0_frac",
            "resultType_1_frac",
            "resultType_2_frac",
            "duration_median",
            "caseType_1_frac",
            "caseType_5_frac",
            "caseType_6_frac",
            "caseType_other_frac",
            "sum_median",
            "sum_std",
            "sum_max",
            "sum_min",
            "instances_unique_amount",
            "case_side_0_frac",
            "case_side_1_frac",
            "case_side_2_frac",
            "case_side_3_frac",
        ]
        self.feature_names_allfeatures = [
            "long_count",
            "long_mean",
            "long_std",
            "long_max",
            "long_min",
            "long_incr",
            "long_decr",
            "short_count",
            "short_mean",
            "short_std",
            "short_max",
            "short_min",
            "balance_count",
            "balance_mean",
            "balance_std",
            "balance_max",
            "balance_min",
            "balance_incr",
            "balance_decr",
            "long_diff_count",
            "long_diff_mean",
            "long_diff_std",
            "long_diff_max",
            "long_diff_min",
            "short_diff_count",
            "short_diff_mean",
            "short_diff_std",
            "short_diff_max",
            "short_diff_min",
            "balance_diff_count",
            "balance_diff_mean",
            "balance_diff_std",
            "balance_diff_max",
            "balance_diff_min",
            "court_count",
            "resultType_0_frac",
            "resultType_1_frac",
            "resultType_2_frac",
            "duration_median",
            "caseType_1_frac",
            "caseType_5_frac",
            "caseType_6_frac",
            "caseType_other_frac",
            "sum_median",
            "sum_std",
            "sum_max",
            "sum_min",
            "instances_unique_amount",
            "case_side_0_frac",
            "case_side_1_frac",
            "case_side_2_frac",
            "case_side_3_frac",
            "case_side_5_frac",
            "case_side_6_frac",
        ]
        self.feature_names_wocasetype1frac = [
            "case_side_1_frac",
            "short_std",
            "resultType_1_frac",
            "court_count",
            "sum_std",
            "balance_diff_min",
            "sum_median",
            "sum_max",
            "duration_median",
            "balance_diff_mean",
            "short_diff_min",
            "balance_min",
            "short_max",
            "long_count",
            "short_mean",
            "resultType_0_frac",
            "caseType_5_frac",
            "short_diff_mean",
            "balance_diff_std",
            "long_diff_min",
            "balance_std",
            "short_diff_std",
            "short_diff_max",
            "short_count",
            "long_diff_count",
            "balance_max",
            "caseType_6_frac",
            "resultType_2_frac",
            "instances_unique_amount",
        ]
        # df_court_dummy
        self.df_court_dummy = pd.DataFrame(
            np.empty((1, len(self.df_court_dict_cols)), dtype="object"),
            columns=self.df_court_dict_cols,
        )
        self.df_court_dummy.loc[0, "inn"] = 0
        self.df_court_dummy.loc[0, "caseNo"] = str(np.random.randint(1e10))
        self.df_court_dummy.loc[0, "resultType"] = "Не удалось определить"
        self.df_court_dummy.loc[0, "caseDate"] = "2021-08-09T10:00:00"
        self.df_court_dummy.loc[0, "caseType"] = "90"
        self.df_court_dummy.loc[0, "caseType_name"] = "не определено"
        self.df_court_dummy.loc[0, "sum"] = 0.0
        self.df_court_dummy.loc[0, "currency"] = 643
        self.df_court_dummy.loc[0, "isActive"] = False
        self.df_court_dummy.loc[0, "currentInstance"] = "none"
        self.df_court_dummy.loc[0, "instanceDate"] = "2021-08-09T10:00:00"
        self.df_court_dummy.loc[0, "case_sides"] = 2
        # df_acc_dummy
        self.df_acc_dummy = pd.DataFrame(
            {
                "inn": 0,
                "year": 2021,
                "long": 0.0,
                "short": 0.0,
                "balance": 0.0,
                "okei": 384,
            },
            index=[0],
        )
        # df_bank_dummy
        self.df_bank_dummy = pd.DataFrame(
            {
                "yearb": 2021,
                "inn": 0,
            },
            index=[0],
        )
