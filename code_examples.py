"""
Here some examples of my code are placed.
This is not a library itself just only something like a portfolio. The original library is private.
All the functions are workable, but their versions were current at the moment they were copied here not now.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display
from datetime import datetime


def any_precision_round(a, precision=1.0):
    """
    Rounds a value (or array-like) out to the nearest multiple of the precision.
    If a precision value is within [10^N; N >> integer], the function works as a
    common round function.
    
        Examples:
        precision = 0.02; 2.845 >> 2.840;  2.852 >> 2.860
        precision = 15;   6.426 >> 0.0;   13.426 >> 15.0
    
    The function solves the problem of the incorrect rounding of values that are
    almost in the middle of the interval between two neighbour multiples of the
    precision.
        
        Examples: 1.499999999; 2.500000001 etc.
        
    The problem arises while the binary representation of decimals.
    The error doesn't arise up to 1e-10 precision of rounding.
    
    a : scalar or array-like
        A value (array of values) to round.
    
    precision : float, default = 1.0
        A multiple of the precision, which is the closest to "a", is the
        rounded value of "a".
    
    return : float or numpy.array
        The rounded value (array of values) according to the input.
    """

    # Recognizing is a scalar or a sequence for providing the correct handling
    a = a
    isscalar = False

    if np.issctype(type(a)):
        a = np.array([a])
        isscalar = True
    else:
        a = np.array(a)
        isscalar = False

    # Computing the rounded value(s)
    a = np.divide(a, precision)

    mask_unequal = np.mod(np.round(a, 12), 0.5) != 0.0
    mask_equal = np.mod(np.round(a, 12), 0.5) == 0.0

    a[mask_unequal] = np.round(a[mask_unequal], 0)
    a[mask_equal] = np.ceil(a[mask_equal])

    a = np.multiply(a, precision)

    # Truncating deep resuduals
    precision_depth = 0
    try:
        precision_depth = len(str(precision).split(".")[1])
    except:
        precision_depth = len(str(precision))

    a = np.round(a, precision_depth)

    # Converting the value to a scalar if it is necessary
    if isscalar:
        a = a[0]

    return a


def qq_plot(
    arr, figsize=(8, 8),
):
    """
    Plots quantile-quantile plot.
    Performs the Shapiro-Wilk test for normality.
    
    For observing the plot of the distribution of 1000 random values
    ("ideal" normal distribution) pass zero to the input variable "arr"
    instead of an array.
    
    For correct plotting, "matplotlib.pyplot" must be imported as "plt".
    
    arr : array-like of numeric
        The array whose distribution is analyzed.
        If zero passed instead of an array the normal distribution plot is built.
    
    figsize : tuple of int of size 2, default = (8, 8)
        Size of the plot.
    
    return : float
        P-value of the Shapiro-Wilk test.
    """
    # If necessary, forming the array of random values for plotting
    # the qq_plot of the normal distribution.
    try:
        if arr == 0:
            arr = np.array(np.round(np.random.rand(1000) * 100), dtype=np.int64)
    except:
        None

    # Performing the Shapiro-Wilk test
    _, pvalue = stats.shapiro(arr)
    print("p-value: {:.05f}".format(pvalue))

    # Plotting
    plt.figure(figsize=figsize)
    stats.probplot(
        arr, dist="norm", plot=plt,
    )
    plt.show()

    # Returning the p-value
    return pvalue


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


def index(df, col_name):
    """
    Returns the index (number) of the column by its name.

    df : pandas.DataFrame
        The dataframe whose column is considered.
    
    col_name : string
        The name of the column whose index is being looked for.
    
    return : int or numpy.nan
        The number of the column if it exists, numpy.nan otherwise.
    """

    index = pd.Series(df.columns)
    index = index[df.columns == col_name]
    if index.shape[0] == 1:
        return index.index[0]
    else:
        print(f'There is no column "{col_name}" here.')
        return np.nan


def delete_cols(
    df, col_names_list,
):
    """
    Deletes the columns which names are passed.
    This is implemented via slicing not dropping because of higher performance.
    
    df : pandas.DataFrame
        The dataframe the columns are deleted from.
    
    col_name : list of strings
        Names of the columns being deleted.
    
    return : pandas.DataFrame
        The dataframe with no columns deleted.
    """

    return df[df.columns[df.columns.isin(col_names_list) != True]]


def groupby_fillna(
    df, groupers, target, aggfunc="median",
):
    """
    Fills NaNs in the target column with the aggregated values calculated
    individually for each group of entries.
    
    Entries are grouped by the columns from "groupers" (the core method is
    "groupby()" from pandas). Aggregated values for each of the groups are
    calculated from target values that are not NaNs.
    
    Each aggregated value is used for filling NaNs only within its group. Such
    an approach helps to achieve higher precision.
    
    The function works well with category and string dtypes too. In these
    cases, the aggregation function is "mode" (the most frequent value within
    the group) regardless of the aggfunc value. The function "mode" is realized
    as an internal function.
    
    df : pandas.DataFrame
        A dataframe containing columns from "groupers" and a target column.
    
    groupers : sequence of strings
        Names of the columns of a given dataframe that are used for grouping
        entries.
    
    target : string
        Name of the column NaNs are filled in.
    
    aggfunc : string, default = 'median'
        Name of the aggregation function used for calculating aggregated values.
        If dtype of the target is "category" or "object" the parameter is
        ignored: NaNs are filled with modes instead.
    
    return: pandas.DataFrame
        The initial dataframe with NaNs of the target column filled.
    """

    # Recognizing the target dtype and changing aggfunc if necessary
    target_dtype = str(df[target].dtype)
    aggfunc = aggfunc
    if (target_dtype == "category") | (target_dtype == "object"):
        aggfunc = "count"

    # Calculating the aggregated values of the target for each group
    target_groups = (
        df[df[target].isna() != True]
        .groupby(groupers)
        .agg(aggfunc)
        .reset_index()[[*groupers, target]]
        .dropna()
        .reset_index(drop=True)
    )

    # If dtype of the target is "category" or "object" aggregated function is mode
    def mode_category(
        x, df, groupers, target,
    ):
        """
        Searches and returns the mode value.
        """
        condition = 0 == 0
        for grouper in groupers:
            condition = condition & (df[grouper] == x[grouper])
        return df[condition][target].value_counts().idxmax()

    if (target_dtype == "category") | (target_dtype == "object"):
        target_groups["cat"] = target_groups.apply(
            mode_category, axis=1, df=df, groupers=groupers, target=target,
        )
        target_groups = target_groups.drop(columns=target).rename(
            columns={"cat": target}
        )
        target_groups[target] = target_groups[target].astype(target_dtype)

    # Filling NaNs with the aggregated value group by group.
    target_groups_without_target = target_groups.drop(columns=target)

    for i in range(target_groups.shape[0]):
        target_groups_dict = target_groups_without_target.loc[i].to_dict()

        condition = 0 == 0
        for key in target_groups_dict.keys():
            condition = condition & (df[key] == target_groups_dict[key])

        df.loc[df[condition].index, target] = df.loc[
            df[condition].index, target
        ].fillna(target_groups.loc[i, target])

    return df


def columns_replace(df, col_a, col_b):
    """
    Changes the column order by swapping the given columns.
    
    df : pandas.DataFrame
        A dataframe that columns should be swapped of.
        
    col_a, col_b : string
        Names of the columns to swap.
    
    return : pandas.DataFrame
        A dataframe with the new order of columns.
    """

    idx_a = index(df, col_a)
    idx_b = index(df, col_b)

    idx_min = idx_a
    idx_max = idx_b
    if idx_b < idx_a:
        idx_min = idx_b
        idx_max = idx_a

    columns_old = df.columns.tolist()
    columns_new = [
        *columns_old[:idx_min],
        columns_old[idx_max],
        *columns_old[idx_min + 1 : idx_max],
        columns_old[idx_min],
        *columns_old[idx_max + 1 :],
    ]

    return df[columns_new]

