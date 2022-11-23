import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from tqdm import tqdm
import math
# import implicit
from scipy.sparse import coo_matrix

# * scores of rules are the bigger the better

class PersonalRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each customer."""

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items
        Returns:
            pd.DataFrame: (customer_id, article_id, method, score)
        """


class UserGroupRetrieveRule(ABC):
    """Use certain rules to respectively retrieve items for each group of customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[*self.cat_cols, self.iid, "method", "score"]]

        user = self.data["user"][[*self.cat_cols, "customer_id"]]
        tmp_df = pd.DataFrame({"customer_id": self.customer_list})
        tmp_df = tmp_df.merge(user, on="customer_id", how="left")
        tmp_df = tmp_df.merge(result, on=[*self.cat_cols], how="left")

        tmp_df = tmp_df[["customer_id", *self.cat_cols, self.iid, "score", "method"]]

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items
        Returns:
            pd.DataFrame: (*group_cols, article_id, method, score)
        """

class GlobalRetrieveRule(ABC):
    """Use certain rules to retrieve items for all customers."""

    def merge(self, result: pd.DataFrame):
        result = result[[self.iid, "method", "score"]]

        num_item = result.shape[0]
        num_user = self.customer_list.shape[0]

        tmp_user = np.repeat(self.customer_list, num_item)
        tmp_df = result.iloc[np.tile(np.arange(num_item), num_user)]
        tmp_df = tmp_df.reset_index(drop=True)
        tmp_df["customer_id"] = tmp_user

        return tmp_df

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Retrieve items
        Returns:
            pd.DataFrame: (article_id, method, score)
        """


class FilterRule(ABC):
    """Use certain rules to remove some retrieved items."""

    @abstractmethod
    def retrieve(self) -> List:
        """Retrieve items
        Returns:
            List: items to be removed
        """


# * ======================= Personal Retrieve Rules ======================= *

class OrderHistory(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer."""

    def __init__(
        self,
        trans_df: pd.DataFrame,
        days: int = 7,
        n: int = None,
        name: str = "1",
        item_id: str = "article_id",
        scale: bool = True,
    ):
        """Initialize OrderHistory.
        Parameters
        ----------
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        days : int, optional
            Length of time window when getting user buying history, by default ``7``.
        n : int, optional
            Get top `n` recently bought items, by default ``None``.
        name : str, optional
            Name of the rule, by default ``1``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]]
        self.days = days
        self.n = n
        self.name = name
        self.scale = scale

    def retrieve(self) -> pd.DataFrame:
        res = self.trans_df.reset_index()
        res["max_dat"] = res.groupby("customer_id").t_dat.transform(max)
        res["diff_dat"] = (res.max_dat - res.t_dat).dt.days
        res = res.loc[res["diff_dat"] < self.days].reset_index(drop=True)

        res = res.sort_values(by=["diff_dat"], ascending=True).reset_index(drop=True)
        res = res.groupby(["customer_id", self.iid], as_index=False).first()
        res["rank"] = res.groupby(["customer_id"])["diff_dat"].rank(
            ascending=False, method="min"
        )


        if self.n is not None:
            res = res.loc[res["rank"] <= self.n]

        res["score"] = -res["diff_dat"]
        

        res["method"] = f"OrderHistory_{self.name}"
        res = res[["customer_id", self.iid, "score", "method"]]

        return res



# * ======================== Global Retrieve Rules ======================== *

class TimeHistory(GlobalRetrieveRule):
    """Retrieve popular items in specified time window."""

    def __init__(
        self,
        customer_list: List,
        trans_df: pd.DataFrame,
        n: int = 12,
        days: int = None,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
        scale: bool = True,
    ):
        """Initialize TimeHistory.
        Parameters
        ----------
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.iid = item_id
        self.customer_list = customer_list
        self.trans_df = trans_df[["customer_id", self.iid, "t_dat"]]
        self.unique = unique
        self.n = n
        self.days = days
        self.name = name
        self.scale = scale

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window
        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        # df["t_dat"] = pd.to_datetime(df["t_dat"])
        max_date = df["t_dat"].max()
        if self.days is not None:
          df = df.loc[(max_date - df["t_dat"]) <= pd.Timedelta(days=self.days)]

        if self.unique:
          df = df.drop_duplicates(["customer_id", self.iid])

        df["count"] = 1
        df = df.groupby(self.iid, as_index=False)["count"].sum()
        df["rank"] = df["count"].rank(ascending=False, method='min')
        df["method"] = "TimeHistory_" + self.name

        df = df[df["rank"] <= self.n]


        df["score"] = df["rank"]
        
        df = df[[self.iid, "score", "method"]]
        df = self.merge(df)

        return df[["customer_id", self.iid, "method", "score"]]

# * ====================== User Group Retrieve Rules ====================== *

class UserGroupTimeHistory(UserGroupRetrieveRule):
    """Retrieve popular items of each **user** group in specified time window."""

    def __init__(
        self,
        data: Dict,
        customer_list: List,
        trans_df: pd.DataFrame,
        cat_cols: List,
        n: int = 12,
        name: str = "1",
        unique: bool = True,
        item_id: str = "article_id",
        scale: bool = True,
    ):
        """Initialize TimeHistory.
        Parameters
        ----------
        data : Dict
            Data dictionary.
        customer_list : List
            List of target customer ids.
        trans_df : pd.DataFrame
            Dataframe of transaction records.
        cat_cols: List
            Name of user group columns.
        n : int, optional
            Get top `n` popular items, by default ``12``.
        name : str, optional
            Name of the rule, by default ``1``.
        unique : bool, optional
            Whether to drop duplicate buying records, by default ``True``.
        item_id : str, optional
            Name of item id, by default ``"article_id"``.
        """
        self.data = data
        self.customer_list = customer_list
        self.iid = item_id
        self.trans_df = trans_df[["customer_id", self.iid, *cat_cols]]
        self.cat_cols = cat_cols
        self.unique = unique
        self.n = n
        self.name = name
        self.scale = scale

    def retrieve(self) -> List[int]:
        """Get popular items in the specified time window
        Returns:
            List[int]: top n popular items
        """
        df = self.trans_df
        if self.unique:
            df = df.drop_duplicates(["customer_id", self.iid])

        df["count"] = 1

        df = df.groupby([*self.cat_cols, self.iid], as_index=False)["count"].sum()
        df["rank"] = df.groupby([*self.cat_cols])["count"].rank(
            ascending=False, method="min"
        )

        df["method"] = "TimeHistory_" + self.name

        if self.scale:
            df["score"] = df["count"] / df["count"].max()
        else:
            df["score"] = df["count"]
        df["method"] = "UGTimeHistory_" + self.name
        df = df[df["rank"] <= self.n][[*self.cat_cols, self.iid, "score", "method"]]
        df = self.merge(df)
        return df[["customer_id", self.iid, "method", "score", *self.cat_cols]]


## helping functions

# https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision/notebook

def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k

def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0
    
def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        
    return ap / min(k, len(y_true))


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])





# https://github.com/Wp-Zhang/H-M-Fashion-RecSys/blob/main/src/data/datahelper.py

import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from sklearn import preprocessing

class DataLoader:
    """Helper class for loadiing, preprocessing and saving data."""

    def __init__(self, data_dir: str):
        """Initialize DataHelper.
        Parameters
        ----------
        data_dir : str
            Data directory.
        raw_dir : str
            Subdirectory to store raw data.
        """
        self.base = Path(data_dir)  # data diectory
        self.raw_dir = self.base # raw data directory

    def _load_raw_data(self) -> dict:
        """Load original raw data
        Returns
        -------
        dict
            Data dictionary, keys: 'item', 'user', 'inter'.
        """

        articles = pd.read_csv(self.raw_dir / "articles.csv")
        customers = pd.read_csv(self.raw_dir / "customers.csv")
        inter = pd.read_csv(self.raw_dir / "transactions_train.csv")
        sample_sub = pd.read_csv(self.raw_dir / "sample_submission.csv")
        
        return {"item": articles, "user": customers, "inter": inter, "subm": sample_sub}



    def parse(self, x):
        l = ['0' + str(i) for i in x]
        l = ' '.join(l[:12])
        return l


    def save_submission(self, data: pd.DataFrame, name: str):
        """Save data dictionary as parquet
        Parameters
        ----------
        data : dict
            Data dictionary, keys: 'item', 'user', 'inter'.
        name : str
            Name of the dataset.
        """
        data.rename(columns={'article_id': 'prediction'}, inplace=True)
        data['prediction'] = data['prediction'].apply(lambda x: self.parse(x))
        path = self.base / "processed"
        file_name = "submission_" + name + ".csv"
        if not os.path.exists(path):
            os.mkdir(path)
        data[['customer_id', 'prediction']].to_csv(path / file_name, index=False)
        return data


    def load_data(self, name: str) -> dict:
        """Load data dictionary from parquet.
        Parameters
        ----------
        name : str
            Name of the dataset.
        Returns
        -------
        dict
            Data dictionary, keys: 'item', 'user', 'inter'.
        Raises
        ------
        OSError
            If the directory does not exist.
        """
        path = self.base / "processed" / name
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")
        data = {}
        data["user"] = pd.read_parquet(path / "user.pqt")
        data["item"] = pd.read_parquet(path / "item.pqt")
        data["inter"] = pd.read_parquet(path / "inter.pqt")
        data["subm"] = pd.read_parquet(path / "subm.pqt")
        return data


    def cut_data(self, trans_df, weeks=5) -> pd.DataFrame:
        train_start_date = trans_df['t_dat'].max() - pd.Timedelta(weeks=weeks)
        trans_df = trans_df[trans_df['t_dat'] >= train_start_date]
        return trans_df
    
  


    def _base_transform(self, data) -> dict:
        """Returns
        -------
        dict
            Preprocessed data.
        """
        inter = data["inter"]
        user = data["user"]
        item = data["item"]
        subm = data["subm"]

        user['customer_id'] = user['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
        inter['customer_id'] = inter['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
        subm['customer_id'] = subm['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
        
        item['article_id'] = item['article_id'].astype('int32') 
        inter['article_id'] = inter['article_id'].astype('int32') 

        ### user basic features
        user['age'].fillna(int(user['age'].median()), inplace=True)
        user['fashion_news_frequency'].fillna("NONE")
        user = user.fillna(0)
        user = pd.get_dummies(user, columns=['fashion_news_frequency'])
        user = pd.get_dummies(user, columns=['club_member_status'])
        user["FN"] = user["FN"].astype('int8')
        user["Active"] = user["Active"].astype('int8')

        ### transaction basic features
        inter['sales_channel_id'] = inter['sales_channel_id'].replace(2, 0)
        inter = inter.fillna(0)
        inter['t_dat'] = pd.to_datetime(inter['t_dat'])
        inter["price"] = inter["price"].astype("float32")
        inter["sales_channel_id"] = inter["sales_channel_id"].astype("int8")
        
        ### item basic features
        item['index_group_name_copy'] = item['index_group_name']
        item = pd.get_dummies(item, columns=['index_group_name'])

        data['user'] = user
        data['item'] = item
        data['inter'] = inter
        data['subm'] = subm

        return data


    def read_submission_data(self):
        sample_sub = pd.read_csv(self.raw_dir / "sample_submission.csv")
        sample_sub['customer_id_2'] = sample_sub['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
        return sample_sub
        

    def preprocess_data(self, save: bool = True, name: str = "encoded_full", weeks=8) -> dict:
        """Preprocess raw data:
            1. encode ids
            2. label encode categorical features
            3. impute
        Parameters
        ----------
        save : bool, optional
            Whether to save the preprocessed data, by default ``True``.
        name : str, optional
            Version name of the data to be saved, by default ``"encoded_full"``.
        Returns
        -------
        dict
            Preprocessed data.
        """
        data = self._load_raw_data()
        data = self._base_transform(data)
        data['inter'] = self.cut_data(data['inter'], weeks=weeks)
        if save:
            self.save_data(data, name + str(weeks))
        return data
        

    def save_data(self, data: dict, name: str):
        """Save data dictionary as parquet
        Parameters
        ----------
        data : dict
            Data dictionary, keys: 'item', 'user', 'inter'.
        name : str
            Name of the dataset.
        """
        path = self.base / "processed" / name
        if not os.path.exists(path):
            os.mkdir(path)

        
        data["user"].to_parquet(path / "user.pqt")
        data["item"].to_parquet(path / "item.pqt")
        data["inter"].to_parquet(path / "inter.pqt")
        data["subm"].to_parquet(path / "subm.pqt")

    def split_data(
        self,
        trans_data: pd.DataFrame,
        train_end_date: str,
        valid_end_date: str,
        item_id: str = "article_id",
    ) -> Tuple[pd.DataFrame]:
        """Split transaction data into train set and valid set
        Parameters
        ----------
        trans_data : pd.DataFrame
            Transaction dataframe.
        train_end_date : str
            End date of train set, max(train_set.date) <= train_end_date.
        valid_end_date : str
            End date of valid set, max(valid_set.date) <= valid_end_date.
        item_id : str, optional
            Name of item id, can be `article_id` or `product_code`, etc. By default ``"article_id"``.
        Returns
        -------
        Tuple[pd.DataFrame]
            [train set, valid set]
        Raises
        ------
        KeyError
            If item_id is not in `trans_data` columns.
        """
        if item_id not in trans_data.columns:
            raise KeyError(f"{item_id} is not one of the columns")

        train_set = trans_data.loc[trans_data["t_dat"] <= train_end_date]
        valid_set = trans_data.loc[
            (train_end_date < trans_data["t_dat"])
            & (trans_data["t_dat"] <= valid_end_date)
        ]
        valid_set_grouped = (
            valid_set.groupby(["customer_id"])[item_id].apply(list).reset_index()
        )

        return train_set, valid_set_grouped, valid_set 

        
                            