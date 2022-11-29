from typing import List
import pandas as pd
import numpy as np





def purhcase_predictors(
    transactions_processed: pd.DataFrame,
    last_date: str,
    train=True
) -> pd.DataFrame:
    """Calculate number of previous purchases for a user.

    Parameters
    ----------
    transactions_processed : pd.DataFrame
        Dataframe of transaction data.
    last_date : pd.Datatime
        Last date to calculate predictors.

    Returns
    -------
    pd.DataFrame
        Modified transaction dataset.
    """
    transactions_processed = transactions_processed[transactions_processed['t_dat'] <= last_date]
    transactions_processed['valid'] = transactions_processed['valid'].astype(int)
    # transactions_processed['price_calc'] = transactions_processed['price'] * transactions_processed['valid']
    # transactions_processed['price_calc'] = transactions_processed['price_calc'].astype(float)

    # sort transactions 
    transactions_processed.sort_values(["customer_id", "t_dat"],
                  axis = 0, ascending = True,
                  inplace = True,
                  na_position = "first")
    

    ## number of purchases
    transactions_processed['purchase_cnt'] = transactions_processed.groupby(['customer_id'])['valid'].cumsum() - transactions_processed['valid']
    transactions_processed['purchase_cnt_cat'] = transactions_processed.groupby(['customer_id', 'index_group_no'])['valid'].cumsum() - transactions_processed['valid']
    return transactions_processed


