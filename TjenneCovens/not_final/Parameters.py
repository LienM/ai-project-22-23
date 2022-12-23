
PARAM = {
    "article": ["article_id", "colour_group_code", "department_no", "garment_group_no", "product_type_name", "section_no", "index_group_no", "detail_desc", "index_code", "graphical_appearance_no"],
    "customer": ["customer_id", "age"],
    "transaction": ['customer_id', 'article_id', 'sales_channel_id', 'price', 't_dat'],
    "samples": ["customer_id", "ordered", "t_dat", "article_id", "department_no", "colour_group_code", "garment_group_no", "section_no", "index_group_no","index_code", "age", "graphical_appearance_no", "material_season", "colour_group_code_season", "price_cat"],
    "article_ranking": ["article_id", "department_no", "colour_group_code", "garment_group_no", "section_no", "index_group_no", "index_code", "age", "graphical_appearance_no", "material_season", "colour_group_code_season", "price_cat"],
    "SELECT": True,
    "SUBMIT": True,
    "LOAD": False,
    "SAVE": False,
    "PP": True,
    "eval": False,
    "score": False

}

FIXED_PARAMS={
    'objective': 'lambdarank',
    'metric': 'map',
    'boosting':'dart',
    'importance_type': "gain",
    'eval_at': 12,
    'n_jobs': 2
    }


SEARCH_PARAMS = {
    'learning_rate': 0.0075,
    'depth': 15,
    'child': 20,
    'estimators': 200,
    'subsample': 0.1,
    'verbose': 0,
    'leaves': 20
    }

# 0.01403
BEST_PARAMS = {
    'learning_rate': 0.01,
    'depth': 15,
    'child': 20,
    'estimators': 10,
    'subsample': 0.2,
    'verbose': 10,
    'leaves': 20

}

# "samples": ["customer_id", "ordered", "article_id", "t_dat", "department_no", "garment_group_no", "index_group_no", "section_no", "colour_group_code_season", "material"],
# "article_ranking": ["article_id", "department_no", "garment_group_no", "index_group_no", "section_no","colour_group_code_season", "material"],


# with color season: 0.00864
# with color season and material: 0.014
# without color season: 0.0011