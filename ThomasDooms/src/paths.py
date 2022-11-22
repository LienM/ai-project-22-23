def path(dataset, kind):
    ext = "csv" if kind == "original" else "feather"
    ext = "pkl" if dataset == "models" else ext
    return f"../data/{dataset}/{kind}.{ext}"
