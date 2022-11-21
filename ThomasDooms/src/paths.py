def path(dataset, kind):
    ext = "csv" if kind == "original" else "feather"
    return f"../data/{dataset}/{kind}.{ext}"
