def add_mas(df, ma_settings):
    """
    ma_settings: [('EMA', 21, True), ('SMA', 50, True), ...]
    """
    for kind, period, enabled in ma_settings:
        if not enabled:
            continue
        col = f"{kind}{period}"
        if kind == "EMA":
            df[col] = df["Close"].ewm(span=period).mean()
        else:  # SMA
            df[col] = df["Close"].rolling(window=period).mean()
    return df
