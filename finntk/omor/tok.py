def get_token_positions(tokenised, text):
    """
    Returns the start positions of a series of tokens produced by
    Omorfi.tokenise(...)
    """
    starts = []
    start = 0
    for token in tokenised:
        start = text.index(token['surf'], start)
        starts.append(start)
    return starts
