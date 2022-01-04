def min_max(x, m=None, M=None):
    if m is not None and M is not None:
        assert m <= M
    if m is not None:
        x = max(x, m)
    if M is not None:
        x = min(x, M)
    return x
