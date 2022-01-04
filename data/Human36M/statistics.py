import torch


MEDIAN_PIXEL = [89 / 255, 35 / 255, 36 / 255]  # From compute_median_background_pixel.py
MEDIAN_PIXEL = [244 / 255, 191 / 255, 175 / 255]  # But we prefer using this one, that was manually selected.
MEDIAN_PIXEL = torch.tensor(MEDIAN_PIXEL)

WIDTHS = {
    (0, 1): 0.045,
    (1, 2): 0.075,
    (2, 3): 0.050,
    (0, 4): 0.045,
    (4, 5): 0.075,
    (5, 6): 0.050,
    (0, 7): 0.050,
    (7, 8): 0.050,
    (8, 9): 0.020,
    (9, 10): 0.020,
    (8, 11): 0.040,
    (11, 12): 0.040,
    (12, 13): 0.035,
    (8, 14): 0.040,
    (14, 15): 0.040,
    (15, 16): 0.035,
}
WIDTHS = torch.tensor([v for v in WIDTHS.values()]).reshape(-1, 1, 1)
