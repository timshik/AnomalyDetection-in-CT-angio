from MainProcess import evaluate
from DataLoader import *


def test(model, path, mask, rate):
    data = get_all_data_dataloader(path, mask, rate)
    evaluate(model, data, 'test1')

