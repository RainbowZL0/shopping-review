import torch
from torch.utils.data import Dataset, DataLoader, random_split


input_txt_path = 'data/001waimai_10k.txt'
out_train_path = 'data/train.txt'
out_test_path = 'data/test.txt'

train_ratio = 0.8

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def txt_to_list(input_txt):
    list_0 = []
    with open(input_txt, 'r', encoding='utf-8') as input_txt_0:
        line = input_txt_0.readline()
        while line:
            list_0.append(line)
            line = input_txt_0.readline()
    return list_0


def output_txt(dataset_0, out_path):
    with open(out_path, 'w', encoding='utf-8') as out_path_0:
        for data in dataset_0:
            out_path_0.write(data)


if __name__ == '__main__':
    txt_list = txt_to_list(input_txt_path)
    dataset = MyDataset(txt_list)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    output_txt(train_dataset, out_train_path)
    output_txt(test_dataset, out_test_path)

