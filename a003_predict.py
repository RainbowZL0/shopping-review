import fasttext
import numpy as np
import matplotlib.pyplot as plt

# 读取训练数据
train_file = './data/train.txt'
test_file = './data/test.txt'
pretrained_vectors = './data/merge_sgns_bigram_char300.vec'


def train_model(train_file_0, learning_rate, epoch):
    """
    Train the model.
    """
    model_0 = fasttext.train_supervised(input=train_file_0, epoch=epoch, lr=learning_rate, wordNgrams=3, loss='ns',
                                        dim=50, ws=5)
    return model_0


def get_real_list(input_txt_0):
    real_label_list = []
    real_data_list = []
    with open(input_txt_0, 'r', encoding='utf-8') as input_txt_0:
        line = input_txt_0.readline()
        while line:
            label = line[9]  # 标签是第9位
            real_label_list.append(label)

            data = line[11:]  # 提取文本data
            data = data.strip()
            real_data_list.append(data)

            line = input_txt_0.readline()
    return real_label_list, real_data_list


def predict_result_list(model_0, real_data_list_0, real_label_list_0):
    predict_label_list_0 = []
    matrix_dict_0 = {}
    index = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for real_data in real_data_list_0:
        prediction = model_0.predict(real_data)
        predicted_label = prediction[0][0]  # prediction是由一些两个元素的元组构成的列表。[(label, probability), (), ...]
        predicted_label = predicted_label[-1]
        predict_label_list_0.append(predicted_label)

        real_label = real_label_list_0[index]  # 取出real_label, 比较real和predict
        index += 1

        if real_label == '1' and predicted_label == '1':
            true_positive += 1
        if real_label == '0' and predicted_label == '0':
            true_negative += 1
        if real_label == '0' and predicted_label == '1':
            false_positive += 1
        if real_label == '1' and predicted_label == '0':
            false_negative += 1

    matrix_dict_0['TP'] = true_positive
    matrix_dict_0['TN'] = true_negative
    matrix_dict_0['FP'] = false_positive
    matrix_dict_0['FN'] = false_negative

    accuracy = (true_positive + true_negative) / len(real_label_list_0)

    return predict_label_list_0, matrix_dict_0, accuracy


def picture(label_list_0, data_list_0):
    learning_rates = np.geomspace(0.0001, 0.1, num=50)  # 等比数列
    acc_list = []
    for lr in learning_rates:
        print(lr)
        model = train_model(train_file_0=train_file, learning_rate=lr, epoch=20)  # 训练
        predict_label_list, matrix_dict, acc = predict_result_list(model, data_list_0, label_list_0)
        acc_list.append(acc)

    plt.plot(learning_rates, acc_list, marker='o')
    plt.xscale('log')
    plt.title('Accuracy vs. learning rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.show()


def generate_matrix(label_list_0, data_list_0):
    epochs = [1, 3, 10]
    for epoch in epochs:
        model = train_model(train_file_0=train_file, learning_rate=0.04, epoch=epoch)  # 训练
        predict_label_list, matrix_dict, acc = predict_result_list(model, data_list_0, label_list_0)
        print(f'epoch = {epoch}')
        print(matrix_dict)
        print()


if __name__ == '__main__':
    label_list, data_list = get_real_list(input_txt_0=test_file)

    # picture(label_list, data_list)
    generate_matrix(label_list, data_list)

    # for key, value in matrix_dict.items():
    #     print(key, value)
