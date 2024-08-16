import re
import pandas as pd
import jieba
import jieba.analyse

input_file = 'data/waimai_10k.csv'
output_file = 'data/001waimai_10k.txt'


def process_file_diagram(input_file_0):
    file = pd.read_excel(input_file_0, header=None)
    for index, content_list in file.iterrows():
        text = content_list[1]
        text = remove_punctuation_digits(text)
        text = tokenize(text)
        file.at[index, 1] = text
    file.to_excel(output_file, index=False)


def process_file_txt(input_file_0, output_file_0):
    with open(input_file_0, 'r', encoding='utf-8') as in_file, \
            open(output_file_0, 'w', encoding='utf-8') as out_file:
        line = in_file.readline()
        while line:
            line = remove_punctuation_digits(line)
            label, line = remove_first_word(line)
            line = tokenize(line)
            line = '__label__' + label + ' ' + line

            out_file.write(line)

            line = in_file.readline()


def remove_punctuation_digits(text_0):
    # text_0 = re.sub(r',', ' ', text_0)
    text_0 = re.sub(r'[^\w\s]', '', text_0)
    # text = re.sub(r'\d+', '   ', text)
    return text_0


def remove_first_word(text_0):
    return text_0[0], text_0[1:]


def tokenize(text_0):
    text_0 = jieba.cut(text_0)
    text_list = list(text_0)

    # text_list = jieba.analyse.extract_tags(text_0, topK=10)

    text_string = ' '.join(text_list)

    # text_string = text_string + '\n'

    return text_string


# def remove_douhao(text_0):
#     text_0 = re.sub(r',', ' ', text_0)
#     return text_0

if __name__ == '__main__':
    process_file_txt(input_file, output_file)
