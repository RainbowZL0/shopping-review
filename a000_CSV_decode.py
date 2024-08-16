import chardet

# 打开 CSV 文件
with open('C:/Users/46733/Desktop/数据集/waimai_10k.csv', 'rb') as f:
    # 读取文件内容
    content = f.read()
    # 检测文件的编码类型
    result = chardet.detect(content)
    # 打印编码类型
    print(result['encoding'])