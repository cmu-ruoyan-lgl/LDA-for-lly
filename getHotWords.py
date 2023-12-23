import os
import re
import csv

# 检查文件是否存在，如果不存在则创建
if not os.path.exists('hotWords.csv'):
    with open('hotWords.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Frequency'])

# 读取source.txt文件
with open('source.txt', 'r') as file:
    data = file.read()

# 提取单词并计数
word_count = {}
words = re.findall(r'\b\w+\b', data, flags=re.UNICODE)
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

# 按照频率降序排序
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# 将结果保存到hotWords.csv文件
with open('hotWords.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['Word', 'Frequency'])
    for word, frequency in sorted_words:
        writer.writerow([word, frequency])
