import os
import re
import csv

# 检查文件是否存在，如果不存在则创建
if not os.path.exists('hotWords.csv'):
    with open('hotWords.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Frequency'])

# 读取source.txt文件
with open('source.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# 提取单词并计数
word_count = {}
words = re.findall(r'\b\w+\b', data, flags=re.UNICODE)
for word in words:
    word_count[word.lower()] = word_count.get(word.lower(), 0) + 1

# 按照频率降序排序
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# 在最后结果保存在hotwords里面之前，有一个屏蔽词文件normalWords.txt如果word在屏蔽词里面就不输出
normal_words = []
with open('normalWords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        normal_words.append(line.strip())
print(normal_words)

# 将结果保存到hotWords.csv文件
with open('hotWords.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Word', 'Frequency'])
    for word, frequency in sorted_words:
        if word not in normal_words:
            writer.writerow([word, frequency])
