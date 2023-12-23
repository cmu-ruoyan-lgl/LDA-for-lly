import pandas as pd
import gensim
from gensim import corpora

# 1. 加载hotwords.csv文件
data = pd.read_csv('hotwords.csv')

# 2. 数据预处理
# 将词袋转换为gensim.corpora.textcorpus.TextCorpus对象
texts = data['Word'].apply(lambda x: x.split())
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 3. 创建LDA模型
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=10,   # 设置主题数
                                   passes=10,       # 迭代次数
                                   random_state=123) # 随机种子

# 4. 可视化主题
import pyLDAvis.gensim_models
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
print('LDA可视化结果')
print(vis)
print('LDA可视化结果保存为html文件')
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print('LDA模型训练完成')