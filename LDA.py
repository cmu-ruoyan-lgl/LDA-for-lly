import pandas as pd
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer

# The LDA parameters are "α = 50 / k and £ = 0.1".

# 1. 加载hotwords.csv文件
data = pd.read_csv('hotwords.csv')

# 2. 数据预处理
# 将词袋转换为gensim.corpora.textcorpus.TextCorpus对象
texts = data['Word'].apply(lambda x: x.split())
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 3. 创建LDA模型
num_topics = 7  # 设置主题数
alpha = 50 / num_topics  # 设置α参数
eta = 0.1  # 设置£参数
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=num_topics,
                                   alpha=alpha,
                                   eta=eta,
                                   passes=10,
                                   random_state=123)

# 4. 可视化主题
import pyLDAvis.gensim_models
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
print('LDA可视化结果')
print(vis)
print('LDA可视化结果存到xlsx中')
pd.DataFrame(vis.topic_info).to_excel("TF-IDF-topic_info.xlsx", index=False)
pd.DataFrame(vis.topic_coordinates).to_excel("TF-IDF-topic_coordinates.xlsx", index=False)
pd.DataFrame(vis.token_table).to_excel("TF-IDF-token_table.xlsx", index=False)


print('LDA可视化结果保存为html文件')
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print('LDA模型训练完成')

# 5. 构造 TF-IDF
tf_idf_vectorizer = TfidfVectorizer()
tf_idf = tf_idf_vectorizer.fit_transform(pd.DataFrame(texts))

# 将数据保存到TF-IDF.xlsx文件中
# pd.DataFrame(texts).to_excel("TF-IDF.xlsx", index=False)

print(pd.DataFrame(texts))
# 特征词列表
feature_names = tf_idf_vectorizer.get_feature_names_out()
# 特征词 TF-IDF 矩阵
print('开始构建TF-IDF矩阵')
matrix = tf_idf.toarray()
feature_names_df = pd.DataFrame(matrix,columns=feature_names)
print('TF-IDF矩阵构建结束')
print(feature_names_df)
feature_names_df