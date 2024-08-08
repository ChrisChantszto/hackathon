from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

# define training data (still very small for illustration purposes)
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

# train model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

# fit a 2d PCA model to the vectors
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

# summarize the loaded model
print(f"Model vocabulary size: {len(model.wv)}")

# summarize vocabulary
# words = list(model.wv.key_to_index)
print(f"Vocabulary: {words}")

# access vector for one word
print(f"Vector for 'sentence': {model.wv['sentence']}")

# save model
model.save('model.bin')

# load model
new_model = Word2Vec.load('model.bin')
print(f"Loaded model vocabulary size: {len(new_model.wv)}")