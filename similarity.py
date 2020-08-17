import json
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from konlpy.tag import Okt
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np


def deform_reviews():
    okt = Okt()

    path = 'json/review_file_deform.json'
    with open(path, 'r', encoding='utf-8') as file:
        json_dict = json.load(file)

    with open(path, 'w', encoding='utf-8') as save:
        json.dump(json_dict, save, indent='\t', ensure_ascii=False)


def deform_sentence(okt, sentence):
    pos = okt.pos(sentence, norm=True, stem=True)
    pos_str = ''
    for pos_elem in pos:
        pos_str += ' ' + pos_elem[0]
    return pos_str


def n_similarity(okt, tf_idf, s1, s2):
    v1 = tf_idf.transform([deform_sentence(okt, s1)]).toarray()[0]
    v2 = tf_idf.transform([deform_sentence(okt, s2)]).toarray()[0]

    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm

    return similarity


okt = Okt()

path = 'json/review_file_deform.json'
with open(path, 'r', encoding='utf-8') as file:
    json_dict = json.load(file)

cnt = 0
sentences = []

for prod_list in json_dict['list']:
    for review in prod_list['review_list']:
        if review['review'] == '완전 강추합니다.' or\
            review['review'] == '가격대비 괜찮네요~~' or\
            review['review'] == '잘 받았습니다. 많이 파세요^^' or\
            review['review'] == '잘받았습니다. 마음에 들어요.' or \
            review['review'].find('배송') > -1:
                continue
        sentences.append([review['tf-idf'], cnt, review['review']])
        cnt += 1

#tf_idf = TfidfVectorizer().fit(sentences)

#print('"좀 이상한데요?", "공 잘 나가네요. 확실히 구장에 있는 공용라켓 쓸때랑 너무 달라요ㅋㅋ" 두 문장 비교')
#print(n_similarity(okt, tf_idf, '좀 이상한데요?', '공 잘 나가네요. 확실히 구장에 있는 공용라켓 쓸때랑 너무 달라요ㅋㅋ'))


tagged_data = [TaggedDocument(words=s, tags=[uid]) for s, uid, r in sentences]

max_epochs = 10

model = Doc2Vec(
    window=10,
    vector_size=150,
    alpha=0.025,
    min_alpha=0.025,
    min_count=2,
    dm=1,
    negative=5,
    seed=9999)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha


model.random.seed(9999)

doc_list = deform_sentence(okt, '좀 이상한데요').split(' ')

inferred_vector = model.infer_vector(doc_list)
return_docs = model.docvecs.most_similar(positive=[inferred_vector], topn=100)
print('"좀 이상한데요"와 유사한 문장')
for doc in return_docs:
    print(sentences[doc[0]][2])
    print(doc[1])