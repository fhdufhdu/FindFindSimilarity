from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank
import json

with open('json/review_file_deform.json', 'r', encoding='utf-8') as file:
    json_dict = json.load(file)

sentences = []
for prod_list in json_dict['list']:
    for review in prod_list['review_list']:
        
        sentences.append(review['review'])

sentences = [normalize(text, english=True, number=True) for text in sentences]

wordrank_extractor = KRWordRank(
    min_count=5,                 # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length=10,               # 단어의 최대 길이
    verbose=True
    )

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10

keywords, rank, graph = wordrank_extractor.extract(sentences, beta, max_iter)

for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))

for prod_list in json_dict['list']:
    for review in prod_list['review_list']:
        rank = 0
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
            if review['review'].find(word) > -1:
                rank += r
        print(review['review']+':'+str(rank))