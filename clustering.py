from sklearn.cluster import DBSCAN
import sentence_transformers
import pickle
import os
import random

'''
clustering, trying to cluster sentences using k means.. for another project..
'''

sentences = [
    "I don't understand why people enjoy frozen desserts.",
    "I love analyzing film techniques and cinematography.",
    "The great indoors is where I feel most comfortable.",
    "I can't start my day without a strong cup of coffee.",
    "Ice cream is my favorite dessert.",
    "I prefer herbal tea over coffee any day.",
    "There's nothing like curling up with a good book.",
    "Camping under the stars is my idea of a perfect weekend.",
    "I'd rather have a warm dessert than ice cream any day.",
    "The aroma of freshly ground coffee beans is heavenly.",
    "I prefer staying in and watching TV to going outside.",
    "Nothing beats a scoop of creamy gelato on a hot day.",
    "I can't wait for the next blockbuster to hit theaters.",
    "Hiking in nature rejuvenates my soul.",
    "The smell of coffee makes me nauseous.",
    "I always have a novel in my bag for downtime.",
    "Ice cream is too cold and hurts my teeth.",
    "There's no better way to start the day than with a trail run.",
    "I love trying different coffee brewing methods.",
    "I don't understand the appeal of caffeine.",
    "Nothing beats a movie night with friends and popcorn.",
    "I can't resist a good sundae with all the toppings.",
    "The thought of eating ice cream makes me feel sick.",
    "Coffee is too bitter for my taste.",
    "I love the challenge of rock climbing.",
    "Why go out when you can have everything delivered?",
    "Frozen yogurt is a delicious alternative to traditional ice cream.",
    "The smell of old books in a library is intoxicating.",
    "I prefer reading the book before watching its movie adaptation.",
    "A good espresso is worth its weight in gold.",
    "Watching classic films is my favorite pastime.",
    "I'd rather play video games than engage in outdoor activities."
]

dump_file = open("dump.txt", "rb")
embeddings = []
model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

if(os.stat("dump.txt").st_size > 0):
    print("reading from cache")
    embeddings = pickle.loads(dump_file.read())
    dump_file.close()
else:
    dump_file.close()
    dump_file = open("dump.txt", "wb")
    for sentence in sentences:
        embedding = model.encode(sentence)
        embeddings.append(embedding)

    pe = pickle.dumps(embeddings)
    dump_file.write(pe)
    dump_file.close()

#get random sentence

r1 = random.randint(0, len(sentences) - 1)
r2 = random.randint(0, len(sentences) - 1)
s1 = embeddings[r1]
s2 = embeddings[r2]

s_test1 = model.encode("During the election campaign, I travelled extensively through the Jammu, Poonch and Rajouri districts. I found that, while the government’s decision to accord Scheduled Tribe status to Pahari-speaking people of all castes had turned public sentiment in the BJP’s favour in the two hill districts, there was widespread dissatisfaction with the party in its traditional stronghold of Jammu")
s_test2 = model.encode("The government's decision to grant Scheduled Tribe status to Pahari-speaking people of various castes sparked varied reactions across the districts of Jammu and Kashmir. While the move bolstered support for the BJP in the two hill districts, discontent with the party was widespread in its traditional stronghold of Jammu.")
similarity = model.similarity(s_test1, s_test2)

print(similarity)

#i want cosine since sentence embeddings
db = DBSCAN(eps = 0.5, min_samples = 4, metric="cosine")
clusters = db.fit(embeddings)

labels = db.labels_
n_clusters = 0
n_noise = 0
for label in labels:
    if label == -1:
        n_noise += 1
    else:
        n_clusters += 1

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

