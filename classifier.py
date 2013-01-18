import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from math import sqrt
import gensim
from sklearn.svm import SVC
import os


def vec2dense(vec, num_terms):

    '''Convert from sparse gensim format to dense list of numbers'''
    return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])

if __name__ == '__main__':

    #Load in corpus, remove newlines, make strings lower-case
    docs = {}
    corpus_dir = 'corpus'
    for filename in os.listdir(corpus_dir):

        path = os.path.join(corpus_dir, filename)
        doc = open(path).read().strip().lower()
        docs[filename] = doc

    names = docs.keys()

    #Remove stopwords and split on spaces
    print "\n---Corpus with Stopwords Removed---"

    stop = ['the', 'of', 'a', 'at', 'is']
    preprocessed_docs = {}
    for name in names:

        text = docs[name].split()
        preprocessed = [word for word in text if word not in stop]
        preprocessed_docs[name] = preprocessed
        print filename, ":", preprocessed

    #Build the dictionary and filter out rare terms
    dct = gensim.corpora.Dictionary(preprocessed_docs.values())
    unfiltered = dct.token2id.keys()
    dct.filter_extremes(no_below=2)
    filtered = dct.token2id.keys()
    filtered_out = set(unfiltered) - set(filtered)

    print "\nThe following super common/rare words were filtered out..."
    print list(filtered_out), '\n'

    print "Vocabulary after filtering..."
    print dct.token2id.keys(), '\n'

    #Build Bag of Words Vectors out of preprocessed corpus
    print "---Bag of Words Corpus---"

    bow_docs = {}
    for name in names:

        sparse = dct.doc2bow(preprocessed_docs[name])
        bow_docs[name] = sparse
        dense = vec2dense(sparse, num_terms=len(dct))
        print name, ":", dense

    #Dimensionality reduction using LSI. Go from 6D to 2D.
    print "\n---LSI Model---"

    lsi_docs = {}
    num_topics = 2
    lsi_model = gensim.models.LsiModel(bow_docs.values(), num_topics=num_topics)
    for name in names:

        vec = bow_docs[name]
        sparse = lsi_model[vec]
        dense = vec2dense(sparse, num_topics)
        lsi_docs[name] = sparse
        print name, ':', dense

    #Normalize LSI vectors by setting each vector to unit length
    print "\n---Unit Vectorization---"

    unit_vecs = {}
    for name in names:

        vec = vec2dense(lsi_docs[name], num_topics)
        norm = sqrt(sum(num ** 2 for num in vec))
        unit_vec = [num / norm for num in vec]
        unit_vecs[name] = unit_vec
        print name, ':', unit_vec

    #Take cosine distances between docs and show best matches
    print "\n---Document Similarities---"

    index = gensim.similarities.MatrixSimilarity(lsi_docs.values())
    for i, name in enumerate(names):

        vec = lsi_docs[name]
        sims = index[vec]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        #Similarities are a list of tuples of the form (doc #, score)
        #In order to extract the doc # we take first value in the tuple
        #Doc # is stored in tuple as numpy format, must cast to int

        if int(sims[0][0]) != i:
            match = int(sims[0][0])
        else:
            match = int(sims[1][0])

        match = names[match]
        print name, "is most similar to...", match

    #We add classes to the mix by labelling dog1.txt and sandwich1.txt
    #We use these as our training set, and test on all documents.
    print "\n---Classification---"

    dog1 = unit_vecs['dog1.txt']
    sandwich1 = unit_vecs['sandwich1.txt']

    train = [dog1, sandwich1]

    # The label '1' represents the 'dog' category
    # The label '2' represents the 'sandwich' category

    label_to_name = dict([(1, 'dogs'), (2, 'sandwiches')])
    labels = [1, 2]
    classifier = SVC()
    classifier.fit(train, labels)

    for name in names:

        vec = unit_vecs[name]
        label = classifier.predict([vec])[0]
        cls = label_to_name[label]
        print name, 'is a document about', cls

    print '\n'
