from preprocessor import toy_corpus, preprocess
from math import sqrt
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity
from gensim.models import LsiModel
from scipy.spatial.distance import cosine
import numpy
from sklearn.svm import SVC
from os import linesep as nl

if __name__ == '__main__':

        print "**********************",nl,nl
        #Load in corpus and preprocess as we did in preprocessor.py
        docs = dict((name,preprocess(doc)) for name,doc in toy_corpus())

        #Display the initial preprocessed text corpus
        print "---Initial Preprocessed Corpus---"
        for filename in docs:
                print filename,":",docs[filename]
        print

        #Build the dictionary and filter out rare terms
        dct = Dictionary(docs.values())
        
        unfiltered = dct.token2id.keys()
        dct.filter_extremes(no_below=2)
        filtered = dct.token2id.keys()

        filtered_out = set(unfiltered) - set(filtered)
        print "The following super common/rare words were filtered out...", list(filtered_out)
        vocab_string = '[' + ', '.join(dct.token2id.keys()) + ']'
        print "Vocabulary after filtering: ", vocab_string, nl, nl

        print "---BoW Corpus---"
        for file_name in docs.keys():

                #Express our docs as BoW vectors
                sparse = dct.doc2bow(docs[file_name])
                docs[file_name] = sparse 
                sparse = dict(sparse)

                #Converting from gensim format to more familiar dense format
                dense = [sparse[i] if i in sparse.keys() else 0 
                                        for i in range(len(dct))]
                print file_name, ":", dense

        print nl
        names = docs.keys()
        vecs = [docs[key] for key in names]
        lsi_vecs = []

        print "---LSI Model---"
        num_topics = 2
        lsi_model = LsiModel(vecs, num_topics=num_topics)
        for i in range(len(names)):

                name = names[i]
                vec = docs[name]

                sparse = dict(lsi_model[vec])
                dense = [sparse[i] if i in sparse.keys() else 0
                                        for i in range(num_topics)]
                print name, ':', dense
                lsi_vecs.append(dense)

        print nl

        print "---Unit Vectorization---"
        for i in range(len(lsi_vecs)):
                
                norm = sqrt(sum(num**2 for num in lsi_vecs[i]))
                unit_vec = [num/norm for num in lsi_vecs[i]]
                print names[i],':',unit_vec
                lsi_vecs[i] = unit_vec

        print nl
        print "---Document Similarities---"
        
        index = MatrixSimilarity(lsi_model[vecs])
        for i in range(len(names)):
                
                name = names[i]
                vec = lsi_model[docs[name]]

                sims = index[vec]
                sims = sorted(enumerate(sims), key=lambda item: -item[1])

                #Similarities are a list of tuples of the form (doc #, score)
                #In order to extract the doc # we take the first value in the tuple

                #Doc # is stored in tuple as numpy format, must cast to int
                match = int(sims[0][0]) if int(sims[0][0]) != i else int(sims[1][0])
                match = names[match]
                print name, "is most similar to...", match

        print nl
        print "---Classification---"

        dog1 = lsi_vecs[names.index('dog1.txt')]
        sandwich1 = lsi_vecs[names.index('sandwich1.txt')]
       
        train = [dog1, sandwich1]

        # The label '1' represents the 'dog' category
        # The label '2' represents the 'sandwich' category

        label_to_name = dict([(1,'dogs'),(2, 'sandwiches')])
        labels = [1,2]
        classifier = SVC()
        classifier.fit(train,labels)

        for i in range(len(names)):

                name = names[i]
                vec = lsi_vecs[i]

                label = classifier.predict([vec])[0]
                cls = label_to_name[label]
                print name,'is a document about', cls



