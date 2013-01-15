import os
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary


def preprocess(text):

        '''
        Takes a block of text, splits it into words, and lemmatizes the words.

        Splitting an arbitrary block of natural language text into words is
        typically more complicated than splitting on spaces, but for the sake
        of our toy example, it will suffice.
        '''

        stop = ['the', 'of', 'a', 'at']
        words = [word for word in text.lower().split() if word not in stop]
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]


def toy_corpus():

        for f in os.listdir('corpus'):

                yield (f,open(os.path.join('corpus', f)).read().strip())


if __name__ == '__main__':

        corpus_dir = 'corpus'
        corpus = []

        for f in os.listdir(corpus_dir):

                text = open(os.path.join(corpus_dir, f)).read().strip()
                processed = preprocess(text)

                print "Raw:", text
                print "Preprocessed:", processed
                print

                corpus.append(processed)

        dct = Dictionary(corpus)
        print "Dictionary before filtering rare words..."
        print dct.token2id
        print

        dct.filter_extremes(no_below=2)
        print "Dictionary after filtering rare words..."
        print dct.token2id
        print
