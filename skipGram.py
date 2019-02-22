from __future__ import division
import argparse
import pandas as pd
import numpy as np
import re  # Used to split text into sentences
import string  # Used to get all punctuation
import pickle  # Used to save and load embeddings
import logging  # Used to save steps in a text file instead of printing them
import tqdm  # Used to time the training
from scipy.special import expit  # Used to compute the gradient


__authors__ = ['Driss Debbagh-Nour','Mehdi Mikou','Soufiane Hadji', 'Mohamed Aymane Benayada']
__emails__  = ['driss.debbagh-nour@student.ecp.fr','mehdi.mikou@student.ecp.fr',
               'soufiane.hadji@student.ecp.fr', "mohamed-aymane.benayada@student.ecp.fr"]


logging.basicConfig(filename='test_log.log',level=logging.INFO,\
      format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


def concat_text(file):
    '''
    Used for documents which contain a long text rather than multiple sentences.
    The functions concatenate the whole text and removes quotes
    '''
    all_text_list = []
    with open(file) as f:
        for l in f:
            l = l[:-1] #We remove the line jumping
            all_text_list.append(l)
    concat_text = ''.join(all_text_list)
    concat_text = concat_text.replace('"', '')
    concat_text = concat_text.replace("\'", '')
    
    return concat_text


def text2sentences(file, only_sentences=True):
    """
    Split words while removing all punctuation, stopwords and non-alpha words while keeping contractions together
    2 types of input: 
    - If it's a whole text: Split text into sentences using punctuation (. ? ! ;) after concatenating all text
    - If it's a bunch of sentences: Split the document using \n 
    Then split sentences by whitespace
    
    Returns the tokenization of our file
    """
    
    final_sentences = []
    
    # Load stop words
    #stop_words = stopwords.words('english')
    #stop_words.append("")
    
    # Differentiate the two types of documents
    if not only_sentences:
        concatenated_file = concat_text(file)
        sentences = re.split(r'[.?!;]', concatenated_file)
    else:
        sentences = [line.rstrip('\n') for line in open(file, encoding="utf8")]
    
    # Tokenize our document
    for sentence in sentences:
        words_list = sentence.lower().split()
        ponctuation_remover = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(ponctuation_remover) for w in words_list 
                    if w not in string.punctuation]
        final_sentence = [word for word in stripped if word.isalpha()]
        final_sentences.append(final_sentence)

    return final_sentences


def count_words(list_words):
    """
    Take a list of words and return a dictionary with words as keys and their occurence as values
    """
    dict_occurences = {}
    for word in list_words:
        try:
            dict_occurences[word] += 1
        except KeyError:
            dict_occurences[word] = 1
    return dict_occurences
            
    
def rare_word_pruning(sentences, min_count):
    """
    Remove words that occures less than min_count time
    """
    words = [word for sentence in sentences for word in sentence]
    dict_occurences = count_words(words)
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if dict_occurences[word] >= min_count:
                new_sentence.append(word)
        new_sentences.append(new_sentence)
    return new_sentences

        
# def high_and_low_frequency_pruning(sentences, min_count, max_count_ratio):
#     """
#     Remove words that occures less than min_count and more than max_count times
#     """
#     words = [word for sentence in sentences for word in sentence]
#     max_count = int(len(words) * max_count_ratio)
#     dict_occurences = count_words(words)
#     new_sentences = []
#     for sentence in sentences:
#         new_sentence = []
#         for word in sentence:
#             if min_count<= dict_occurences[word] <= max_count:
#                 new_sentence.append(word)
#         new_sentences.append(new_sentence)
#     return new_sentences


def get_positive_pairs(processed_sentences, winSize):
    """
    Get Pairs of words that co-occur (in the window delimited by winSize): Positive examples
    """
    # List of positive pairs to return
    positive_pairs = []
    
    # Initialize dictionaries that will allow to move from strings to their indexes
    words_voc = {}  
    context_voc = {}
    
    # Initialize indexes for words and contexts
    indexer_words = 0
    indexer_context = 0
    
    for sentence in processed_sentences:
        for i, word in enumerate(sentence):
            if word not in words_voc:
                words_voc[word] = indexer_words
                indexer_words += 1
            
            for j in range(max(0, i - winSize//2), min(i + winSize//2 + 1, len(sentence))):  # Be careful of edges
                if i != j:  # word != context
                    context = sentence[j]
                    if context not in context_voc:
                        context_voc[context] = indexer_context
                        indexer_context += 1
                    positive_pairs.append((words_voc[word], context_voc[context]))
                    
    return positive_pairs, words_voc, context_voc


def get_negative_pairs(positive_pairs, negativeRate):
    """
    Get Pairs of words that don't co-occur: Negative examples 
    Size: negativeRate * size(positive_pairs)
    """
    # List of negative pairs to return
    negative_pairs = []
    nbr_positive_pairs = len(positive_pairs)
    
    for p_pair in positive_pairs:
        word_index = p_pair[0]  # target word
        for _ in range(negativeRate):
            pair_index = np.random.randint(nbr_positive_pairs)  # Random index for a pair (can be improved)
            negative_context = positive_pairs[pair_index][1]  # Get the random pair context index
            negative_pairs.append((word_index, negative_context))
            
    return negative_pairs


def gradient(theta, nEmbed, positive_pairs, negative_pairs, nb_words, nb_contexts):
    """
    Compute gradient at init_theta for positive and negative pairs
    """
    # Initialize gradient
    grad = np.zeros(len(theta))
    
    # Embedding matrix of target words
    words_matrix = theta[: nEmbed * nb_words].reshape(nb_words, nEmbed)
    
    # Embedding matrix of contextes
    contexts_matrix = theta[nEmbed * nb_words:].reshape(nb_contexts, nEmbed)
    
    logging.info("Compute gradient")
    
    # Positive pairs
    logging.info("Positive pairs...")
    for p_pair in positive_pairs:
        
        # Get indexes of (word, context)
        word_index = p_pair[0]
        context_index = p_pair[1]
        
        # Get the actual embedding of the word and its context
        word = words_matrix[word_index]
        context = contexts_matrix[context_index]

        # We compute the derivative of the formula given by 'Yoav Goldberg' and 'Omer Levy'
        df_word = context * expit(-word.dot(context))
        df_context = word * expit(-word.dot(context))
        
        # We actualize the gradient of the word and its context
        grad[word_index * nEmbed: (word_index + 1) * nEmbed] += df_word
        grad[(nb_words + context_index) * nEmbed: (nb_words + context_index + 1) * nEmbed] += df_context
    logging.info("Done")
    
    # Negative pairs
    logging.info("Negative pairs...")
    for n_pair in negative_pairs:
        
        # Get indexes of (word, negative context)
        word_index = n_pair[0]
        context_index = n_pair[1]
        
        # Get the actual embedding of the word and its context
        word = words_matrix[word_index]
        context = contexts_matrix[context_index]
        
        # We compute the derivative of the formula given by 'Yoav Goldberg' and 'Omer Levy'
        df_word = -context * expit(word.dot(context))
        df_context = -word * expit(word.dot(context))
        
        # We actualize the gradient of the word and its negative context
        grad[word_index * nEmbed: (word_index + 1) * nEmbed] += df_word
        grad[(nb_words + context_index) * nEmbed: (nb_words + context_index + 1) * nEmbed] += df_context
    logging.info("Done")
    
    return grad

        
def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=7, minCount=5):
        # Preprocessing the sentences
        # Remove rare words
        print("1. Processing sentences...")
        processed_sentences= rare_word_pruning(sentences, minCount)
        print("Done\n")
        
        # Generate positive and negative pairs
        print("2. Generating positive pairs...")
        self.positive_pairs, self.words_voc, self.context_voc = get_positive_pairs(processed_sentences, winSize)
        print("Done\n")
        
        print("3. Generating negative samples...")
        # Generate negative samples
        self.negative_pairs = get_negative_pairs(self.positive_pairs, negativeRate)
        print("Done")

    def train(self, learning_rate=0.01, epochs=5, batchsize=500, nEmbed=100, negativeRate=5):
        """Create W matrix containing the embeddings of all words"""
        
        # Get nb of words, nb of contexts, nb of pairs
        nb_words = len(list(self.words_voc.keys()))
        nb_contexts = len(list(self.context_voc.keys()))
        nb_pairs = len(self.positive_pairs)
  
        # Initialize theta: vector of parameters
        nb_param = nEmbed * (nb_words + nb_contexts)  # Number of parameters
        theta = np.random.random(nb_param) * 1e-5

        # Compute Stochastic Gradient
        print("TRAINING: epochs: {}, learning_rate: {}, batch size: {}".format(epochs, learning_rate, batchsize))
        logging.info("TRAINING: epochs: {}, learning_rate: {}, batch size: {}".format(epochs, learning_rate, batchsize))
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            logging.info("Epoch {}/{}".format(epoch+1, epochs))
            
            # We update theta after computing the gradient of each batch (which size is batchsize)
            for batch_number in tqdm.tqdm(range(nb_pairs // batchsize)):
                logging.info("batch_number {}/{}".format(batch_number+1, nb_pairs // batchsize))
                batch_begin = batch_number * batchsize
                batch_end = min((batch_number + 1) * batchsize, nb_pairs)
                batch_positive = self.positive_pairs[batch_begin: batch_end]
                batch_negative = self.negative_pairs[negativeRate * batch_begin: negativeRate * batch_end]
                
                # Compute the gradient at theta
                grad = gradient(theta, nEmbed, batch_positive, batch_negative, nb_words, nb_contexts)
                
                # Actualize theta (since we want to maximize the 'loss', we add grad)
                theta = theta + learning_rate*grad
                
            logging.info(theta)

        self.theta = theta
        
        # Matrix of embeddings
        self.W = theta[:nEmbed * nb_words].reshape(nb_words, nEmbed)

        
    def save(self, path):
        """Save in binary Theta and W"""
        with open(path + '\\theta', 'wb') as fichier:
            mon_pickler = pickle.Pickler(fichier)
            mon_pickler.dump(self.theta)
        with open(path + '\\W', 'wb') as fichier:
            mon_pickler = pickle.Pickler(fichier)
            mon_pickler.dump(self.W)        
        with open(path + '\\words_voc', 'wb') as fichier:
            mon_pickler = pickle.Pickler(fichier)
            mon_pickler.dump(self.words_voc)
            
    @staticmethod  
    def similarity(word1, word2, words_voc, W):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1: 1st word
        :param word2: 2nd word
        :param words_voc: dictionary of words kept from original document as keys and their index as value 
        :param W: matrix of embeddings
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        nEmbed = W.shape[1]
        
        # For words that aren't in the training set, we considere them as the same "OOV" 
        # and give them the same vector with low values
        default_embd = np.ones(nEmbed) * 0.01
        
        # Get word_1 vector
        if word1 in words_voc:
            idx_word1 = words_voc[word1]
            embd_word1 = W[idx_word1]
        else:
            print("Out of Vocabulary:", str(word1))
            embd_word1 = default_embd
        
        # Get word_2 vector
        if word2 in words_voc:
            idx_word2 = words_voc[word2]
            embd_word2 = W[idx_word2]
        else:
            print("Out of Vocabulary:", str(word2))
            embd_word2 = default_embd
        
        # Compute the cosine distance
        return abs(embd_word1.dot(embd_word2) / (np.linalg.norm(embd_word1) * np.linalg.norm(embd_word2)))

		
    def K_most_similar(self, word, K, words_voc, W):
        dict_words_similarity = {}
        for elt in words_voc:
            dict_words_similarity[elt] = self.similarity(word, elt, words_voc, W)
            
        ranked_similar_words = sorted(dict_words_similarity, key=dict_words_similarity.get, reverse=True)
        print("Similar words for", word, ":")
        for i in range(1, K + 1):
            print("-", ranked_similar_words[i], ':', dict_words_similarity[ranked_similar_words[i]])
    
    
    @staticmethod
    def load(path):
        with open(path + '\\W', 'rb') as fichier:
            my_depickler = pickle.Unpickler(fichier)
            W = my_depickler.load()
        with open(path + '\\words_voc', 'rb') as fichier:
            my_depickler = pickle.Unpickler(fichier)
            words_voc = my_depickler.load()
                  
        return W, words_voc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help="Path to the training data set", required=True)
    parser.add_argument('--model', help='path to embedding (W and words_voc)', required=True)
    parser.add_argument('--test', help='', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        W, words_voc = SkipGram.load(opts.model)
        
        for a,b,_ in pairs:
            print(SkipGram.similarity(a, b, words_voc, W))

