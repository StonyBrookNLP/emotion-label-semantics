'''
Code taken from https://github.com/pytorch/examples/tree/master/word_language_model
'''
import os
import argparse
import pickle
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import hyperparameters_emo as hyp
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy import sparse
import itertools
from sklearn import metrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from random import shuffle
import random
import torch.nn.utils.rnn as rnn_utils
import progressbar
import codecs
import ast
import gc
import sys
import math
from random import randint
import numpy
import collections
import re
from numpy import array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from sklearn.metrics import hamming_loss
try:
    import cPickle as pickle
except:
    import pickle
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
tknzr = TweetTokenizer()

# import utils.model_utils as model_utils
# import utils.utils as utils
import pandas as pd
from evaluator import MLTEvaluator
# from pytorch_fast_elmo import FastElmoEmbedding, load_and_build_vocab2id, batch_to_word_ids
from allennlp.modules.elmo import Elmo, batch_to_ids



rand_num = 20
# print "The random seed is:"
# print rand_num
torch.backends.cudnn.deterministic = False
random.seed(rand_num)
torch.manual_seed(rand_num)
torch.cuda.manual_seed_all(rand_num)
np.random.seed(rand_num)

def create_emb_layer(obj, vocab_size, padding_idx, non_trainable=False):
    # print matrix
    # print vocab
    num_embeddings = vocab_size + 1   ##+1 to take into account the unk token
    embedding_dim = hyp.embedding_dim

    padding_idx = torch.cuda.LongTensor([padding_idx])
    # print padding_idx
    #initialize with vocabulary size in the training set and the dimension of each embedding vector
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    emb_dict_pre_train = dict() ##to get the embeddings before training

    train_vocab = obj.term2index

    for word, i in obj.term2index.items():

        if word in ["<start>", "<end>", "<pad>"]:
            emb_layer.weight.data[i].zero_()

            # print (type(emb_layer.weight.data[i].numpy()))
            emb_dict_pre_train[word] = emb_layer.weight.data[i].numpy()   

            continue

        elif word in glove_dict:
            vec = torch.cuda.FloatTensor(glove_dict[word])
            emb_layer.weight.data[i] = vec
            emb_dict_pre_train[word] = glove_dict[word]

        else:
            ##for words not present in the glove dictionary
            emb_layer.weight.data[i] = torch.tensor(torch.from_numpy(np.random.uniform(-0.2, 0.2, embedding_dim))).float().to(device)
            emb_dict_pre_train[word] = emb_layer.weight.data[i].numpy()   
    
    ##contains embeddings for both labels and vocab
    pickle.dump(emb_dict_pre_train, open("emb_before_train.p", "wb"))

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings

def create_emb_labels(obj, vocab_size, padding_idx, non_trainable=False):
    # print matrix
    # print vocab
    plutchik = ["joy","trust","fear","surprise","sad","disgust","anger","anticipation"] 
    
    num_embeddings = vocab_size
    embedding_dim = hyp.embedding_dim

    padding_idx = torch.cuda.LongTensor([padding_idx])

    #initialize with vocabulary size in the training set and the dimension of each embedding vector
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    emb_dict_pre_train = dict() ##to get the embeddings before training

    for word in plutchik:

        i = obj.term2index[word]

        vec = torch.cuda.FloatTensor(glove_dict[word])
        # print "Embedding:"
        # print vec
        # print
        emb_layer.weight.data[i] = vec

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings


## Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None

    for i in range(rnn_outputs.size(1)):
        h_i = rnn_outputs[:, i]
        a_i = att_weights[:, i]
        prod = a_i * h_i
        prod = prod.view(prod.size(0), 1, rnn_outputs.size(2))
        
        if(attn_vectors is None):
            attn_vectors = prod
        else:
            attn_vectors = torch.cat((attn_vectors,prod),1)

    return torch.sum(attn_vectors, 1)

class Model(torch.nn.Module):
    def __init__(self, obj, vocab_size, embedding_dim, elmo_embedding_dim, hidden_dim, num_layers):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.elmo_embedding_dim = elmo_embedding_dim

        #Get the pretrained glove embeddings
        self.embeddings1, _ = create_emb_layer(obj, vocab_size, 0, False)
        
        self.embeddings2, _ = create_emb_layer(obj, vocab_size, 0, False)

        self.RNN1 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=True, dropout = hyp.enc_dropout, batch_first=True)
        self.RNN2 = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=True, dropout = hyp.enc_dropout, batch_first=True)

        self.dropout = nn.Dropout(hyp.dropout)
        self.linear_sent_char = nn.Linear(4*self.hidden_dim, 8)

        # self.tanh = nn.Tanh()
# (batch_sentences, batch_contexts, sent_lens1, ctx_lens1, batch_labels, keys)
    def forward(self, input_sent, input_sent_ids, input_char, input_char_ids,sent_lens, ctx_lens, input_label, input_label_ids):

        ###For sentences############################################        
        ##sort instances by sequence length in descending order
        sent_lengths, sent_perm_idx = sent_lens.sort(0, descending=True)

        ##sort instances by sequence kength in descending order
        input_sent_ids = input_sent_ids[sent_perm_idx]

        ###For context#############################################
        ##sort instances by sequence length in descending order
        ctx_lengths, ctx_perm_idx = ctx_lens.sort(0, descending=True)

        ##sort instances by sequence kength in descending order
        input_char_ids = input_char_ids[ctx_perm_idx]

        ##get the embeddings for the sentence, character
        self.x_sent_ids = self.embeddings1(input_sent_ids)
        self.x_ctx_ids = self.embeddings2(input_char_ids)

        print (input_sent)
        print (input_char)
        ##call pack_padded_sequence with embedded instances ans sequence lengths
        packed_sent = rnn_utils.pack_padded_sequence(
            self.x_sent_ids,
            sent_lengths,
            batch_first=True)
        packed_ctx = rnn_utils.pack_padded_sequence(
            self.x_ctx_ids,
            ctx_lengths,
            batch_first=True)
        ##pass packed sequences in LSTM
        out_sent, (hn_sent, cn_sent) = self.RNN1(packed_sent)
        out_ctx, (hn_ctx, cn_ctx) = self.RNN2(packed_ctx)

        hn_sent = hn_sent.view(self.num_layers, 2, -1, self.hidden_dim)[1]

        encoder_sent = torch.cat((hn_sent[0], hn_sent[1]), 1)

        hn_ctx = hn_ctx.view(self.num_layers, 2, -1, self.hidden_dim)[1]
        
        encoder_ctx = torch.cat((hn_ctx[0], hn_ctx[1]), 1)

        print (encoder_sent.size())
        print (encoder_ctx.size())

        out_sent_ctx = torch.cat((encoder_sent, encoder_ctx), 1)

        ##added dropout between the 
        linear_h_e = self.linear_sent_char(self.dropout(out_sent_ctx))
        
        return linear_h_e

class Human_needs(object):
    def __init__(self):
        
        self.UNK = "<unk>"
        self.word2id = None
        self.embedding_matrix=[]
        self.term2index = None
        self.index2term = None

    def build_vocabs(self, data_train, data_dev, data_test, dim_embedding, embedding_path=None):
        
        #Convert the training data into a list 
        data_source = list(data_train) 

        #The class labels in the order they occur
        plutchik = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"] 
        
        #Get the different attributes of the training data
        id, sentences, context, label_distribution= zip(*data_source) 
        
        ##get unique tokens from the sentence
        wp_vocab = set(token for sent in sentences for token in sent.split(' '))

        ## get unique tokens from the context
        wp_vocab_context = set(token for sent in context for token in sent.split(' '))
    
        #form union from vocab of context and sentence
        wp_vocab = wp_vocab.union(wp_vocab_context)

        ##form union from vocab of label, contexts and sentences
        wp_vocab = wp_vocab.union(set(plutchik))

        ## for unk initialize with uniform random
        unk = numpy.random.uniform(-0.2, 0.2, dim_embedding)
        embeddings = {'UNK': unk}
        
        ##specify the path of pre-trained embedding file
        embed_file= codecs.open(embedding_path,encoding='utf-8',mode='r')
        
        ##Read the lines from the embeddings file
        lines = embed_file.readlines()
        pre_embed={} ##Dictionary to store the term and word embedding for the words in the vocabulary
        self.term2index={}   #Dictionary mapping the word to index
        count=0
        
        #to store the frequency of each of the words in the training data
        word_counter = collections.Counter()
        
        ##assign an integer id for each token in the sentence
        for word in sentences:
            for token in word.split(' '):
                w = token
                #Lowercase the token
                if hyp.lowercase == True:
                    w = w.lower()
                #Replace digits in the text
                if hyp.replace_digits == True:
                    w = re.sub(r'\d', '0', w)

                word_counter[w] += 1
                # self.term2index[w] = count
                count = count + 1

        ##assign an integer id for each token in the context
        for word in context:
            for token in word.split(' '):
                w = token
                if hyp.lowercase == True:
                    w = w.lower()
                if hyp.replace_digits == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                # self.term2index[w] = count
                count = count + 1

        for word in plutchik:
            w = word
            if hyp.lowercase == True:
                w = w.lower()
            if hyp.replace_digits == True:
                w = re.sub(r'\d', '0', w)
            word_counter[w] += 1
            # self.term2index[w] = count
            count = count + 1

        ##specify 0 id for the unk token
        self.word2id = collections.OrderedDict([(self.UNK, 0)])

        ##Create a mapping between word and its index
        for word, count in word_counter.most_common():
            if hyp.min_word_freq <= 0 or count >= hyp.min_word_freq:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        
        ##form the vocabulary for the embeddings file 
        if embedding_path != None and hyp.vocab_only_embedded == True:
            self.embedding_vocab = set([self.UNK])
            with codecs.open(embedding_path, encoding='utf-8',mode='r') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <= 2:
                        continue
                    ##Word is the first token in the line
                    w = line_parts[0]

                    if hyp.lowercase == True:
                        w = w.lower()
                    if hyp.replace_digits == True:
                        w = re.sub(r'\d', '0', w)
                    #Add word to the embedding vocabulary
                    self.embedding_vocab.add(w)
            
            ##add word in the word2id revised only if it occurs in the embedding vocabulary
            word2id_revised = collections.OrderedDict()

            for word in self.word2id:
                if (word in self.embedding_vocab) and (word not in word2id_revised):
                    word2id_revised[word] = len(word2id_revised)

                elif (word in wp_vocab) and (word not in word2id_revised):
                    word2id_revised[word] = len(word2id_revised)

            
            self.word2id = word2id_revised
        # Assign the unk token for words that occur in the test set but not in the training set
        self.word2id["<unk>"] = len(self.word2id)
        self.index2term={}
        self.term2index = self.word2id
                                                                                                                                                                                                                                                                                 
        ##get reverse of term to index mapping -- so from index to term
        self.index2term = {v:k for k,v in self.term2index.items()}
        # print("n_words: " + str(len(list(wp_vocab)))) 

    def translate2id(self, token, token2id, unk_token, lowercase=True, replace_digits=False):
    
        if lowercase == True:
            token = token.lower()
        if replace_digits == True:
            token = re.sub(r'\d', '0', token)
        token_id = None
        if token in token2id:
            token_id = token2id[token]
        elif unk_token != None:
            token_id = token2id[unk_token]
        else:
            raise ValueError("Unable to handle value, no UNK token: " + str(token))
        return token_id

    def _make_padding(self, sequences):
    #padding the training data
       padded = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=0, batch_first=True)
       return(padded)


    def extract_input(self,X,y,l):
        
        plutchik = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"] 
                
        sentence_lengths=[]
        max_length_count=[]
        sentence_list=[]
        max_1=0

        if l==0:
          for i in range(len(X)):
            #x = X[i].split(' ')
            ##tokenize sentences into tokens
            x = tknzr.tokenize(X[i])
            ## only non-empty tokens retained
            x = [k for k in x if k]
            ##get length of each sentence
            sentence_lengths.append(len(x))  
            
        elif l==1:
           max_1=0       
           for i in range(len(X)):
            if max_1<len(X[i]):
                max_1=len(X[i])
           all_lengths=[0]*max_1
           
           for i in range(len(X)):
              for j in range(len(X[i])):
                  #x = X[i][j].split(' ')
                  x = tknzr.tokenize(X[i][j])
                  x = [k for k in x if k]        
                  all_lengths[j]=len(x)
              sentence_lengths.append(all_lengths)
              all_lengths=[0]*max_1
        
        ##get max of the semtences in each batch
        max_sentence_length = max(sentence_lengths)
        sentence_classes = [[]]
        sentence_labels = numpy.zeros((len(X), len(plutchik)), dtype=numpy.float32)
        
        if l==0:
        ##specify the matrix to hold the padded id data
          word_ids = numpy.zeros((len(X),max_sentence_length), dtype=numpy.int32)
          sentence_list = [[' '] * max_sentence_length for i in range(len(X))]
          
          for i in range(len(X)):
            #x = X[i].split(' ')
            x = tknzr.tokenize(X[i])
            x = [k for k in x if k]
            count =0

            for j in range(len(x)): 
                ##get the actual sentence tokens for the batch
                 sentence_list[i][j]=x[j]
                ## get the token ids for the batch
                 word_ids[i][j] = self.translate2id(x[j], self.term2index, self.UNK, lowercase=hyp.lowercase, replace_digits=hyp.replace_digits)
                 count+=1
            ##get labels for batch
            a = y[i].strip()
            a = ast.literal_eval(a)
            pos = []
            pos = [i for i, j in enumerate(a) if j == 1]

            sentence_classes[i]= [0]*(len(plutchik))

            for l in pos:
                    # print (l)
                    # print ()
                    sentence_classes[i][l]=1
            if i<len(X)-1:
                sentence_classes.append([]) 
                
        return word_ids, sentence_lengths, sentence_classes, sentence_list



    def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
    
        max_length = 0
        max_lengths=[]
        sentence_max_lengths=[]
        # knowledge_max_lengths=[]
        context_max_lengths=[]
        sentences_pad = []
        # knowledge_pad = []
        context_pad = []
        ##get the input data
        id, sentences, context, label_distribution= zip(*batch)
        ## 
        word_ids, sentence_lengths, sentence_classes, sentence_tokens = self.extract_input(sentences, label_distribution,0)
        # print ("Batch sent ids:")
        # print (word_ids)
        # print (word_ids.shape)
        # word_ids_knowledge, knowledge_lengths, sentence_classes, knowledge_tokens = self.extract_input(knowledge,label_distribution,1)
        word_ids_context, context_lengths, sentence_classes, context_tokens = self.extract_input(context,label_distribution,0) 
        # print ("Batch ctx ids:")
        # print (word_ids_context)
        # print (word_ids_context.shape)

        
        if hyp.sentence_composition == "last" or hyp.sentence_composition == "attention" :
            # max_length_know = 0
            max_length_sent = 0
            max_length_context = 0
            ##max of sentence lengths in the batch -- for sentences
            max_length_sent = max(sentence_lengths)
            ## max of sentence lengths in the batch -- for contexts
            max_length_context = max(context_lengths)
                
            input_dictionary = {"word_ids": word_ids, "batch_size": len(sentences), "max_length_sent": max_length_sent, "max_length_context": max_length_context, "sentence_lengths": sentence_lengths, "sentence_labels": sentence_classes, "sentence_tokens": sentence_tokens, "context_tokens":context_tokens, "word_ids_context": word_ids_context, "context_lengths": context_lengths, "learningrate": learningrate, "is_training": is_training}#self.word_ids_knowledge: word_ids_knowledge,self.knowledge_lengths: knowledge_lengths,
                                    
        elif hyp.sentence_composition == "attention":
            input_dictionary = {"word_ids": word_ids, "batch_size": len(sentences), "max_length_sent": max_length_sent, "max_length_context": max_length_context, "sentence_lengths": sentence_lengths, "sentence_labels": sentence_classes, "sentence_tokens": sentence_tokens, "context_tokens":context_tokens, "word_ids_context": word_ids_context, "context_lengths": context_lengths, "learningrate": learningrate, "is_training": is_training}
        
        return input_dictionary

    def process_batch(self, data, batch, is_training, learningrate):
    
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
        # cost, sentence_scores = self.session.run([self.loss, self.sentence_scores] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:2]
        # token_scores=[]
        return feed_dict


    def get_words_batch(self, batch):
        batch_words = list()

        for i in range(batch.shape[0]):

            batch_words.append([self.index2term[id_] if id_ != 0 else id_ for id_ in batch[i, :]])

        return batch_words


def eval_class(model, eval_data):
    num = 0

    ##evaluation phase of the model
    model.eval()

    # Initialize scores to track
    plutchik = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]

    # Initialize answers to track
    start = time.time()

    random.shuffle(eval_data)

    id, sentences, context, eval_labels = zip(*eval_data) 

    ##Get weights of the different labels in the eval dataset
    eval_labels_processed = list()

    for i in range(len(eval_labels)):
        eval_labels_processed.append(ast.literal_eval(eval_labels[i].strip()))

    eval_labels_processed = np.asarray(eval_labels_processed)
    eval_labels_sum = np.sum(eval_labels_processed, axis=0)
    # print ("Shape of eval labels is:")
    # print (eval_labels_processed.shape)
    eval_labels_sum_norm = eval_labels_sum / eval_labels_processed.shape[0]

    # train_labels_sum_norm_ = 1.0 - train_labels_sum_norm
    eval_weights = 1.0 - np.exp(-1.0 * np.sqrt(eval_labels_sum_norm))
    # print ("Eval weights are:")
    # print (eval_weights)
    eval_weights = torch.tensor(torch.from_numpy(eval_weights)).float().to(device)

    all_batches = process_sentences(eval_data, sentences, model_data, is_training=False, learningrate=0.0001, name="dev", epoch=1)

    all_predicted = np.zeros((1, 8)) #zero array of one row and 8 columns
    all_true = np.zeros((1, 8)) #zero array of one row and 8 columns


    for m_batch in all_batches:

        # id_to_word_batch = model_data.get_words_batch(m_batch["word_ids"])
        batch_sentences = m_batch["sentence_tokens"]
        batch_sentences_ids = torch.tensor(torch.from_numpy(m_batch["word_ids"])).long().to(device)

        sent_lens = torch.tensor(torch.from_numpy(np.asarray(m_batch["sentence_lengths"], dtype=np.float32))).long().to(device)
                        
        batch_contexts = m_batch["context_tokens"]
        batch_contexts_ids = torch.tensor(torch.from_numpy(m_batch["word_ids_context"])).long().to(device)

        ctx_lens = torch.tensor(torch.from_numpy(np.asarray(m_batch["context_lengths"], dtype=np.float32))).long().to(device)
        
        batch_actual_scores = torch.tensor(torch.from_numpy(np.asarray(m_batch["sentence_labels"]))).float().to(device)
        print ("Actual scores are:")
        print (batch_actual_scores)
        print ("                             ")
        # ##padding sentences and contexts
        batch_sentences_ids = model_data._make_padding(batch_sentences_ids)
        batch_contexts_ids = model_data._make_padding(batch_contexts_ids)

        label_idx = []

        for label in plutchik:
            # print ("Check for label:")
            # print (model_data.term2index[label])
            label_idx.append(model_data.term2index[label])

        bs = batch_actual_scores.size(0)
        batch_labels = [plutchik for k in range(bs)]
        batch_labels_ids = torch.tensor([label_idx for k in range(bs)]).long().to(device)

        model.zero_grad()

        predicted_scores = model(batch_sentences, batch_sentences_ids, batch_contexts, batch_contexts_ids,sent_lens, ctx_lens, batch_labels, batch_labels_ids)

        loss = F.binary_cross_entropy_with_logits(predicted_scores, batch_actual_scores, weight=eval_weights)

        # print "Loss is:", loss
        ##input the inputs through a sigmoid
        prob_layer = nn.Sigmoid()

        prob = prob_layer(Variable(predicted_scores.data, volatile=True))

        all_predicted = np.concatenate((all_predicted, prob.data.round().cpu().numpy()), axis=0)
        all_true = np.concatenate((all_true, batch_actual_scores.cpu().numpy()), axis=0)
    
    print('Dev loss :%g'%(loss.data.item()))

    print ("Dev evaluation completed in: {} s".format(time.time() - start))

    micro_P, micro_R, micro_F1, macro_P, macro_R, macro_F1 = get_performance(all_true[1:, :], all_predicted[1:, :])
    print ("Dev data performance:")
    print ("micro_P", "micro_R", "micro_F1", "macro_P", "macro_R", "macro_F1")
    print (micro_P, micro_R, micro_F1, macro_P, macro_R, macro_F1)
    return micro_P, micro_R, micro_F1, macro_P, macro_R, macro_F1


def read_input_files(file_path, max_sentence_length=-1):
    """
    Reads input files in tab-separated format.
    Will split file_paths on comma, reading from multiple files.
    """
    sentences = []  #stores the list of sentences in the train / dev or test set
    labels = [] #stores the list of labels for each of the instances
    line_length = None #
    label_distribution=[] #Get the label distribution
    characters = [] #Get the characters in the batch
    knowledge_per = [] 
    context_s = []
    knowledge = []
    story_id_know=[] #List to store the story id already seen
    knowledge_final = []
    lst2=[]
    x=''
    length=[]
    max_length=0
    sub_path=[]
    id=[]
    w=[]
    weight=[]
    weight_per = []

    ##initialize - list of class labels (emotions)
    plutchik = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
    
    id2label_dict = dict([(idx, lab) for idx, lab in enumerate(plutchik)])


    with codecs.open(file_path,encoding="utf-8",mode="r") as f: 
        
        #For each row in the input data file
        for line in f:
            #Get the id of the story
            story_id = line.split('\t')[1]
            

            if story_id.strip(' ') in story_id_know:
                pos = story_id_know.index(story_id.strip(' '))

            #Get the context for the sentence in the story
            context = "[CLS] " + line.split('\t')[2].replace('|', " ") + " [SEP]" 
            # print (context)

            #Get the sentence in the story
            sent = "[CLS] " + line.split('\t')[3] + " [SEP]"
            # print (sent)
            #Get the character mentioned in the story
            char = line.split('\t')[4]
            # print (char)
            characters.append(char) ##save the character for who the event is occurring

            motiv = line.split('\t')[5]
            # print (motiv)
            # print ("motiv")
            # print (motiv)
            label = line.split('\t')[-1].strip()
            # print (label)

            lab = ast.literal_eval(label)

            pos = [i for i, j in enumerate(lab) if j == 1]
            path=[]
                           
            label_distribution.append(label)
            # sentences.append(char+'#'+sent)  #Add the name of the character as the prefix before the sentence
            sentences.append(sent)

            context_s.append(context)

            id.append(story_id)       

    #Batch of all the training / dev / test samples 
    batch = list(zip(sentences,context_s,label_distribution, characters))

    return id,sentences,context_s,label_distribution, characters

def padding(input, maxlen):
    """ 
    Padding the input sequence.....
    """
    sentences,context_s,label_distribution = zip(*input) 
    sentences = torch.nn.utils.rnn.pad_sequence.pad_sequence(list(sentences), padding_value=0)
    # knowledge = torch.nn.utils.rnn.pad_sequence.pad_sequence(knowledge, padding='post', truncating='post', maxlen=maxlen)
    context_s = torch.nn.utils.rnn.pad_sequence.pad_sequence(list(context_s), padding_value=0)

    return list(zip(sentences,context_s,label_distribution))

def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size == True:
        sentence_ids_by_length = collections.OrderedDict()
        sentence_length_sum = 0.0
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (max_batch_size > 0 and len(current_batch) >= max_batch_size) \
              or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids

def process_sentences(data_train, data, model_data, is_training, learningrate, name, epoch):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluator = MLTEvaluator()
    ##each id represents a combination of (id, sentences, context, label_distribution)
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, hyp.batch_equal_size, hyp.max_batch_size)
    
    all_batches = list()
    if is_training == True:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data_train[i] for i in sentence_ids_in_batch]
        
        # ##each batch contains a combination of (id, sentences, context, label_distribution)
        # print ("Batch is:")
        # print (batch)
        batch_data = model_data.process_batch(data_train, batch, is_training, learningrate)
        # print ("Returned batch is:")
        # print (batch_data)
        # print ()
        all_batches.append(batch_data)

    while hyp.garbage_collection == True and gc.collect() > 0:
            pass

    return all_batches


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")

    return model

def get_performance(true_labels, pred_labels):

    num_labels = 8
    tn_all = 0.0
    fp_all = 0.0
    fn_all = 0.0
    tp_all = 0.0
    p_all = 0.0
    r_all = 0.0
    f1_all = 0.0


    for i in range(num_labels):
        true_i = true_labels[:, i]
        pred_i = pred_labels[:, i]

        p, r, f1, support = precision_recall_fscore_support(true_i, pred_i, average="binary")
        print ("Per class performance:")
        print ("precision:", p, "recall:",r, "F1:", f1)
        p_all += float(p)
        r_all += float(r)
        f1_all += float(f1)

        tn, fp, fn, tp = confusion_matrix(true_i, pred_i).ravel()

        tn_all += float(tn)
        fp_all += float(fp)
        fn_all += float(fn)
        tp_all += float(tp)


    macro_p = p_all / float(num_labels)
    macro_r = r_all / float(num_labels)
    macro_f1 = f1_all / float(num_labels)
    micro_p = tp_all / (tp_all + fp_all)
    micro_r = tp_all / (tp_all + fn_all)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    return micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

# pickle.dump(glove_dict, open('../../data_for_code/glove.6B.100d.dat', 'wb'))
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)
def check_label_val(train_label, label):
    all_label = train_label

    # for sample_label in label_ids:
    unique, counts = np.unique(all_label, return_counts=True)
    
    for l in unique:
        if (l != 0) and (l != 1):
            print ("Found error")
            print ("All label")
            print (all_label)
            print ("        ")
            print ("Train label")
            print (label)

            # print ("        ")
            # print ("Dev label")
            # print (set(dev_label))
            print ("          ")
            # print (label_ids)
            print ("----------")

def run_experiment():    
    #initialize the train data, dev data and test data to None
    data_train, data_dev, data_test = None, None, None

    ##Emotion class labels
    class_labels = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]

    #Get the data for training
    if hyp.path_train != None and len(hyp.path_train) > 0:
        # id,sentences,context_s,label_distribution
        train_id, train_sentences,train_contexts,train_labels, train_characters = read_input_files(hyp.path_train, hyp.max_train_sent_length)
    #Get the data for dev
    if hyp.path_dev != None and len(hyp.path_dev) > 0:
        dev_id, dev_sentences,dev_contexts,dev_labels, dev_characters = read_input_files(hyp.path_dev)
    
    #Get the data for test
    if hyp.path_test != None and len(hyp.path_test) > 0:
        test_id, test_sentences,test_contexts,test_labels, test_characters = read_input_files(hyp.path_test)

    print("Word_embedding_size",hyp.word_embedding_size)
    
    print("Word_embedding_path",hyp.preload_vectors)

    #Class labels (prediction and true occurs in this order)
    plutchik = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
    ##add the sentence and context
    # Use // separator to separate between the context and the sentence
    train_text = [a + "//" + b for a,b in zip(train_sentences, train_contexts)]
    dev_text = [a + "//" + b for a,b in zip(dev_sentences, dev_contexts)]
    test_text = [a + "//" + b for a,b in zip(test_sentences, test_contexts)]

    print ("Combined texts:")
    print (train_text)

    train_labels_zip = list(zip(*train_labels))
    dev_labels_zip = list(zip(*dev_labels))
    test_labels_zip = list(zip(*test_labels))

    train_labels_0 = train_labels_zip[0]
    train_labels_1 = train_labels_zip[1]
    train_labels_2 = train_labels_zip[2]
    train_labels_3 = train_labels_zip[3]
    train_labels_4 = train_labels_zip[4]
    train_labels_5 = train_labels_zip[5]
    train_labels_6 = train_labels_zip[6]
    train_labels_7 = train_labels_zip[7]

    dev_labels_0 = dev_labels_zip[0]
    dev_labels_1 = dev_labels_zip[1]
    dev_labels_2 = dev_labels_zip[2]
    dev_labels_3 = dev_labels_zip[3]
    dev_labels_4 = dev_labels_zip[4]
    dev_labels_5 = dev_labels_zip[5]
    dev_labels_6 = dev_labels_zip[6]
    dev_labels_7 = dev_labels_zip[7]

    test_labels_0 = test_labels_zip[0]
    test_labels_1 = test_labels_zip[1]
    test_labels_2 = test_labels_zip[2]
    test_labels_3 = test_labels_zip[3]
    test_labels_4 = test_labels_zip[4]
    test_labels_5 = test_labels_zip[5]
    test_labels_6 = test_labels_zip[6]
    test_labels_7 = test_labels_zip[7]
    print (len(train_labels))
    print (len(dev_labels))
    print (len(test_labels))

    train_df = pd.DataFrame.from_dict({"train_id" : train_id + dev_id, "train_text": train_text + dev_text, "train_characters" : train_characters + dev_characters,"joy" : train_labels_0 + dev_labels_0,"trust" : train_labels_1 + dev_labels_1, "fear" : train_labels_2 + dev_labels_2,"surprise" : train_labels_3 + dev_labels_3,"sadness" : train_labels_4 + dev_labels_4,"disgust" : train_labels_5 + dev_labels_5,"anger" : train_labels_6 + dev_labels_6,"anticipation" : train_labels_7 + dev_labels_7})

    # check_label_val(train_labels_0, dev_labels_0)
    # check_label_val(train_labels_1, dev_labels_1)
    # check_label_val(train_labels_2, dev_labels_2)
    # check_label_val(train_labels_3, dev_labels_3)
    # check_label_val(train_labels_4, dev_labels_4)
    # check_label_val(train_labels_5, dev_labels_5)
    # check_label_val(train_labels_6, dev_labels_6)
    # check_label_val(train_labels_7, dev_labels_7)


    test_df = pd.DataFrame.from_dict({"test_id" : test_id, "test_text": test_text, "test_characters" : test_characters, "joy" : test_labels_0,"trust" : test_labels_1, "fear" : test_labels_2,"surprise" : test_labels_3,"sadness" : test_labels_4,"disgust" : test_labels_5,"anger" : test_labels_6,"anticipation" : test_labels_7})

    print (train_df)

    train_df.to_csv("/home/rgaonkar/context_home/rgaonkar/label_embeddings/code/Bert-Multi-Label-Text-Classification/pybert/dataset/train_char_label_df.csv", sep="\t", index=False)

    test_df.to_csv("/home/rgaonkar/context_home/rgaonkar/label_embeddings/code/Bert-Multi-Label-Text-Classification/pybert/dataset/test_char_label_df.csv", sep="\t", index=False)

def get_actual_scores(labels):
    #list to hold the label scores
    labels_batch_score = list()  #initialize
    for label_list in labels:
        labels_score = [0.0 for i in range(len(labels_unique))]  #initialize

        label_list = list(np.asarray(label_list))

        for i, label in enumerate(label_list):
            if (label == 0):
                pass
            else:
                labels_score[label_to_id[label]] = 1.0

        labels_batch_score.append(labels_score)

    return labels_batch_score


# remove the zero padding module
def zero_padding(X):
    #X is the data that is being passed
    #get the length of each sentence text in sentences / char context / labels
    X_lengths = [len(sentence) for sentence in X]

    #create an empty matrix with padding tokens
    pad_token = 0
    longest_sent = max(X_lengths)
    batch_size = len(X)
    padded_X = np.ones((batch_size, longest_sent)) * pad_token

    #copy over the actual sequences 
    for i, x_len in enumerate(X_lengths):
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
    
    return padded_X

####Remove the map_to_id module#############################################################
def map_to_id(labels):
    label_to_id = dict()    
    id_to_label = dict()

    for label in labels:
        try:
            label_to_id[label] 
        except Exception as e:
            label_to_id[label] = len(label_to_id)
    # print label_to_id

    for label in label_to_id:
        id_to_label[label_to_id[label]] = label

    return label_to_id, id_to_label

def map_to_word(obj):

    id_word = dict()

    for idx in id_to_label:
        id_word[idx] = obj.train_idx2word[id_to_label[idx]]

    return id_word

def get_predict_scores(scores, labels_unique):

    batch_predict_scores = list()

    for label_predict_line in scores:
        
        label_predict_new = [0 for i in range(len(labels_unique))]

        for i, score in enumerate(label_predict_line):

            label_id = labels_unique[i]

            label_predict_new[label_to_id[label_id]] = score

        batch_predict_scores.append(label_predict_new)

    return batch_predict_scores

def get_batch(data, indices):

    batch = [data[idx] for idx in indices]
    return batch

def get_class_weights(data_labels):
    ##initialize a dictionary to hold the class weights
    class_wt = dict()

    for row in data_labels:
        for label in row:
            if label != 0:

                try:
                    class_wt[label] += 1.0
                except Exception as e:
                    class_wt[label] = 1.0

    total = sum(class_wt.values())

    for label in class_wt:
        class_wt[label] = class_wt[label] / total

    ##get the labels weight vector
    class_wt_vec = [0.0 for i in range(len(labels_unique))]


    for label in class_wt:
        class_wt_vec[label_to_id[label]] = class_wt[label]
    return class_wt_vec


if torch.cuda.is_available():
    if not hyp.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if hyp.cuda else "cpu")

#######Load data##############################################

glove_dict = loadGloveModel("/home/rgaonkar/context_home/rgaonkar/label_embeddings/data/glove.6B.100d.txt")
model_data = Human_needs()
run_experiment()
train_batch_size = hyp.train_batch_size

#specify the number of layers in the network
num_layers = hyp.num_layers


