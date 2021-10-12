import codecs
import torch
import torch.nn as nn
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

alltypes = {
0: 'society or culture',
1: 'science or mathematics',
2: 'health',
3: 'education or reference',
4: 'computers or Internet',
5: 'sports',
6: 'business or finance',
7: 'entertainment or music',
8: 'family or relationships',
9: 'politics or government'}

# alltypes = [

# 'sadness',
# 'joy',
# 'anger',
# 'disgust',
# 'fear',
# 'surprise',
# 'shame',
# 'guilt',
# 'love'
# ]

class InputExample(object):
    def __init__(self, guid, text, label, topic):
        self.guid = guid
        self.text = text
        self.label = label
        self.topic = topic

class Inputfeatures(object):
    def __init__(self, text_padded_ids, text_length, topic_padded_ids, label_id):
        self.text_padded_ids = text_padded_ids
        self.text_length = text_length
        self.topic_padded_ids = topic_padded_ids
        self.label_id = label_id


def get_examples_topic_train(filename):
    line_co = 0
    exam_co = 0
    examples = []
    label_list = []

    '''first get all the seen types, 
       since we will only create positive and negative examples in seen types'''
    with codecs.open(filename, 'r', 'utf-8') as f:
        lines = f.readlines()
    seen_types = set()
    for line in lines:
        line = line.strip().split('\t')
        if len(line)==2: # label_id, domain, text
            type_index =  line[0].strip()
            seen_types.add(type_index)

    for line in lines:
        line = line.strip().split('\t')
        if len(line)==2: # label_id, text
            type_index = line[0].strip()
            for index, type_ in alltypes.items():
                if str(index) == str(type_index):
                    '''positive pair'''
                    guid = "train-"+str(exam_co)
                    text = line[1]
                    label = 'related' 
                    examples.append(InputExample(guid = guid, text = text, 
                                    label = label, topic = type_))
                    exam_co+=1

                elif str(index) in seen_types:
                    '''negative pair'''
                    guid = "train-"+str(exam_co)
                    text = line[1]
                    label = 'unrelated' 
                    examples.append(InputExample(guid = guid, text = text, 
                                    label = label, topic = type_))
                    exam_co+=1

        line_co+=1

    print('loaded size:', line_co)
    print('seen_types:', seen_types)
    return examples, seen_types

def get_examples_topic_test(filename, seen_types):
    with codecs.open(filename, 'r', 'utf-8') as f:
        lines = f.readlines()
    line_co = 0
    exam_co = 0
    examples=[]
    type_index = []
    seen_str_indicator = []

    for i in range(10):
        type_index.append(i)
        if i in seen_types:
            seen_str_indicator.append('seen')# this is for a seen type
        else:
            seen_str_indicator.append('unseen')

    gold_label_list = []
    for line in lines:
        line = line.strip().split('\t')
        if len(line)==2: # label_id, text
            real_topic = line[0].strip()
            gold_label_list.append(real_topic)
            for index, type_ in alltypes.items():
                if str(index) == str(real_topic):
                    '''positive pair'''
                    guid = "test-"+str(exam_co)
                    text = line[1]
                    label = 'related' 
                    examples.append(InputExample(guid = guid, text = text, 
                                    label = label, topic = type_))
                    exam_co+=1
                else:
                    '''negative pair'''
                    guid = "test-"+str(exam_co)
                    text = line[1]
                    label = 'unrelated' 
                    examples.append(InputExample(guid = guid, text = text, 
                                    label = label, topic = type_))
                    exam_co+=1
            line_co+=1

    print('loaded size:', line_co)
    return examples, gold_label_list, seen_str_indicator, type_index

def get_labels():
    """See base class."""
    return ["unrelated", "related"]

def truncate_seq(tokens, max_length):
    """Truncates a sequence in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        if len(tokens) <= max_length:
            break
        else:
            tokens.pop()

def create_map(train_examples, val_examples, label_list, min_word_freq = 1):
    """
    creates word label maps

    """
    label_map = {label : i for i, label in enumerate(label_list)}
    word_freq = Counter()
    for train_example, val_example in zip(train_examples, val_examples):
        word_freq.update(train_example.text.split(' '))
        word_freq.update(val_example.text.split(' '))
        word_freq.update(train_example.topic.split(' '))
        word_freq.update(val_example.topic.split(' '))
    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    word_map['<pad>'] = 0
    word_map['<unk>'] = len(word_map)
    
    return word_map, label_map


def create_input_features(examples, word_map, label_map, seq_max_length = 1014, topic_max_length = 3):
    '''
    Creates input features that will be used to create a PyTorch Dataset.
    '''
    input_features = []
    for example in examples:
        text = example.text.split(' ')
        topic = example.topic.split(' ')
        truncate_seq(text, seq_max_length)
        truncate_seq(topic, topic_max_length)
        text_ids = list(map(lambda w: word_map.get(w, word_map['<unk>']), text))
        topic_ids = list(map(lambda w: word_map.get(w, word_map['<unk>']), topic))

        # Zero-pad up to the sequence length.
        text_padding = [0] * (seq_max_length - len(text_ids))
        topic_padding = [0] * (topic_max_length - len(topic_ids))
        text_padded_ids = text_ids + text_padding
        topic_padded_ids = topic_ids + topic_padding

        assert len(text_padded_ids) == seq_max_length
        assert len(topic_padded_ids) == topic_max_length

        label_id = label_map[example.label]
        input_features.append(Inputfeatures(text_padded_ids = text_padded_ids, text_length = len(text_ids),
                             topic_padded_ids = topic_padded_ids, label_id = label_id))
    return input_features


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    :return:
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(word_map, emb_dim, expand_vocab=True):
    """
    Load pre-trained embeddings for words in the word map.

    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """

    Word2vecmodel = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True, limit = 400000)
    print('loading pretrained word embeddings successfully')

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = torch.FloatTensor(len(word_map), emb_dim)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for emb_word in Word2vecmodel.vocab:

        embedding = Word2vecmodel[emb_word] 

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
        embeddings = torch.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.size(0) == len(word_map)

    print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

    return embeddings, word_map, lm_vocab_size

def evaluate(pred_probs, eval_label_list, eval_seen_str_indicator, eval_type_index, seen_types):
    '''
    pred_probs: a list, the prob for relatedness
    eval_label_list: the gold type index; list length == lines in file.txt
    eval_seen_str_indicator: totally hypo size, seen or unseen
    eval_type_index:: total hypo size, the type in [0,...n]
    seen_types: a set of type indices
    '''
    pred_probs = list(pred_probs)
    total_hypo_size = len(eval_seen_str_indicator)
    total_premise_size = len(eval_label_list)
    print(len(eval_seen_str_indicator))
    print(len(eval_type_index))

    assert len(pred_probs) == total_premise_size*total_hypo_size
    assert len(eval_seen_str_indicator) == len(eval_type_index)

    seen_hit=0
    unseen_hit = 0
    seen_size = 0
    unseen_size = 0

    for i in range(total_premise_size):
        pred_probs_per_premise = pred_probs[i*total_hypo_size: (i+1)*total_hypo_size]

        max_prob = -100.0
        max_index = -1
        for j in range(total_hypo_size):
            if pred_probs_per_premise[j] > max_prob:
                max_prob = pred_probs_per_premise[j]
                max_index = j

        pred_type = eval_type_index[max_index]
        gold_type = eval_label_list[i]

        # print('pred_type:', pred_type, 'gold_type:', gold_type)
        if gold_type in seen_types:
            seen_size+=1
            if gold_type == pred_type:
                seen_hit+=1
        else:
            unseen_size+=1
            if gold_type == pred_type:
                unseen_hit+=1

    seen_acc = seen_hit/(1e-6+seen_size)
    unseen_acc = unseen_hit/(1e-6+unseen_size)

    return seen_acc, unseen_acc

