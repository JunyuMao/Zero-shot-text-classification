import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import opts 
from utils import *
from model import *
from weight_init import weights_init
import copy
from torch.utils.data import (DataLoader, TensorDataset)
#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, model, optimizer, loss_function):
    model.train()
    tr_loss = 0
    tr_steps = 0
    for i, (train_text_ids, train_text_lengths, train_topic_ids, train_label_ids) in enumerate(train_dataloader):
        # Reduce batch's padded length to maximum in-batch sequence
        max_text_length = max(train_text_lengths.tolist())
        train_text_ids = train_text_ids[: ,:max_text_length].to(device)
        train_topic_ids = train_topic_ids.to(device)
        train_label_ids = train_label_ids.to(device)
        #calculate
        pred_probs = model(train_text_ids, train_topic_ids, train_text_lengths)
        loss = loss_function(pred_probs, train_label_ids.view(-1, 1).float())
        #back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        tr_steps += 1
        if i%1000 == 0 :
            print('average_train_loss_{0}: {1}'.format(i,tr_loss/tr_steps))

        
def validation(dev_dataloader, eval_label_list, eval_seen_str_indicator,
               eval_type_index, seen_types, model, optimizer, loss_function):
    model.eval()
    dev_loss = 0
    dev_steps = 0 
    preds = []   
    for i, (dev_text_ids, dev_text_lengths, dev_topic_ids, dev_label_ids) in enumerate(dev_dataloader):
        # Reduce batch's padded length to maximum in-batch sequence
        max_text_length = max(dev_text_lengths.tolist())
        dev_text_ids = dev_text_ids[: ,:max_text_length].to(device)
        dev_topic_ids = dev_topic_ids.to(device)
        dev_label_ids = dev_label_ids.to(device)
        #calculate
        pred_probs = model(dev_text_ids, dev_topic_ids, dev_text_lengths)
        loss = loss_function(pred_probs, dev_label_ids.view(-1, 1).float())
        dev_loss += loss.item()
        dev_steps += 1
        if len(preds) == 0:
            preds.append(pred_probs.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], pred_probs.detach().cpu().numpy(), axis=0) 
    preds = preds[0]
    print(preds)
    print(eval_label_list)
    seen_acc, unseen_acc = evaluate(preds, eval_label_list, eval_seen_str_indicator, eval_type_index, seen_types)
    print('seen_acc:{}\t''unseen_acc:{}\t'.format(seen_acc, unseen_acc))
    print('average_dev_loss: {}'.format(dev_loss/dev_steps))
    return seen_acc, unseen_acc
    
def main(opt):
    # fix random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_examples = None
    best_unseen_acc = -0.1
  
    #get train, validation and test sets.
    train_examples, seen_types = get_examples_topic_train(opt.train_file)
    val_examples, val_gold_label_list, \
    val_seen_str_indicator, val_type_index = get_examples_topic_test(opt.dev_file, seen_types)
    test_examples, test_gold_label_list, \
    test_seen_str_indicator, test_type_index = get_examples_topic_test(opt.test_file, seen_types)   
    #create map
    label_list = get_labels()
    word_map, label_map = create_map(train_examples, val_examples, label_list, opt.min_word_freq)
    #initialize 
    embeddings, word_map, lm_vocab_size = load_embeddings(word_map, opt.word_emb_dim, opt.expand_vocab)
    model = Arch3(len(word_map),opt.word_emb_dim).to(device)
    model.apply(weights_init)
    model.init_word_embeddings(embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
    model.fine_tune_word_embeddings(opt.fine_tune_word_embeddings)  # fine-tune
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr = opt.learning_rate)
    loss_function = nn.BCELoss().to(device)
    #create input features
    train_features = create_input_features(train_examples, word_map, label_map, 
                                                 opt.seq_max_length, opt.topic_max_length)
    val_features = create_input_features(val_examples, word_map, label_map, 
                                                 opt.seq_max_length, opt.topic_max_length)
    test_features = create_input_features(test_examples, word_map, label_map, 
                                                 opt.seq_max_length, opt.topic_max_length)                                                
    #create dataloader
    train_text_ids = torch.tensor([f.text_padded_ids for f in train_features], dtype=torch.long)
    train_text_lengths = torch.tensor([f.text_length for f in train_features], dtype=torch.long)
    train_topic_ids = torch.tensor([f.topic_padded_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    val_text_ids = torch.tensor([f.text_padded_ids for f in val_features], dtype=torch.long)
    val_text_lengths = torch.tensor([f.text_length for f in val_features], dtype=torch.long)
    val_topic_ids = torch.tensor([f.topic_padded_ids for f in val_features], dtype=torch.long)
    val_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long)   
    test_text_ids = torch.tensor([f.text_padded_ids for f in test_features], dtype=torch.long)
    test_text_lengths = torch.tensor([f.text_length for f in test_features], dtype=torch.long)
    test_topic_ids = torch.tensor([f.topic_padded_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long) 
    train_data = TensorDataset(train_text_ids, train_text_lengths, train_topic_ids, train_label_ids)
    val_data = TensorDataset(val_text_ids, val_text_lengths, val_topic_ids, val_label_ids)
    test_data = TensorDataset(test_text_ids, test_text_lengths, test_topic_ids, test_label_ids)
    train_dataloader = DataLoader(train_data, shuffle = True, batch_size = opt.train_batch_size)
    val_dataloader = DataLoader(val_data, shuffle = False, batch_size = opt.eval_batch_size)
    test_dataloader = DataLoader(test_data, shuffle = False, batch_size = opt.eval_batch_size)

    for epoch in range(opt.epochs):
        train(train_dataloader, model, optimizer, loss_function)
        val_seen_acc, val_unseen_acc = validation(val_dataloader, 
                                                  val_gold_label_list, 
                                                  val_seen_str_indicator, 
                                                  val_type_index, 
                                                  seen_types, model,
                                                  optimizer, loss_function)
        is_best = val_unseen_acc > best_unseen_acc
        best_unseen_acc = max(val_unseen_acc, best_unseen_acc)   
        if is_best:
            best_model = copy.deepcopy(model)
    print('the results of test set:\n')
    seen_acc, unseen_acc = validation(test_dataloader, 
                                      test_gold_label_list, 
                                      test_seen_str_indicator, 
                                      test_type_index, 
                                      seen_types, 
                                      best_model,
                                      optimizer, loss_function)

if __name__ == "__main__":
    opt = opts.parse_opt()
    main(opt)