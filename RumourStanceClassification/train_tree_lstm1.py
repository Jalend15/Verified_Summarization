import torch

from treelstm import TreeLSTM, calculate_evaluation_orders, batch_tree_input, TreeDataset, convert_tree_to_tensors

from torch.utils.data import Dataset, IterableDataset, DataLoader

import os
import codecs
from sklearn.metrics import f1_score
import random
import numpy as np

seed_val = 12

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

tree_path = '/Users/jalend15/Downloads/Verified-Summarization-master/Parsed_finale_80/'
# tree_path = '/Users/jalend15/Downloads/jalend/Jal/'
# test_set = ['charliehebdo-all-rnr-threads.txt']
# test_set = ['germanwings-crash-all-rnr-threads.txt']


test_set = 	[
			'charliehebdo-all-rnr-threads.txt',
			#'ebola-essien-all-rnr-threads.txt',
			# 'germanwings-crash-all-rnr-threads.txt',
			# 'sydneysiege-all-rnr-threads.txt',
			#'gurlitt-all-rnr-threads.txt',
			#'prince-toronto-all-rnr-threads.txt',
			# 'ottawashooting-all-rnr-threads.txt',
			#'putinmissing-all-rnr-threads.txt',
			# 'ferguson-all-rnr-threads.txt'
			]

from random import shuffle

IN_FEATURES = 80
OUT_FEATURES = 2
NUM_ITERATIONS = 40
BATCH_SIZE = 50
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001

files = os.listdir(tree_path)
print(files)
for test_file in test_set:
    print('Training Set:', set(files) - {test_file})

    test_trees = []
    train_trees = []

    for filename in files:
        if(filename=='.DS_Store'):
            continue
        input_file = codecs.open(tree_path + filename, 'r', 'utf-8')

        tree_li = []
        pos_trees = []
        neg_trees = []

        for row in input_file:
            s = row.strip().split('\t')

            tweet_id = s[0]
            curr_tree = eval(s[1])
            # print(row)
            # curr_tensor, curr_label = convert_tree_to_tensors(curr_tree)
            try:
                curr_tensor, curr_label = convert_tree_to_tensors(curr_tree)
            except:
                # print('S')
                continue

            curr_tensor['tweet_id'] = tweet_id

            if curr_label == 1:
                pos_trees.append(curr_tensor)
            else:
                neg_trees.append(curr_tensor)

        input_file.close()

        if filename == test_file:
            tree_li = pos_trees + neg_trees
            test_trees = tree_li

        else:
            tree_li = pos_trees + neg_trees

            shuffle(tree_li)

            train_trees += tree_li

    model = TreeLSTM(IN_FEATURES, OUT_FEATURES, HIDDEN_UNITS).train()

    loss_function = torch.nn.CrossEntropyLoss()
    weights = [0.035, 0.135, 0.128, 0.075] # charlie
   # weights = [0.066,0.267,0.15,0.137]  # german
   #  weights = [0.0422, 0.114, 0.11,0.096]  #ottawa
   #  weights = [0.036, 0.106, 0.095, 0.0811]  #sydney

    class_weights = torch.FloatTensor(weights)
    print(class_weights)
    input = torch.tensor([[0.4,2,1,0.6]], dtype=torch.float)
    loss_function_s = torch.nn.CrossEntropyLoss(weight =class_weights,ignore_index=-1)
    # loss_function_s = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i in range(NUM_ITERATIONS):
        total_loss = 0

        optimizer.zero_grad()

        curr_tree_dataset = TreeDataset(train_trees)

        train_data_generator = DataLoader(
            curr_tree_dataset,
            collate_fn=batch_tree_input,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for tree_batch in train_data_generator:
            try:
                h, h_root, c = model(
                    tree_batch['f'],
                    tree_batch['node_order'],
                    tree_batch['adjacency_list'],
                    tree_batch['edge_order'],
                    tree_batch['root_node'],
                    tree_batch['root_label']
                )
            except:
                continue

            labels = tree_batch['l']
            stance_labels = (tree_batch['stance']).type(torch.long)
            root_labels = tree_batch['root_label']


            loss = loss_function(h_root, root_labels)
            loss_s = loss_function_s(h, stance_labels)
            tot_loss = loss + loss_s
            tot_loss.backward()

            optimizer.step()

            total_loss += tot_loss

        print(f'Iteration {i + 1} Loss: {total_loss}')


    print('Training Complete')

    print('Now Testing:', test_file)

    acc = 0
    total = 0
    acc_s = 0
    total_s = 0

    pred_label_li = []
    true_label_li = []
    pred_label_s_li = []
    true_label_s_li = []

    for test in test_trees:
        try:
            h_test, h_test_root, c = model(
                test['f'],
                test['node_order'],
                test['adjacency_list'],
                test['edge_order'],
                test['root_n'],
                test['root_l']
            )
        except:
            continue

        pred_v, pred_label = torch.max(h_test_root, 1)
        pred_stance_v, pred_label_s = torch.max(h_test, 1)

        true_label = test['root_l']
        true_label_s = (test['stance']).type(torch.long)

        if pred_label == true_label:
            acc += 1


        for true_stance, pred_stance in zip(true_label_s, pred_label_s):
            total_s += 1
            if true_stance == -1:
                total_s-=1
                continue
            if true_stance == pred_stance:
                acc_s += 1
            pred_label_s_li.append(pred_stance)
            true_label_s_li.append(true_stance)


        pred_label_li.append(pred_label)
        true_label_li.append(true_label)

        total += 1



    macro_f1 = f1_score(pred_label_li, true_label_li, average='macro')
    macro_f1_stance = f1_score(pred_label_s_li, true_label_s_li, average='macro')

    print("Rumour:-")
    print(test_file, 'accuracy:', acc / total)
    print(test_file, 'f1:', macro_f1)
    print("Stance:-")
    print(acc_s, total_s)
    print(test_file, 'accuracy:', acc_s / total_s)
    print(test_file, 'f1:', macro_f1_stance)
    print(test_file, 'total tested:', total)
