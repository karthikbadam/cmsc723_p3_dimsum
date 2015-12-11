'''
    ss = supersense
'''

# Creating the dictionary with filenames and supersenses
lexnames = open('lexnames.txt').read().splitlines()

lex_dict = {}
for lex_val in lexnames:
    word_lex_list = lex_val.split('\t')
    lex_dict[word_lex_list[0]] = word_lex_list[1]

import json
import numpy as np

for category in ['noun']:

    f_labeled = open('labeled_' + category + '.txt', 'w')

    labeled_word_dict = {}

    # Format data of indexes as a matrix
    data = open('data_' + category + '.txt').read().splitlines()
    l = []
    for line in data:
        l.append(line.split(' ')[:3])

    index_matrix = np.array(l)
    index_list = list(index_matrix[:, 0])
    fileno_list = list(index_matrix[:, 1])

    # Find supersense for each word
    indexes = open('index_' + category + '.txt').read().splitlines()
    for val in indexes:
        data_list = val.split(' ')
        word = data_list[0]

        # Get index of the word
        p_cnt = data_list[3]
        index_i = data_list[3 + int(p_cnt) + 3]

        # Get supersense of the word
        file_no = fileno_list[index_list.index(index_i)]
        supersense = lex_dict[file_no]

        labeled_word_dict[word] = supersense
        print word + ' ' + supersense

    json.dump(labeled_word_dict, f_labeled)

    # f_labeled = open('labeled_' + category + '.txt', 'r')
    # print "***** FILE READ ****"
    # print f_labeled.read()
