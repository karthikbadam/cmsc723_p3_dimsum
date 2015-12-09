import pyvw

valid_labels = {'n.act': 0,
'n.animal': 1,
'n.artifact': 2,
'n.attribute': 3,
'n.body': 4,
'n.cognition': 5,
'n.communication': 6,
'n.event': 7,
'n.feeling': 8,
'n.food': 9,
'n.group': 10,
'n.location': 11,
'n.motive': 12,
'n.natural_object': 13,
'n.other': 14,
'n.person': 15,
'n.phenomenon': 16,
'n.plant': 17,
'n.possession': 18,
'n.process': 19,
'n.quantity': 20,
'n.relation': 21,
'n.shape': 22,
'n.state': 23,
'n.substance': 24,
'n.time': 25,
'v.body': 26,
'v.change': 27,
'v.cognition': 28,
'v.communication': 29,
'v.competition': 30,
'v.consumption': 31,
'v.contact': 32,
'v.creation': 33,
'v.emotion': 34,
'v.motion': 35,
'v.perception': 36,
'v.possession': 37,
'v.social': 38,
'v.stative': 39,
'v.weather': 40}
valid_labels_rev = { v:k for k,v in valid_labels.iteritems() }

class BIO:
    # construct a BIO object using a bio type ('O', 'B' or 'I') and a
    # optionally a label (that can be used to capture the supersense tag).
    # this additionally computes a numeric_label to be used by vw
    def __init__(self, bio, label=None):
        if bio != 'O' and bio != 'B' and bio != 'I':
            raise TypeError
        self.bio = bio
        self.label = label   # the label will only be needed for supersenses
        self.numeric_label_bio = 1
        self.numeric_label_supersense= None
        if self.bio == 'B':
            self.numeric_label_bio = 2
        elif self.bio == 'I':
            self.numeric_label_bio = 3

        if label != None and len(label) > 2:
            self.numeric_label_supersense = valid_labels[label]

    # a.can_follow(b) returns true if:
    #    a is O and b is I or O or
    #    a is B and b is I or O or
    #    ...
    def can_follow(self, prev):
        return (self.bio == 'O' and (prev.bio == 'I' or prev.bio == 'O') ) or \
               (self.bio == 'B' and (prev.bio == 'I' or prev.bio == 'O') ) or \
               (self.bio == 'I' and (prev.bio == 'B' or prev.bio == 'I') )

    # given a label, produce a list of all valid BIO items that can
    # come next.
    def valid_next(self):
        valid = [] #TODO
        for next_label in ['B', 'I', 'O']:
            next_BIO_class= BIO(next_label)
            if next_BIO_class.can_follow(self):
                valid.append(next_BIO_class)
        return valid


    # produce a human-readable string
    def __str__( self): return self.bio #return 'O' if self.bio == 'O' else (self.bio + '-' + self.label)
    def __repr__(self): return self.__str__()

    # compute equality
    def __eq__(self, other):
        if not isinstance(other, BIO): return False
        return self.bio == other.bio and self.label == other.label
    def __ne__(self, other): return not self.__eq__(other)

# convert a numerical prediction back to a BIO label
def numeric_label_to_BIO(num_bio,num_supersense=None):
    if not isinstance(num_bio, int):
        raise TypeError
    if num_supersense !=None:
        if not isinstance(num_supersense, int):
            raise TypeError
    if num_bio == 1:
        if num_supersense!=None:
            label= valid_labels_rev[num_supersense]
            return BIO('O', label)
        else:
            return BIO('O')
    elif num_bio == 2:
        if num_supersense!=None:
            label= valid_labels_rev[num_supersense]
            return BIO('B', label)
        else:
            return BIO('B')
    elif num_bio == 3:
        if num_supersense!=None:
            label= valid_labels_rev[num_supersense]
            return BIO('I', label)
        else:
            return BIO('I')

# given a previous PREDICTED label (prev), which may be incorrect; and
# the current TRUE label (truth), generate a list of valid reference
# actions. the return type should be [BIO]. for example, if the truth
# is O or B, then regardless of what prev is the correct thing to do
# is [truth]. the most important thing is to handle the case when, for
# instance, truth is I but prev is neither I nor B
def compute_reference(prev, truth):
    if truth.bio=='I' :
        if truth not in prev.valid_next():
            truth= BIO('O', prev.label)

    return [ truth ]  # TODO


class MWE(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        # you must must must initialize the parent class
        # this will automatically store self.sch <- sch, self.vw <- vw
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

        # for now we will use AUTO_HAMMING_LOSS; in Part II, you should remove this and implement a more task-focused loss
        # like one-minus-F-measure.
        sch.set_options( sch.AUTO_CONDITION_FEATURES )

    def _run(self, sentence):
        output = []
        prev = BIO('O')   # store the previous prediction
        #loss=0.
        labels_str={}
        true_positives={}
        false_positives={}
        false_negatives={}
        for n in range(len(sentence)):
            # label is a BIO, word is a string and pos is a string
            if n>=1:
                plabel,pword,plemma,ppos = sentence[n-1]
            else:
                pword,plemma,ppos = "ROOTW", "ROOTL", "ROOTP"

            if n>=2:
                pplabel,ppword,pplemma,pppos = sentence[n-2]
            else:
                ppword,pplemma,pppos = "ROOTW", "ROOTL", "ROOTP"

            if n<len(sentence)-1:
                nlabel,nword,nlemma,npos = sentence[n+1]
            else:
                nword,nlemma,npos = "ENDW", "ENDL", "ENDP"

            if n<len(sentence)-2:
                nnlabel,nnword,nnlemma,nnpos = sentence[n+2]
            else:
                nnword,nnlemma,nnpos = "ENDW", "ENDL", "ENDP"

            label,word,lemma,pos = sentence[n]

            with self.make_example(word, lemma, pos, pword, plemma, ppos, nword, nlemma, npos, ppword, pplemma, pppos, nnword, nnlemma, nnpos) as ex:  # construct the VW example

                # first, compute the numeric labels for all valid reference actions

                refs_bio  = [ bio.numeric_label_bio for bio in compute_reference(prev, label) ]

                # next, because some actions are invalid based on the
                # previous decision, we need to compute a list of
                # valid actions available at this point
                valid = [ bio.numeric_label_bio for bio in prev.valid_next() ]

                pred_bio  = self.sch.predict(examples   = ex,
                                         my_tag     = n+1,
                                         oracle     = refs_bio,
                                         condition  = [(n, 'p'), (n-1, 'q')],
                                         allowed    = valid)

                refs_supersense  = [ bio.numeric_label_supersense for bio in compute_reference(prev, label) ]

                if not all(p is None for p in refs_supersense):
                    pred_supersense  = self.sch.predict(examples   = ex,
                                            my_tag     = n+1,
                                            oracle     = refs_supersense,
                                            condition  = [(n, 'p'), (n-1, 'q')])

                if pred_bio not in refs_bio:
                    if pred_bio not in false_negatives.keys():
                        false_negatives[pred_bio]=1.
                    else:
                        false_negatives[pred_bio]+=1
                    for r in refs_bio:
                        if r not in false_positives.keys():
                            false_positives[r]=1.
                        else:
                            false_positives[r]+=1
                    #loss +=1
                else:
                    labels_str[pred_bio]=0
                    if pred_bio not in true_positives.keys():
                        true_positives[pred_bio]=1.
                    else:
                        true_positives[pred_bio]+=1

                if not all(p is None for p in refs_supersense):
                    if pred_supersense not in refs_supersense:
                        if pred_supersense not in false_negatives.keys():
                            false_negatives[pred_supersense]=1.
                        else:
                            false_negatives[pred_supersense]+=1
                        for r in refs_supersense:
                            if r not in false_positives.keys():
                                false_positives[r]=1.
                            else:
                                false_positives[r]+=1
                    #loss +=1
                    else:
                        labels_str[pred_supersense]=0
                        if pred_supersense not in true_positives.keys():
                            true_positives[pred_supersense]=1.
                        else:
                            true_positives[pred_supersense]+=1

                # map that prediction back to a BIO label
                if not all(p is None for p in refs_supersense):
                    this  = numeric_label_to_BIO(pred_bio,pred_supersense)
                else:
                    this  = numeric_label_to_BIO(pred_bio)
                # append it to output
                output.append(this)
                # update the 'previous' prediction to the current
                prev  = this

        Precision={}
        Recall={}
        F_Score={}

        for l in labels_str.keys():
            if l not in false_positives.keys():
                false_positives[l]=0.
            if l not in false_negatives.keys():
                false_negatives[l]=0.
            Precision[l]=true_positives[l]/(true_positives[l]+false_positives[l])
            Recall[l]=true_positives[l]/(true_positives[l]+false_negatives[l])
            F_Score[l]=2*Precision[l]*Recall[l]/(Precision[l]+Recall[l])

        s=0
        for l in F_Score.keys():
            s+=F_Score[l]

        if len(F_Score)>0:
            F_Score_mean=s/len(F_Score)
        else:
            F_Score_mean=0

        self.sch.loss(1-F_Score_mean)

        # return the list of predictions as BIO labels
        return output

    def make_example(self, word, lemma, pos, pword, plemma, ppos, nword, nlemma, npos, ppword, pplemma, pppos, nnword, nnlemma, nnpos):
        ex = {
            'a': ["1_"+word, "0_"+pword, "2_"+nword, "0_"+pword+"1_"+word, "1_"+word+"2_"+nword, "0_"+pword+"2_"+nword, "-1_"+ppword+"1_"+word, "1_"+word+"3_"+nnword],
            'b': ["1_"+lemma, "0_"+plemma, "0_"+plemma+"1_"+lemma, "1_"+lemma+"2_"+nlemma, "0_"+plemma+"2_"+nlemma, "-1_"+pplemma+"1_"+lemma, "1_"+lemma+"3_"+nnlemma],
            'c': ["1_"+pos, "0_"+ppos, "0_"+ppos+"1_"+pos, "1_"+pos+"2_"+npos, "0_"+ppos+"2_"+npos, "-1_"+pppos+"1_"+pos, "1_"+pos+"3_"+nnpos],
            'd': ["a1_"+word+"b1_"+lemma],
            'e': ["b1_"+lemma+"c1_"+pos],
            'f': ["a1_"+word+"c1_"+pos]
            #'g': ["1_"+word+"--1_"+ppword+": -2", "1_"+word+"-0_"+pword+": -1", "1_"+word+"-2_"+nword+": 1", "1_"+word+"-3_"+nnword+": 2"]
        }

        if pword+"_"+word in wordnetNounMWE:
            ex['g'] = ["0_"+pword+"_1_"+word+":nounmwe"]
        else:
            ex['g'] = ["0_"+pword+"_1_"+word+":notnounmwe"]

        if pword+"_"+word in wordnetVerbMWE:
            ex['g'] = ["0_"+pword+"_1_"+word+":verbmwe"]
        else:
            ex['g'] = ["0_"+pword+"_1_"+word+":notverbmwe"]

        if word+"_"+nword in wordnetNounMWE:
            ex['h'] = ["0_"+word+"_1_"+nword+":B"]
        else:
            ex['h'] = ["0_"+word+"_1_"+nword+":notB"]

        if word+"_"+nword in wordnetVerbMWE:
            ex['h'] = ["0_"+word+"_1_"+nword+":B"]
        else:
            ex['h'] = ["0_"+word+"_1_"+nword+":notB"]


        # if word+"_"+nword in wordnetNounMWE:
        #     ex['g'] = ["1_"+word+"_2_"+nword+":nounmwe"]
        #
        # if word+"_"+nword in wordnetVerbMWE:
        #     ex['g'] = ["1_"+word+"_2_"+nword+":verbmwe"]
        #
        # if plemma+"_"+lemma in wordnetNounMWE:
        #     ex['h'] = ["0_"+plemma+"_1_"+lemma+":nounmwe"]
        #
        # if plemma+"_"+lemma in wordnetVerbMWE:
        #     ex['h'] = ["0_"+plemma+"_1_"+lemma+":verbmwe"]
        #
        # if lemma+"_"+nlemma in wordnetNounMWE:
        #     ex['h'] = ["1_"+lemma+"_2_"+nlemma+":nounmwe"]
        #
        # if lemma+"_"+nlemma in wordnetVerbMWE:
        #     ex['h'] = ["1_"+lemma+"_2_"+nlemma+":verbmwe"]

        return self.example(ex)

def make_data(BIO,filename):
    data = []
    sentence = []
    f = open(filename,'r')
    for l in f:
        l = l.strip()
        # at end of sentence
        if l == "":
            data.append(sentence)
            sentence = []
        else:
            [offset,word,lemma,pos,mwe,parent,strength,ssense,sid] = l.split('\t')
            sentence.append((BIO(mwe),word,lemma,pos))
    return data


wordnetNounMWE = open('nouns_mwes_in_wordnet3.1.txt').read().splitlines()
wordnetVerbMWE = open('verbs_mwes_in_wordnet3.1.txt').read().splitlines()

if __name__ == "__main__":
    # input/output files
    trainfilename='dimsum16.p3.train.contiguous'
    testfilename='dimsum16.p3.test.contiguous'
    outfilename='dimsum16.p3.test.contiguous.out'

    # read in some examples to be used as training/dev set
    train_data = make_data(BIO,trainfilename)

    # initialize VW and sequence labeler as learning to search
    vw = pyvw.vw(search=3, quiet=True, search_task='hook', ring_size=1024, \
                 search_rollin='learn', search_rollout='none')

    # tell VW to construct your search task object
    sequenceLabeler = vw.init_search_task(MWE)

    # train!
    # we make 5 passes over the training data, training on the first 80%
    # examples (we retain the last 20% as development data)
    print 'training!'
    N = int(0.8 * len(train_data))
    for i in xrange(5):
        print 'iteration ', i, ' ...'
        sequenceLabeler.learn(train_data[0:N])

    # now see the predictions on 20% held-out sentences
    print 'predicting!'
    hamming_loss, total_words = 0,0
    for n in range(N, len(train_data)):
        truth = [label for label,word,lemma,pos in train_data[n]]
        pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )
        for i,t in enumerate(truth):
            if t != pred[i]:
                hamming_loss += 1
            total_words += 1
            print 'predicted:', '\t'.join(map(str, pred))
            print '    truth:', '\t'.join(map(str, truth))
            print ''

    print 'total hamming loss on dev set:', hamming_loss, '/', total_words

    # In Part II, you will have to output predictions on the test set.
    #test_data = make_data(BIO,testfilename)
    #for n in range(N, len(test_data)):
        # make predictions for current sentence
        #pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )
