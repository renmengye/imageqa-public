#!/usr/bin/env python

# Mateusz Malinowski
# mmalinow@mpi-inf.mpg.de
# Hash Speedup Modification made by Mengye Ren
# mren@cs.toronto.edu

# it assumes there are two files
# - first file with ground truth answers
# - second file with predicted answers
# both answers are line-aligned

import sys
import re

#import enchant

from numpy import prod
from nltk.corpus import wordnet as wn

word_pair_dict = {}

def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k for k in 
            [k.strip() for k in f.readlines()] 
        if len(k) > 0]

    return lines


def list2file(filepath,mylist):
    mylist='\n'.join(mylist)
    with open(filepath,'w') as f:
        f.writelines(mylist)


def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):
    """
    A: list of A items 
    TT: list of TT items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A 

    This function implements a fuzzy accuracy score:
        score(A,TT) = min{min_{a \in A} m(a \in TT), min_{t \in TT} m(a \in A)}
        where A and TT are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1

    """
    Warning!! This is only for single-word answer!!
    """
    A = [A]
    T = [T]

    score_left=0 if A==[] else prod(map(lambda a: m(a,T), A))
    score_right=0 if T==[] else prod(map(lambda t: m(t,A),T))
    return min(score_left,score_right) 


# implementations of different measure functions
def dirac_measure(a,b):
    """
    Returns 1 iff a=b and 0 otherwise.
    """
    return float(a==b)


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    if word_pair_dict.has_key(a+','+b):
        return  word_pair_dict[a+','+b]

    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)

        # I think, doesn't help much; anyway I want to capture highly accurate
        # results
        #if semantic_field == []:
        ## if empty try spelling correction, downweight, and try again
        #    spelling = enchant.Dict('en_US')
        #    if a == '':
        #        return ([],0)
        # 
        #    tmp = spelling.suggest(a)
        #
        #    if tmp == []:
        #        return ([],0)
            
        #    a = tmp[0]
        #    weight = 0.1
        #    semantic_field = wn.synsets(a,pos=wn.NOUN)

        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        # if the objects have ids, remove them
        if re.match('.*:.*id/2',a):
            a = re.search(':.*id',a).group().lstrip(':').rstrip('id')
            weight = 0.9
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    word_pair_dict[a+','+b] = final_score
    return final_score 
###

def runAll(gt_filepath, pred_filepath, thresh):
    global word_pair_dict
    word_pair_dict = {}
    input_gt=file2list(gt_filepath)
    input_pred=file2list(pred_filepath)

    # print 'input gt'
    # print input_gt
    # print 'input pred'
    # print input_pred

    if thresh == -1:
        our_element_membership=dirac_measure
    else:
        our_element_membership=lambda x,y: wup_measure(x,y,thresh)

    our_set_membership=\
            lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)

    if thresh == -1:
        print 'standard Accuracy is used'
    else:
        print 'soft WUPS is used'
    score_list=[score_it(ta,pa,our_set_membership) 
            for (ta,pa) in zip(input_gt,input_pred)]
    print 'computing the final score'
    final_score=sum(map(lambda x:float(x)/float(len(score_list)),score_list))

    # filtering to obtain the results
    #print 'full score:', score_list
    #print 'full score:', score_list
    print 'final score:', final_score
    return final_score


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print 'Usage: path to true answers, path to predicted answers, threshold'
        print 'If threshold is -1, then the standard Accuracy is used'
        sys.exit("3 arguments must be given")

    # folders
    gt_filepath=sys.argv[1]
    pred_filepath=sys.argv[2]
    thresh=float(sys.argv[3])
    runAll(gt_filepath, pred_filepath, thresh)
