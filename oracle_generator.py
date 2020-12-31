
# Credits
# Originally Written by Shashi Narayan to use original ROUGE
# Improved by Yang Liu to use a must faster ROUGE

# Refactored by Alex Zhuang to use Nallapati's procedure
import itertools
from multiprocessing import Pool
import rouge


def cal_rouge(fullset, sentdata, golddata):
    fullset.sort()
    model_highlights = [sentdata[idx] for idx in range(len(sentdata)) if idx in fullset]
    rouge_1 = rouge.rouge_n(model_highlights, golddata, 1)['f']
    rouge_2 = rouge.rouge_n(model_highlights, golddata, 2)['f']
    rouge_l = rouge.rouge_l_summary_level(model_highlights, golddata)['f']
    rouge_score = (rouge_1 + rouge_2 + rouge_l)/3.0
    return (rouge_score, fullset)


def _multi_run_wrapper(args):
    return cal_rouge(*args)

def nallapati_method(article_sents:list, abstract_sents:list):
    """implementation of gen_sentence_labels using the method from Nallapati et al. 2016 
    After calculating the toprougesentences, greedily append starting from the top and
    evaluate rouge against the current summary of sentences until rouge does not improve.
    """
    
    sentids_lst = [[sentid] for sentid in range(len(article_sents))]
    rougescore_sentwise = []
    for sentids in sentids_lst:
        rougescore_sentwise.append(cal_rouge(sentids, article_sents, abstract_sents))

    rougescore_sentwise.sort(reverse=True)
    
    toprougesentences = [item[1][0] for item in rougescore_sentwise]
    print(toprougesentences)
    
    selected_summary = []
    candidate_summary = []
    prev_score = -1
    cur_score = 0 #placeholder
    topn = 1
    while prev_score < cur_score:
        prev_score = cur_score
        selected_summary = candidate_summary
        candidate_summary = selected_summary + [toprougesentences[topn-1]]
        cur_score, candidate_summary = cal_rouge(candidate_summary, article_sents, abstract_sents)
        topn += 1
    return (cur_score, selected_summary)
    

def gen_sentence_labels(article_sents:list, abstract_sents:list):
    pool = Pool(8)
    sent_limit = 4

    sentids_lst = [[sentid] for sentid in range(len(article_sents))]
    rougescore_sentwise = []
    for sentids in sentids_lst:
        rougescore_sentwise.append(cal_rouge(sentids, article_sents, abstract_sents))

    rougescore_sentwise.sort(reverse=True)

    toprougesentences = [item[1][0] for item in rougescore_sentwise[:10]]
    toprougesentences.sort()

    arguments_list = []
    for itemcount in range(2, sent_limit + 1):
        arguments_list += [(list(sentids), article_sents, abstract_sents) for sentids in
                           itertools.combinations(toprougesentences, itemcount)]
    rougescore_sentids = pool.map(_multi_run_wrapper, arguments_list)

    # Process results
    # for rougescore, arguments in zip(rougescore_list, arguments_list):
    #     rougescore_sentids.append((rougescore, arguments[0]))

    # rougescore_sentids = []
    # for sentids in sentids_lst:
    #     rougescore_sentids.append(cal_rouge(sentids, sentdata, golddata))

    rougescore_sentids.sort(reverse=True)
    return rougescore_sentids