import spacy
from spacy.matcher import Matcher
from collections import Counter
from operator import itemgetter
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from argparse import ArgumentParser

def calculate_dict(female_array, male_array):
    counter_f_h = Counter(female_array)
    counter_m_h = Counter(male_array)
    # make sure there is no key lookup error
    for key in set(counter_f_h) - set(counter_m_h):
        counter_m_h[key] = 0
    for key in set(counter_m_h) - set(counter_f_h):
        counter_f_h[key] = 0
    return counter_f_h, counter_m_h

def odds_ratio(f_dict, m_dict, topk=50, threshold=20):
    very_small_value = 0.00001
    if len(f_dict.keys()) != len(m_dict.keys()):
        raise Exception('The category for analyzing the male and female should be the same!')
    else:
        odds_ratio = {}
        total_num_f = sum(f_dict.values())
        total_num_m = sum(m_dict.values())
        for key in f_dict.keys():
            m_num = m_dict[key]
            f_num = f_dict[key]
            non_f_num = total_num_f - f_num
            non_m_num = total_num_m - m_num
            if f_num >= threshold and m_num >= threshold:
                # we only consider the events where there are at least {thresohld} occurences for both gender
                odds_ratio[key] = round((m_num / f_num) / (non_m_num / non_f_num), 2)
            else:
                continue
        return dict(sorted(odds_ratio.items(), key=itemgetter(1), reverse=True)[:topk]), dict(
            sorted(odds_ratio.items(), key=itemgetter(1))[:topk])

class Word_Extraction:
    def __init__(self, word_types=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        patterns = []

        for word_type in word_types:
            if word_type == 'noun':
                patterns.append([{'POS':'NOUN'}])
            elif word_type == 'adj':
                patterns.append([{'POS':'ADJ'}])
            elif word_type == 'verb':
                patterns.append([{"POS": "VERB"}])
        self.matcher.add("demo", patterns)

    def extract_word(self, doc):
        doc = self.nlp(doc)
        matches = self.matcher(doc)
        vocab = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            vocab.append(span.text)
        return vocab
    
def compute_lexical_content(list1, list2, threshold=10):
    
    noun_f, noun_m = [], []
    adj_f, adj_m = [], []
    len_f, len_m = [], []

    noun_extract = Word_Extraction(['noun'])
    adj_extract = Word_Extraction(['adj'])
    ability_m, standout_m, ability_f, standout_f = 0, 0, 0, 0
    masculine_m, feminine_m, masculine_f, feminine_f = 0, 0, 0, 0
    for i in tqdm(range(len(list1)), ascii=True):
        noun_vocab_f = noun_extract.extract_word(list1[i])
        # For normal analysis
        for v in noun_vocab_f:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            noun_f.append(v)
        
        adj_vocab_f = adj_extract.extract_word(list1[i])
        for v in adj_vocab_f:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            adj_f.append(v)


    for i in tqdm(range(len(list2)), ascii=True):
        noun_vocab_m = noun_extract.extract_word(list2[i])
        # For normal analysis
        for v in noun_vocab_m:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            noun_m.append(v)
        
        adj_vocab_m = adj_extract.extract_word(list2[i])
        for v in adj_vocab_m:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            adj_m.append(v)

    # For normal analysis
    noun_counter_f, noun_counter_m = calculate_dict(noun_f, noun_m)
    noun_res_m, noun_res_f = odds_ratio(noun_counter_f, noun_counter_m, threshold=threshold)
    adj_counter_f, adj_counter_m = calculate_dict(adj_f, adj_m)
    adj_res_m, adj_res_f = odds_ratio(adj_counter_f, adj_counter_m, threshold=threshold)

    output = {}
    output['noun_male'] = ", ".join(list(noun_res_m.keys())[:10])
    output['noun_female'] = ", ".join(list(noun_res_f.keys())[:10])
    output['adj_male'] = ", ".join(list(adj_res_m.keys())[:10])
    output['adj_female'] = ", ".join(list(adj_res_f.keys())[:10])

    # want to make df where cols are key of output and second col is list of values
    data = {
        'male': [output['noun_male'], output['adj_male']],
        'female': [output['noun_female'], output['adj_female']]
    }
    df = pd.DataFrame(data, index=['noun', 'adj'])
    return df