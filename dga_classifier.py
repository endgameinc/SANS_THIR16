import numpy as np
import sklearn.feature_extraction
import sklearn.ensemble
import pandas as pd
import matplotlib
import tldextract
import math
from collections import Counter
import pickle
import json
import sys
from functools import partial
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
import argparse

#http://s3.amazonaws.com/alexa-static/top-1m.csv.zip
ALEXA_FILEPATH = 'top-1m.csv'
DICT_FILEPATH = '/usr/share/dict/words'
#http://osint.bambenekconsulting.com/feeds/dga-feed.txt
DGA_FILEPATH = 'dga-feed.txt'
CLASSIFIER_STORAGE = 'dga_classifier.pickle'

def get_domain(hostname):
    try:
        return tldextract.extract(hostname).domain
    except ValueError:
        print 'Error extracting domain from %s'%(hostname,)
    return np.nan

def get_subdomain(hostname):
    try:
        return tldextract.extract(hostname).subdomain
    except ValueError:
        print 'Error extracting domain from %s'%(hostname,)
    return np.nan

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def longest_consonant_sequence(s):
    vowels = set('aeiou')
    longest = 0
    current = 0
    for c in s:
        if c not in vowels:
            current += 1
        else:
            if current >= longest:
                longest = current
            current = 0
    if current >= longest:
        longest = current
        current = 0    
    return longest

def vowel_consonant_ratio(s):
    classes = {v:'v' for v in 'aeiou'}
    classes.update({'.':'d'})
    d = Counter([classes.get(c, 'c') for c in s])
    return float(d.get('v', 0))/d.get('c', 0) if d.get('c', 0) else np.nan

def strip_non_alpha(string):
    #Time to move to Python 3?
    delchars = '0123456789-'
    if isinstance(string, unicode):
        table = {ord(c):None for c in delchars}
        return string.translate(table)
    else:
        return string.translate(None, delchars)

def train_vectorizer(series):
    alexa_cv = sklearn.feature_extraction.text.CountVectorizer(
                   analyzer='char',
                   ngram_range=(3, 5),
                   min_df=1e-4,
                   max_df=1.0)
    counts_matrix = alexa_cv.fit_transform(series)
    alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
    return alexa_cv, alexa_counts

def calc_ngram_hits(df, cv, counts):
    return counts * cv.transform(df['domain_alpha_chars']).T


def cross_validate(fts, labels, clf, nfolds):
    scores = []
    true_labels = []
    for fold in range(nfolds):
        X_train, X_test, y_train, y_test = train_test_split(fts, labels, test_size=.2)
        clf.fit(X_train, y_train)

        scores.append(clf.predict_proba(X_test)[:,1])
        true_labels.append(y_test)
    ret = {}
    ret['fpr'], ret['tpr'], ret['thr'] = roc_curve(np.array(true_labels).ravel(), np.array(scores).ravel())
    ret['auc'] = auc(ret['fpr'], ret['tpr'])
    print ret['auc']
    return ret

def train(df, features, test_training=True, max_fpr=.05, nfolds = 10):
    for feature, feature_func in features.items():
        df[feature] = feature_func(df)

    df = df.dropna()
    X = df.as_matrix(features.keys())
    y = np.array(df['class'].tolist())
    # Make 0-1
    y = [x=='dga' for x in y]
    try:
        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
        validation_data = cross_validate(X, y, clf, nfolds)
        clf.fit(X, y)
        thr = validation_data['thr'][np.max(np.where(validation_data['fpr'] < .05))]
    except Exception as e:
        import pdb; pdb.set_trace()
        raise e
    return clf, thr, validation_data

def predict(clf, df, features, threshold):
    for feature, feature_func in features.items():
        df[feature] = feature_func(df)
    df = df.dropna()
    hold_X = df.as_matrix(features.keys())
    hold_y_pred = clf.predict(hold_X)
    prob = clf.predict_proba(hold_X)
    df['label'] = ['DGA' if x > threshold else 'Benign' for x in prob[:, 1]]
    df['prob_dga'] = prob[:, 1]
    return df

def prepare_df(df):
    df['domain'] = df['raw_domain'].apply(get_domain)
    df = df.dropna()
    df.loc[:,'domain_alpha_chars'] = df['domain'].apply(strip_non_alpha)
    return df

def read_alexa_df(filepath):
    alexa_df = pd.read_csv(filepath, names=('rank', 'raw_domain'), header=None, encoding='utf-8')
    alexa_df = alexa_df[:500000]
    del alexa_df['rank']
    alexa_df = prepare_df(alexa_df)
    alexa_df['class'] = 'benign'
    print 'Number of Alexa domains: %d' % alexa_df.shape[0]
    alexa_df = alexa_df.reindex(np.random.permutation(alexa_df.index))
    return alexa_df

def read_dga_df(filepath):
    dga_df = pd.read_csv(filepath, names=['raw_domain', 'family', 'date', 'link'], \
                         header=None, encoding='utf-8', comment='#')
    del dga_df['family']
    del dga_df['date']
    del dga_df['link']
    dga_df = dga_df.drop_duplicates()
    dga_df = prepare_df(dga_df)
    dga_df['class'] = 'dga'
    dga_df = dga_df.dropna()
    print 'Number of DGA domains: %d' % dga_df.shape[0]
    return dga_df

def train_and_serialize(filepath, max_fpr=.05, nfolds=10, dispr=True):
    alexa_df = read_alexa_df(ALEXA_FILEPATH)
    dga_df = read_dga_df(DGA_FILEPATH)
    all_domains = pd.concat([alexa_df, dga_df], ignore_index=True)
    alexa_cv, alexa_counts = train_vectorizer(alexa_df['domain_alpha_chars'])
    dict_df = pd.read_csv(DICT_FILEPATH, names=['word',]).dropna()
    dict_cv, dict_counts = train_vectorizer(dict_df['word'])
    features = {
        'len': lambda df: df['domain'].apply(len),
        'entropy':lambda df: df['domain'].apply(entropy),
        'vowel_consonant_ratio': lambda df: df['domain'].apply(vowel_consonant_ratio),
        'longest_consonant_sequence': lambda df: df['domain'].apply(longest_consonant_sequence),
        'alexa_ngrams': partial(calc_ngram_hits, cv=alexa_cv, counts=alexa_counts),
        'dict_ngrams': partial(calc_ngram_hits, cv=dict_cv, counts=dict_counts),
    }
    clf, thr, validation_data = train(all_domains, features, max_fpr=max_fpr, nfolds=nfolds)

    outf = {'clf':clf,
            'thr':thr,
            'alexa_cv':alexa_cv,
            'alexa_counts':alexa_counts,
            'dict_cv':dict_cv,
            'dict_counts':dict_counts,
            'validation_data':validation_data}

    with open(filepath, 'w') as fp:
        pickle.dump(outf, fp)

    if dispr:
        display_roc(outf)

def display_roc(data):
    import matplotlib.pyplot as plt
    plt.plot(data['validation_data']['fpr'], data['validation_data']['tpr'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(data['validation_data']['auc']),
             linewidth=2)
    idx = np.where(data['validation_data']['thr'] == data['thr'])[0]
    ax = plt.axes()
    ax.annotate("Threshold = %f" % (data['thr'], ),
            xy=(data['validation_data']['fpr'][idx], data['validation_data']['tpr'][idx]),
            xycoords='data',
            xytext=(data['validation_data']['fpr'][idx]+.1, data['validation_data']['tpr'][idx]-.3),
            textcoords='data',
            size=16, va="center", ha="left",
            arrowprops=dict(arrowstyle="simple",
                            facecolor='black'),
            )


    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def load_and_predict(filepath, df):
    with open(filepath, 'r') as fp:
        data = pickle.load(fp)
    clf = data['clf']
    alexa_cv, alexa_counts = data['alexa_cv'], data['alexa_counts']
    dict_cv, dict_counts = data['dict_cv'], data['dict_counts']
    threshold = data['thr']
    features = {
        'len': lambda df: df['domain'].apply(len),
        'entropy':lambda df: df['domain'].apply(entropy),
        'vowel_consonant_ratio': lambda df: df['domain'].apply(vowel_consonant_ratio),
        'longest_consonant_sequence': lambda df: df['domain'].apply(longest_consonant_sequence),
        'alexa_ngrams': partial(calc_ngram_hits, cv=alexa_cv, counts=alexa_counts),
        'dict_ngrams': partial(calc_ngram_hits, cv=dict_cv, counts=dict_counts),
    }
    return predict(clf, df, features, threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit or Predict the DGA classifier')
    parser.add_argument('-f', '--fit',
                        action='store_true',
                        help='Predict a new DGA classifier model based',
                        default=False)
    parser.add_argument('-p', '--predict',
                        action='store',
                        help='Predict label for domains from a file containing JSON encoded list of domains',
                        default=None)
    args = parser.parse_args()
    if args.fit:
        train_and_serialize(CLASSIFIER_STORAGE, max_fpr=.05, nfolds=10, dispr=True)
    elif args.predict:
        inputfile = args.predict
        with open(inputfile, 'r') as fp:
            data = json.load(fp)
        df = pd.DataFrame(data, columns=['raw_domain'])
        df = prepare_df(df)
        df = load_and_predict(CLASSIFIER_STORAGE, df)
        print df[df['label'] == 'DGA'][['raw_domain', 'prob_dga']].sample(n=10)
        print df[df['label'] != 'DGA'][['raw_domain', 'prob_dga']].sample(n=10)
        print 'DGA = %d, Benign = %d, Total = %d'%( \
                    len(df[df['label'] == 'DGA']),
                    len(df[df['label'] != 'DGA']),
                    len(df))
        df_list = df.to_dict(orient='list')
        res = {}
        keys = ('raw_domain', 'prob_dga', 'label')
        for (domain, prob_dga, label) in zip(*(df_list[key] for key in keys)):
            res[domain] = (label, prob_dga)
        with open(inputfile.split('.')[0] + '_res.json', 'w') as fp:
            json.dump(res, fp)
    else:
        parser.error('One of the options fit or predict must be selected')
