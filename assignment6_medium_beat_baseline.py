
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). Edited by Sergey Kolchenko (@KolchenkoSergey). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# ## <center>Assignment #6
# ### <center> Beating baselines in "How good is your Medium article?"
#     
# 
# [Competition](https://www.kaggle.com/c/how-good-is-your-medium-article). The task is to beat "A6 baseline" (~1.45 Public LB score). Do not forget about our shared ["primitive" baseline](https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline) - you'll find something valuable there.
# 
# **Your task:**
#  1. "Freeride". Come up with good features to beat the baseline "A6 baseline" (for now, public LB is only considered)
#  2. You need to name your [team](https://www.kaggle.com/c/how-good-is-your-medium-article/team) (out of 1 person) in full accordance with the [course rating](https://drive.google.com/open?id=19AGEhUQUol6_kNLKSzBsjcGUU3qWy3BNUg8x8IFkO3Q). You can think of it as a part of the assignment. 16 credits for beating the mentioned baseline and correct team naming.

# In[276]:


import os
import json
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


# The following code will help to throw away all HTML tags from an article content.

# In[4]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Supplementary function to read a JSON line without crashing on escape characters.

# In[5]:


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# Extract features `content`, `published`, `title` and `author`, write them to separate files for train and test sets.

# In[148]:


def extract_features_and_write(path_to_data,
                               inp_filename, is_train=True):
    
    features = ['content', 'published', 'title', 'author']
    prefix = 'train' if is_train else 'test'
    feature_files = [open(os.path.join(path_to_data, 'generated',
                                       '{}_{}.txt'.format(prefix, feat)),
                          'w', encoding='utf-8')
                     for feat in features]
    
    with open(os.path.join(path_to_data, inp_filename), 
              encoding='utf-8') as inp_json_file:

        for line in tqdm_notebook(inp_json_file):
            json_data = read_json_line(line)

            for key, file in zip(features, feature_files):
                if key == 'content':
                    file.write(strip_tags(
                        json_data[key].replace('\n', ' ').replace('\r', ' '))+'\n')
                elif key == 'published':
                    file.write(list(json_data[key].values())[0]+'\n')
                elif key == 'author':
                    file.write(json_data[key]['url'] + '\n')
                elif key == 'title':
                    file.write(
                        json_data[key].replace('\n', ' ').replace('\r', ' ')+'\n')
                else:
                    file.write(json_data[key]+'\n')

        map(lambda file: file.close(), feature_files)
        


# In[149]:


with open('data/train.json') as f:
    for i in range(3):
        jsond = read_json_line(f.readline())


# In[151]:


PATH_TO_DATA = 'data/' # modify this if you need to


# In[152]:


extract_features_and_write(PATH_TO_DATA, 'train.json', is_train=True)


# In[153]:


extract_features_and_write(PATH_TO_DATA, 'test.json', is_train=False)


# **Add the following groups of features:**
#     - Tf-Idf with article content (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Tf-Idf with article titles (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Time features: publication hour, whether it's morning, day, night, whether it's a weekend
#     - Bag of authors (i.e. One-Hot-Encoded author names)

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


# #### Datetime features

# In[227]:


X_datetime_train = pd.read_csv('data/generated/train_published.txt', 
            names=['datetime','b'], 
            parse_dates=['datetime'])[['datetime']]
X_datetime_test = pd.read_csv('data/generated/test_published.txt', 
            names=['datetime','b'], 
            parse_dates=['datetime'])[['datetime']]

X_datetime_train.shape, X_datetime_test.shape


# In[258]:


def add_datetime_features(df):
    hour = df.datetime.apply(lambda time: time.hour)
    weekday = df.datetime.apply(lambda date: date.dayofweek)
    weekend = (weekday.isin([5,6]))
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
#     X = hstack([hour.values.reshape(-1, 1),
#                 weekend.values.reshape(-1,1),
#                 morning.values.reshape(-1,1),
#                 day.values.reshape(-1,1),
#                 evening.values.reshape(-1,1),
#                 night.values.reshape(-1,1)])
    return csr_matrix(np.hstack([hour.values.reshape(-1,1),
                weekend.values.reshape(-1,1),
                morning.values.reshape(-1,1),
                day.values.reshape(-1,1),
                evening.values.reshape(-1,1),
                night.values.reshape(-1,1)]))


# In[263]:


X_datetime_train_sparse = add_datetime_features(X_datetime_train)
X_datetime_test_sparse = add_datetime_features(X_datetime_test)


# #### Author features

# In[260]:


def getnick(path):
    return os.path.basename(os.path.normpath(path))


# In[261]:


X_authors_train = pd.read_csv(open('data/generated/train_author.txt'), 
                        names=['url']).url.apply(getnick)
X_authors_test = pd.read_csv(open('data/generated/test_author.txt'), 
                        names=['url']).url.apply(getnick)


# In[262]:


X_authors_train.shape, X_authors_test.shape


# In[199]:


ohe = OneHotEncoder(handle_unknown='ignore')  # Ignore unknown authors in test
X_authors_train_ohe = ohe.fit_transform(X_authors_train.values.reshape(-1, 1))
X_authors_test_ohe = ohe.transform(X_authors_test.values.reshape(-1, 1))


# #### Title and Content features

# In[175]:


tf_idf_vecr = TfidfVectorizer(ngram_range=(1,2), max_features=100000)


# In[176]:


with open('data/generated/train_title.txt') as input_file:
    X_title_train_tfidf = tf_idf_vecr.fit_transform(input_file)

with open('data/generated/test_title.txt') as input_file:
    X_title_test_tfidf = tf_idf_vecr.transform(input_file)
    
X_title_train_tfidf.shape, X_title_test_tfidf.shape


# In[178]:


get_ipython().run_cell_magic('time', '', "with open('data/generated/train_content.txt') as f:\n    X_content_train_tfidf = tf_idf_vecr.fit_transform(f)\nwith open('data/generated/test_content.txt') as f:\n    X_content_test_tfidf = tf_idf_vecr.transform(f)\n\nX_content_train_tfidf.shape, X_content_test_tfidf.shape")


# In[179]:


X_content_train_tfidf.shape, X_content_test_tfidf.shape


# **Join all sparse matrices.**

# In[267]:


X_train_sparse = hstack([X_content_train_tfidf, X_title_train_tfidf,
                         X_datetime_train_sparse, X_authors_train_ohe]).tocsr()
X_test_sparse = hstack([X_content_test_tfidf, X_title_test_tfidf,
                        X_datetime_test_sparse, X_authors_test_ohe]).tocsr()

X_train_sparse = hstack([X_train_content_sparse, X_train_title_sparse,
                         X_train_author_sparse, 
                         X_train_time_features_sparse]).tocsr()X_test_sparse = hstack([X_test_content_sparse, X_test_title_sparse,
                        X_test_author_sparse, 
                        X_test_time_features_sparse]).tocsr()
# **Read train target and split data for validation.**

# In[268]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')
y_train = train_target['log_recommends'].values


# In[272]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part_sparse = X_train_sparse[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid_sparse =  X_train_sparse[train_part_size:, :]
y_valid = y_train[train_part_size:]


# **Train a simple Ridge model and check MAE on the validation set.**

# In[273]:


ridge_regr = Ridge()
ridge_regr.fit(X_train_part_sparse, y_train_part)
mean_absolute_error(y_valid, ridge_regr.predict(X_valid_sparse))


# **Train the same Ridge with all available data, make predictions for the test set and form a submission file.**

# In[275]:


get_ipython().run_cell_magic('time', '', 'y_submit = ridge_regr.fit(X_train_sparse, y_train).predict(X_test_sparse)')


# In[288]:


plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(y_submit, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
#plt.hist(np.zeros_like(y_submit), bins=30, alpha=.5, color='green', label='pred', range=(0,10));

plt.legend();


# In[283]:


def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_DATA, 
                                                      'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# In[284]:


write_submission_file(y_submit, os.path.join(PATH_TO_DATA,
                                                    'assignment6_medium_submission.csv'))


# **Now's the time for dirty Kaggle hacks. Form a submission file with all zeroes. Make a submission. What do you get if you think about it? How is it going to help you with modifying your predictions?**

# In[286]:


write_submission_file(np.zeros_like(y_submit), 
                      os.path.join(PATH_TO_DATA,
                                   'medium_all_zeros_submission.csv'))


# **Modify predictions in an appropriate way (based on your all-zero submission) and make a new submission.**

# In[291]:


mean_right = 4.33328


# In[293]:


diff = mean_right - y_submit.mean()


# In[296]:


diff


# In[297]:


ridge_test_pred_modif = y_submit + diff # You code here


# In[298]:


write_submission_file(ridge_test_pred_modif, 
                      os.path.join(PATH_TO_DATA,
                                   'assignment6_medium_submission_with_hack.csv'))


# That's it for the assignment. Much more credits will be given to the winners in this competition, check [course roadmap](https://mlcourse.ai/roadmap). Do not spoil the assignment and the competition - don't share high-performing kernels (with MAE < 1.5).
# 
# Some ideas for improvement:
# 
# - Engineer good features, this is the key to success. Some simple features will be based on publication time, authors, content length and so on
# - You may not ignore HTML and extract some features from there
# - You'd better experiment with your validation scheme. You should see a correlation between your local improvements and LB score
# - Try TF-IDF, ngrams, Word2Vec and GloVe embeddings
# - Try various NLP techniques like stemming and lemmatization
# - Tune hyperparameters. In our example, we've left only 50k features and used C=1 as a regularization parameter, this can be changed
# - SGD and Vowpal Wabbit will learn much faster
# - Play around with blending and/or stacking. An intro is given in [this Kernel](https://www.kaggle.com/kashnitsky/ridge-and-lightgbm-simple-blending) by @yorko 
# - In our course, we don't cover neural nets. But it's not obliged to use GRUs/LSTMs/whatever in this competition.
# 
# Good luck!
# 
