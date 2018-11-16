
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). Edited by Sergey Kolchenko (@KolchenkoSergey). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# ## <center>Assignment #6
# ### <center> Beating baselines in "How good is your Medium article?"
#     
# <img src='../../img/medium_claps.jpg' width=40% />
# 
# 
# [Competition](https://www.kaggle.com/c/how-good-is-your-medium-article). The task is to beat "A6 baseline" (~1.45 Public LB score). Do not forget about our shared ["primitive" baseline](https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline) - you'll find something valuable there.
# 
# **Your task:**
#  1. "Freeride". Come up with good features to beat the baseline "A6 baseline" (for now, public LB is only considered)
#  2. You need to name your [team](https://www.kaggle.com/c/how-good-is-your-medium-article/team) (out of 1 person) in full accordance with the [course rating](https://drive.google.com/open?id=19AGEhUQUol6_kNLKSzBsjcGUU3qWy3BNUg8x8IFkO3Q). You can think of it as a part of the assignment. 16 credits for beating the mentioned baseline and correct team naming.

# In[ ]:


import os
import json
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge


# The following code will help to throw away all HTML tags from an article content.

# In[ ]:


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

# In[ ]:


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

# In[ ]:


def extract_features_and_write(path_to_data,
                               inp_filename, is_train=True):
    
    features = ['content', 'published', 'title', 'author']
    prefix = 'train' if is_train else 'test'
    feature_files = [open(os.path.join(path_to_data,
                                       '{}_{}.txt'.format(prefix, feat)),
                          'w', encoding='utf-8')
                     for feat in features]
    
    with open(os.path.join(path_to_data, inp_filename), 
              encoding='utf-8') as inp_json_file:

        for line in tqdm_notebook(inp_json_file):
            json_data = read_json_line(line)
            
            # You code here


# In[ ]:


PATH_TO_DATA = '../../data/kaggle_medium' # modify this if you need to


# In[ ]:


extract_features_and_write(PATH_TO_DATA, 'train.json', is_train=True)


# In[ ]:


extract_features_and_write(PATH_TO_DATA, 'test.json', is_train=False)


# **Add the following groups of features:**
#     - Tf-Idf with article content (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Tf-Idf with article titles (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Time features: publication hour, whether it's morning, day, night, whether it's a weekend
#     - Bag of authors (i.e. One-Hot-Encoded author names)

# In[ ]:


# You code here


# **Join all sparse matrices.**

# In[ ]:


X_train_sparse = hstack([X_train_content_sparse, X_train_title_sparse,
                         X_train_author_sparse, 
                         X_train_time_features_sparse]).tocsr()


# In[ ]:


X_test_sparse = hstack([X_test_content_sparse, X_test_title_sparse,
                        X_test_author_sparse, 
                        X_test_time_features_sparse]).tocsr()


# **Read train target and split data for validation.**

# In[ ]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')
y_train = train_target['log_recommends'].values


# In[ ]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part_sparse = X_train_sparse[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid_sparse =  X_train_sparse[train_part_size:, :]
y_valid = y_train[train_part_size:]


# **Train a simple Ridge model and check MAE on the validation set.**

# In[ ]:


# You code here


# **Train the same Ridge with all available data, make predictions for the test set and form a submission file.**

# In[ ]:


# You code here


# In[ ]:


def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_DATA, 
                                                      'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# In[ ]:


write_submission_file(ridge_test_pred, os.path.join(PATH_TO_DATA,
                                                    'assignment6_medium_submission.csv'))


# **Now's the time for dirty Kaggle hacks. Form a submission file with all zeroes. Make a submission. What do you get if you think about it? How is it going to help you with modifying your predictions?**

# In[ ]:


write_submission_file(np.zeros_like(ridge_test_pred), 
                      os.path.join(PATH_TO_DATA,
                                   'medium_all_zeros_submission.csv'))


# **Modify predictions in an appropriate way (based on your all-zero submission) and make a new submission.**

# In[ ]:


ridge_test_pred_modif = ridge_test_pred # You code here


# In[ ]:


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
# <img src='../../img/kaggle_shakeup.png' width=50%>
