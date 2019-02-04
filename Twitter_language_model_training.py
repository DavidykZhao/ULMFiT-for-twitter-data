
# coding: utf-8

# In[84]:


from fastai.text import *
import html


# In[106]:


data_path1 = Path("/home/paperspace/data/training.1600000.processed.noemoticon.csv")
data_path2 = Path("/home/paperspace/data/testdata.manual.2009.06.14.csv")
lm_path_tt = Path("/home/paperspace/data/twitter/lm")


# In[67]:


twitter_lm_path = Path("/home/paperspace/data/twitter_lm")
twitter_lm_path.mkdir(exist_ok=True)


# In[12]:


data = pd.read_csv(data_path, encoding = "latin1",header=None)


# In[22]:


data2 = pd.read_csv(data_path2, encoding = "latin1",header=None)


# data.head()
# len(data)

# In[15]:


data_txt = data.iloc[:,[5]]


# In[23]:


data2_txt = data2.iloc[:,[5]]


# In[24]:


data2_txt.head(5)


# In[17]:


data_txt.head(5)


# In[51]:


data_txt; 


# In[61]:


trn_lm_text, test_lm_text = sklearn.model_selection.train_test_split(np.concatenate([data_txt.iloc[:,0], data2_txt.iloc[:,0]]), test_size = 0.1)


# In[62]:


print(len(trn_lm_text)); 
print(len(test_lm_text))
print(trn_lm_text[1:5])


# In[121]:


[len(i) for i in trn_lm_text]


# In[63]:


col_names = ["labels","text"]


# In[68]:


lm_trn = pd.DataFrame({"text": trn_lm_text, "labels": [0] * len(trn_lm_text)}, columns=col_names)
lm_test = pd.DataFrame({"text": test_lm_text, "labels": [0] * len(test_lm_text)}, columns=col_names)

lm_trn.to_csv(twitter_lm_path/"train.csv", header = False, index = False, encoding = "utf-8")
lm_test.to_csv(twitter_lm_path/"test.csv", header = False, index = False, encoding = "utf-8")


# In[78]:



## functions pulled from the fast.ai notebook for text tokenization

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


# In[79]:


BOS = "xboxs"
FLD = "xfld"


# In[80]:


chunksize = 24000


# In[81]:


twitter_lm_train_dl = pd.read_csv(twitter_lm_path/"train.csv", header=None, chunksize=chunksize)
twitter_lm_test_dl = pd.read_csv(twitter_lm_path/"test.csv", header=None, chunksize=chunksize)


# In[85]:


tok_trn_twitterlm, trn_labels_twittterlm = get_all(twitter_lm_train_dl, 1)
tok_test_twitterlm, test_labels_twittterlm = get_all(twitter_lm_test_dl, 1)


# In[86]:


(twitter_lm_path/"tmp").mkdir(exist_ok = True)
np.save(twitter_lm_path/"tmp/tok_trn_twitter.npy", tok_trn_twitterlm)
np.save(twitter_lm_path/"tmp/tok_test_twitter.npy", tok_test_twitterlm)
tok_trn_tt = np.load(twitter_lm_path/"tmp/tok_trn_twitter.npy")
tok_test_tt = np.load(twitter_lm_path/"tmp/tok_test_twitter.npy")


# In[88]:


freq = Counter(p for o in tok_trn_tt for p in o)
freq.most_common(25)


# In[89]:


max_vocab = 60000
min_freq = 2


# In[91]:


#itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
itos_tt = [o for o,c in freq.most_common(max_vocab)]

len(itos_tt)


# In[92]:


itos_tt.insert(0, "_pad_")
itos_tt.insert(1, "_unk_")


# In[93]:


stoi_tt = collections.defaultdict(lambda:0, {o:c for c,o in enumerate(itos_tt)})
len(stoi_tt)


# In[96]:


trn_lm_tt = np.array([[stoi_tt[o] for o in p] for p in tok_trn_tt])
test_lm_tt = np.array([[stoi_tt[o] for o in p] for p in tok_test_tt])


# In[97]:


np.save(twitter_lm_path/"tmp/trn_ids.npy", trn_lm_tt)
np.save(twitter_lm_path/"tmp/test_ids.npy", test_lm_tt)


# In[98]:


pickle.dump(itos_tt, open(twitter_lm_path/"tmp/itos.pkl", "wb"))


# In[100]:


trn_lm_tt = np.load(twitter_lm_path/"tmp/trn_ids.npy")
test_lm_tt = np.load(twitter_lm_path/"tmp/test_ids.npy")
itos_tt = pickle.load(open(twitter_lm_path/"tmp/itos.pkl", "rb"))


# In[102]:


vs=len(itos_tt)
vs,len(trn_lm_tt)


# # Language model 

# In[104]:


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# In[111]:


em_sz, nh, nl = 400, 1150, 3


# In[107]:


trn_dl_tt = LanguageModelLoader(np.concatenate(trn_lm_tt), bs, bptt)
test_dl_tt = LanguageModelLoader(np.concatenate(test_lm_tt), bs, bptt)
md_tt = LanguageModelData(lm_path_tt, 1, vs, trn_dl_tt, test_dl_tt, bs=bs, bptt=bptt)


# In[108]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


# In[112]:


learner= md_tt.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[113]:


lr=1e-3
lrs = lr


# In[114]:


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# In[122]:


learner.fit(lrs, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# In[123]:


learner.save('lm_last_tt_ft')
learner.load('lm_last_tt_ft')
learner.unfreeze()


# In[124]:


learner.fit(lrs, 1, wds=wd, use_clr_beta=(20,20, 0.95,0.85), cycle_len=10)


# In[125]:


learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


# In[126]:


learner.sched.plot()


# In[127]:


learner.fit(lrs, 1, wds=wd, use_clr_beta=(20,10,0.95,0.85), cycle_len=2)


# In[128]:


learner.fit(lrs/10, 1, wds=wd, use_clr_beta=(20,20,0.95,0.85), cycle_len=1)


# In[129]:


learner.save("lm_tt")


# In[130]:


learner.save_encoder("lm_tt_enc")


# In[131]:


learner.sched.plot_loss()

