
# coding: utf-8

# In[4]:


from fastai.text import *


# In[ ]:





# In[5]:


PATH = Path("/home/paperspace/fastai/courses/dl2/data")


# In[6]:


data = pd.read_csv(PATH/"gop_sentiments.csv", encoding= "latin1")


# In[7]:


data = data[["sentiment", "text"]]


# In[259]:


data.columns = ['labels', 'text']
data.head(200)


# In[8]:


CLAS_PATH = Path("/home/paperspace/fastai/courses/dl2/data/gop_class")
CLAS_PATH.mkdir(exist_ok=True)
LM_PATH = Path("/home/paperspace/fastai/courses/dl2/data/gop_lm")
LM_PATH.mkdir(exist_ok= True)


# In[9]:


cleanup_nums = {"labels":     {"Negative": 0, "Neutral": 1, "Positive":2}
                }
data.replace(cleanup_nums, inplace=True)
data.head


# In[10]:


split_index = int(data.shape[0] * .8)
train_df, test_df = np.split(data, [split_index], axis=0)


# In[364]:


print(train_df.shape)
print(test_df.shape)
print(train_df.head(10))


# In[11]:


## write the clasification repo
train_df.to_csv(CLAS_PATH/"train.csv", header=False, index=False,encoding="utf-8")
test_df.to_csv(CLAS_PATH/"test.csv", header=False, index=False,encoding="utf-8")

train_texts, test_texts = sklearn.model_selection.train_test_split(
    np.concatenate([train_df["text"],test_df["text"]]), test_size = 0.1)

colnames = ["labels","text"]
df_train = pd.DataFrame({"text": train_texts, "labels": [0]*len(train_texts)}, columns=colnames)
df_test = pd.DataFrame({"text": test_texts, "labels": [0]*len(test_texts)}, columns=colnames)

## write to lm repo
df_train.to_csv(LM_PATH/"train.csv", header=False, index=False, columns=colnames, encoding="utf-8")
df_test.to_csv(LM_PATH/"test.csv", header=False, index=False, columns=colnames, encoding="utf-8")

#train_df.to_csv(LM_PATH/'train.csv', header=False, index=False, encoding='utf-8')
#test_df.to_csv(LM_PATH/'test.csv', header=False, index=False, encoding='utf-8')


# In[12]:


print(df_train.shape)
print(df_test.shape)


# In[13]:


CLASSES = ["Neutral","Positive","Negative"]
(CLAS_PATH/"classes.txt").open("w", encoding = "utf-8").writelines(f"{cls}\n" for cls in CLASSES)


# In[279]:



#print(train_df.head(20))


# In[14]:


# chunksize for pandas so it doesn't run into any memory limits
chunksize=2400


# In[15]:



## functions pulled from the fast.ai notebook for text tokenization

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


# In[16]:


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


# In[17]:


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


# In[284]:


df_train = pd.read_csv(LM_PATH/"train.csv", header=None, chunksize=chunksize)
df_test = pd.read_csv(LM_PATH/"test.csv", header=None, chunksize=chunksize)


# In[285]:


next(iter(df_train))


# In[286]:


BOS = "xbos"
FLD = "xfld"


# In[287]:



tok_train, train_labels = get_all(df_train,1)
tok_test, test_labels = get_all(df_test,1)


# In[20]:


(LM_PATH/"tmp").mkdir(exist_ok = True)
tmp = Path("/home/paperspace/fastai/courses/dl2/data/gop_lm/tmp")


# In[18]:


np.save(tmp/"tok_trn.npy", tok_train)
np.save(tmp/"tok_test.npy", tok_test)


# In[21]:


tok_train = np.load(tmp/"tok_trn.npy")
tok_test = np.load(tmp/"tok_test.npy")


# In[23]:


len(tok_train) # len = 10083
len(tok_test)
#len(train_labels)


# In[24]:


freq = Counter(p for o in tok_train for p in o)
freq.most_common(25)


# In[25]:


max_vocab = 60000
min_freq = 2


# In[399]:


#itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
itos = [o for o,c in freq.most_common(max_vocab)]

len(itos)


# In[400]:


itos.insert(0, "_pad_")
itos.insert(1, "_unk_")


# In[401]:


stoi = collections.defaultdict(lambda:0, {o:c for c,o in enumerate(itos)})
len(stoi)


# In[402]:


trn_lm = np.array([[stoi[o] for o in p] for p in tok_train])
test_lm = np.array([[stoi[o] for o in p] for p in tok_test])
#"".join(str(o) for o in trn_lm[1])
len(trn_lm)


# In[403]:


np.save(tmp/"trn_ids.npy", trn_lm)
np.save(tmp/"test_ids.npy", test_lm)


# In[404]:


pickle.dump(itos, open(tmp/"itos.pkl", "wb"))


# In[26]:


trn_lm = np.load(tmp/"trn_ids.npy")
test_lm = np.load(tmp/"test_ids.npy")
itos = pickle.load(open(tmp/"itos.pkl", "rb"))


# In[27]:


vs=len(itos)
vs,len(trn_lm)


# In[28]:


em_sz, nh, nl = 400, 1150, 3


# In[29]:


PRE_PATH = Path("/home/paperspace/fastai/courses/dl2/data/aclImdb/models/wt103")
PRE_LM_PATH = Path(PRE_PATH/"fwd_wt103.h5")


# In[30]:


wgts = torch.load(PRE_LM_PATH, map_location = lambda storage, loc: storage)


# In[31]:


enc_wgts = to_np(wgts["0.encoder.weight"])
row_m = enc_wgts.mean(0)


# In[32]:


itos2 = pickle.load((PRE_PATH/"itos_wt103.pkl").open("rb"))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# In[33]:


new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


# In[34]:


wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


# ###Language model

# In[35]:


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# In[36]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
test_dl = LanguageModelLoader(np.concatenate(test_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, test_dl, bs=bs, bptt=bptt)


# In[37]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


# In[38]:


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)


# In[39]:


learner.model.load_state_dict(wgts)


# In[40]:


lr=1e-3
lrs = lr


# In[41]:


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


# In[42]:


learner.save('lm_last_ft')


# In[43]:


learner.load('lm_last_ft')


# In[44]:


learner.unfreeze()


# In[45]:


learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


# In[46]:


learner.sched.plot()


# In[47]:


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=10)


# In[ ]:


learner.save('lm1')


# In[ ]:


learner.save_encoder('lm1_enc')


# In[ ]:


learner.sched.plot_loss()


# In[430]:


train_df = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
test_df = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)


# In[ ]:





# In[431]:


tok_trn, trn_labels = get_all(train_df, 1)
tok_val, val_labels = get_all(test_df, 1)




# In[432]:


(CLAS_PATH/'tmp').mkdir(exist_ok=True)

np.save(CLAS_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(CLAS_PATH/'tmp'/'tok_val.npy', tok_val)

np.save(CLAS_PATH/'tmp'/'trn_labels.npy', trn_labels)
np.save(CLAS_PATH/'tmp'/'val_labels.npy', val_labels)


# In[53]:


tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')


# In[55]:


itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[56]:


trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])


# In[57]:


np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)
trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')


# ## CLssification

# In[58]:


trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))


# In[59]:


bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48


# In[60]:


min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1


# In[61]:


trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)


# In[62]:


dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5


# In[63]:


m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[64]:


opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# In[65]:


learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=.25
learn.metrics = [accuracy]


# In[66]:


lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])


# In[445]:


lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])


# In[67]:


wd = 1e-7
wd = 0
learn.load_encoder('lm1_enc')


# In[68]:


learn.freeze_to(-1)


# In[69]:


learn.lr_find(lrs/1000)
learn.sched.plot()


# In[70]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[71]:


learn.save('clas_0')


# In[72]:


learn.load('clas_0')


# In[73]:


learn.freeze_to(-2)


# In[74]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))


# In[75]:


learn.save('clas_1')


# In[76]:


learn.load('clas_1')


# In[77]:


learn.unfreeze()


# In[78]:


learn.fit(lrs, 1, wds=wd, cycle_len=5, use_clr=(32,10))


# In[79]:


learn.sched.plot_loss()


# In[398]:


learn.save('clas_2')

