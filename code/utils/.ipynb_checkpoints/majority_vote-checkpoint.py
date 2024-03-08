import pandas as pd

def count_(v, list_):
    c = 0
    if v in list_:
        c = list_.count(v)
    else:
        c = 0
    return c

def majority_vote(df, parent_text):
    df = df.sort_values(by=['id_text'], ignore_index=True) #new sort of the dataset restoring a new index
    df['label'] = df['label'].astype(str)

    if parent_text:
        df_aggregato = pd.DataFrame(columns=['id_text', 'text', 'parent_text', 'label'])
    else:
        df_aggregato = pd.DataFrame(columns=['id_text', 'text', 'label'])  
        
    dict_df = {}
    
    for i in range(0, len(df['id_text'].tolist())):
        if df['id_text'].tolist()[i] not in dict_df.keys():
            index=[]
            l = []
            l.append(df['label'].tolist()[i])
            index.append(i)
            
            if parent_text:
                dict_df[df['id_text'].tolist()[i]] = (df['text'].tolist()[i], l, index, df['parent_text'].tolist()[i])
            else:
                dict_df[df['id_text'].tolist()[i]] = (df['text'].tolist()[i], l, index)

        else:
            index.append(i)
            l.append(df['label'].tolist()[i])
            
            if parent_text:
                dict_df[df['id_text'].tolist()[i]] = (df['text'].tolist()[i], l, index, df['parent_text'].tolist()[i])
            else:
                dict_df[df['id_text'].tolist()[i]] = (df['text'].tolist()[i], l, index)
    
    txt = []
    label = []
    parent=[]
    
    for k, v in dict_df.items():
        txt.append(v[0])
    
        if parent_text:
            parent.append(v[3])
            
        c1= count_('1', v[1])
        c0= count_('0', v[1])
        # print(k,v[1], c1, c0)
        
        if c1 == c0:
            label.append('both')
            # print(k, v[1], v[2], v[0])
        elif c1>c0:
            label.append('1')
        else:
            label.append('0')

    df_aggregato['id_text'] = dict_df.keys()
    df_aggregato['text'] = txt
    df_aggregato['label'] = label
    
    if parent_text:
        df_aggregato['parent_text'] = parent
        
    # print(df_aggregato.shape)
    # print(df_aggregato)
    return df_aggregato

