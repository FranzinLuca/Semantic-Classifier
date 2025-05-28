from nltk.lm import Vocabulary
from wikidata.client import Client
import pandas as pd
import itertools
from utils import *

# Create a new column in the dataset to store the properties
def create_claims_col(csv_file, dataset):
    try:
        properties_df = pd.read_csv(csv_file)
        properties_df['claim'] = properties_df['claim'].apply(convert_str_to_list)
    except:
        client = Client()
        properties_df = pd.DataFrame(columns=['item', 'statement', 'claim'])
        for idx, row in dataset.iterrows():
            """add the properties to the dataset
            """
            
            item = row['item']
            id = extract_entity_id(item)
            #prop_dict, prop_cache, claim_cache = extract_properties_str(id, client, prop_cache, claim_cache)
            prop_dict = extract_properties(id, client)
            keys = prop_dict.keys()
            
            for key in keys:
                index = properties_df.index.max() + 1 if not properties_df.empty else 0
                properties_df.loc[index, 'item'] = id
                properties_df.loc[index, 'statement'] = key
                properties_df.loc[index, 'claim'] = prop_dict[key]
            
            print(f"Processed {idx} rows", end='\r')

        properties_df.to_csv(csv_file, index=False)
    return properties_df

def create_vocab(value_list, isListOfLists, min_word_frequency):
    if isListOfLists:
        flatten_list = [item for sublist in value_list for item in sublist]
    else: flatten_list = value_list
    
    vocab_prop_conc = Vocabulary(flatten_list, unk_cutoff=min_word_frequency)
    known_words = sorted(list(vocab_prop_conc))
    UNK_LABEL = vocab_prop_conc.unk_label
    word_to_idx = {UNK_LABEL: 0}
    for i, word in enumerate(known_words):
        if word != UNK_LABEL:
            word_to_idx[word] = i + 1
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return vocab_prop_conc, word_to_idx, idx_to_word

def conv_word_to_idx(row, word_to_idx, unk_label, col_name):
    claims = row[col_name]
    list_of_claims = []
    if isinstance(claims, list):
        for claim in claims:
            if claim in word_to_idx:
                list_of_claims.append(word_to_idx[claim]) 
            else:
                list_of_claims.append(word_to_idx[unk_label])
    else:
        if claims in word_to_idx:
            list_of_claims.append(word_to_idx[claims]) 
        else:
            list_of_claims.append(word_to_idx[unk_label])
    
    return list_of_claims

def create_statement_claim_pairs(row):
    props = row['statement']
    claims = row['claim']
    pairs = []
    
    for claim in claims:
        pairs.append((props[0], claim))
    return pairs

def prepare_dataset(dataset, file_cache, is_train=True, word_to_idx_p = None, word_to_idx_q = None, unk_label = None, min_word_frequency = 1):
    """function used to prepare the dataset for the model

    Args:
        dataset (pandas dataframe): the dataset to prepare
        file_cache (string): the path to the file cache
        
    Returns:
       int: size of the p vocabulary
       vocabulary: p vocabulary
       int: size of the q vocabulary
       vocabulary: q vocabulary
       pandas dataframe: dataset modified 
    """
    
    # extract all the properties with a specific code (PXXX or QXXX)
    properties_df = create_claims_col(file_cache, dataset)
    p_list = properties_df['statement'].tolist()
    q_list = properties_df['claim'].tolist()
    
    if is_train:
        # create the vocabularies for the properties and the claims
        vocab_p, word_to_idx_p, idx_to_word_p = create_vocab(p_list, False, min_word_frequency)
        vocab_q, word_to_idx_q, idx_to_word_q = create_vocab(q_list, True, min_word_frequency)
        unk_label = vocab_p.unk_label
    
    # retrive the size of the vocabularies
    vocab_size_p = len(word_to_idx_p) + 1
    vocab_size_q = len(word_to_idx_q) + 1
    
    # convert all the properties to their index
    converted_properties = properties_df.copy()
    converted_properties['statement'] = properties_df.apply(lambda row: conv_word_to_idx(row, word_to_idx_p, unk_label, 'statement'),axis=1)
    converted_properties['claim'] = properties_df.apply(lambda row: conv_word_to_idx(row, word_to_idx_q, unk_label, 'claim'),axis=1)
    converted_properties['pairs'] = converted_properties.apply(create_statement_claim_pairs, axis=1)
    
    # group the prop_conv_idx by item flattening the list
    grouped_properties = converted_properties.groupby('item').agg(pairs=('pairs', list)).reset_index()
    grouped_properties['pairs'] = grouped_properties['pairs'].apply(lambda x: list(itertools.chain.from_iterable(x)))
    
    dataset['id_item'] = dataset['item'].apply(lambda x: extract_entity_id(x))
    # merge the dataset with the properties_df_conc
    prop_conc_df = grouped_properties[['item', 'pairs']]
    dataset_conc = pd.merge(dataset, prop_conc_df, left_on='id_item', right_on='item', how='left')
    dataset_conc.drop(columns=['item_y'], inplace=True)
    # convert the label to a int 0 if label = cultural exclusive, 1 is cultural representative 2 otherwise
    dataset_conc['label_int'] = dataset_conc['label'].apply(lambda x: 0 if x == 'cultural exclusive' else (1 if x == 'cultural representative' else 2))
    
    return vocab_size_p, word_to_idx_p, vocab_size_q, word_to_idx_q, dataset_conc, unk_label