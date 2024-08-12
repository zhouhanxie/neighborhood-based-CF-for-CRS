# setup these file dirs, you can ignore the rest unless you are modifying the code
input_file_dir = 'datasets/reddit_small/reddit_small_train.csv'
output_file_dir = 'reddit_small_semantic_embs_20000.pt'

# generating item embeddings
import pandas as pd
reddit_posts_train = pd.read_csv(input_file_dir)
from itertools import chain
from collections import defaultdict
from collections import Counter

eligible_movies = [eval(i) for i in reddit_posts_train['movies']]
frequency = Counter(list(chain.from_iterable(eligible_movies)))
sum(i[1] for i in frequency.most_common(20000)) / sum(i[1] for i in frequency.items())

unique_movie_names = sorted(i[0] for i in frequency.most_common(20000))
movie2id = {movie: idx for idx, movie in enumerate(unique_movie_names)}
id2movie = {idx:movie for movie,idx in movie2id.items()}
len(movie2id)

import numpy as np
posts = np.array(list(reddit_posts_train['full_situation']))

from collections import defaultdict
movie2post_ids = defaultdict(set)

for post_id, movies in enumerate(eligible_movies):
    for movie in movies:
        if movie in movie2id:
            movie2post_ids[movie].add(post_id)

movie2post_ids = dict(movie2post_ids)

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Sentences we want sentence embeddings for
sentences = [str(i) for i in posts]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
model.eval()

# Initialize an empty list to hold the sentence embeddings
all_sentence_embeddings = []

# Process sentences in batches
batch_size = 64

from tqdm import tqdm
for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences)//batch_size):
    batch_sentences = sentences[i:i+batch_size]
    
    # Tokenize sentences
    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
    
    all_sentence_embeddings.append(sentence_embeddings)

# Concatenate all batched embeddings
all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#print("Sentence embeddings:")
#print(all_sentence_embeddings)

all_sentence_embeddings = all_sentence_embeddings.cpu()

movie_embeddings = []

for movie in tqdm(unique_movie_names):
    associated_post_ids = torch.Tensor(list(movie2post_ids[movie]))
    movie_embedding = torch.mean(
        all_sentence_embeddings[list(movie2post_ids[movie])], dim=0
    )
    movie_embeddings.append(movie_embedding.cpu().numpy().tolist())

embs = torch.tensor(movie_embeddings)

torch.save(embs, output_file_dir)