import pandas as pd
from itertools import chain
from collections import defaultdict
from collections import Counter
import numpy as np
from tqdm import tqdm

def get_reddit_data(csv_path='datasets/reddit/reddit_large_train.csv', max_movies=20000):
    """
    Misc preprocessing for loading the dataset into training format.

    Returns:
        reddit_data: context and labels from the training split
        movie_vocab: most frequent N=max_movies movies
    """

    reddit_posts_train = pd.read_csv(csv_path)
    eligible_movies = [eval(i) for i in reddit_posts_train['movies']]
    frequency = Counter(list(chain.from_iterable(eligible_movies)))
    most_freq_N = max_movies
    print(f'interaction preservance rate at {most_freq_N} items')
    print(sum(i[1] for i in frequency.most_common(most_freq_N)) / sum(i[1] for i in frequency.items()))

    unique_movie_names = sorted(i[0] for i in frequency.most_common(most_freq_N))
    movie2id = {movie: idx for idx, movie in enumerate(unique_movie_names)}
    id2movie = {idx:movie for movie,idx in movie2id.items()}
    movie_vocab = unique_movie_names

    print('num items ', len(movie2id))
    positive_samples = []
    for post in eligible_movies:
        filtered_post = [movie for movie in post if movie in movie2id]
        positive_samples.append([movie2id[movie] for movie in filtered_post])

    reddit_data = {
        'context':[],
        'label':[]
    }

    # full situation is the entire post (there is another field that has post title only
    contexts = list(reddit_posts_train['full_situation'])
    print('flattening posts into training data')

    for i in tqdm(range(len(positive_samples)), total = len(positive_samples)):
        context = contexts[i]

        for positive_sample in positive_samples[i]:
            reddit_data['context'].append(context)
            reddit_data['label'].append(positive_sample)

    return reddit_data, movie_vocab


def get_reddit_data_with_heldout(
    csv_path='datasets/reddit/reddit_large_train.csv', 
    max_movies=20000,
    heldout_portion=0.2
    ):
    """
    Misc preprocessing for loading the dataset into training format.

    Returns:
        reddit_data: context and labels from the training split
        movie_vocab: most frequent N=max_movies movies
    """

    reddit_posts_train = pd.read_csv(csv_path)
    eligible_movies = [eval(i) for i in reddit_posts_train['movies']]
    frequency = Counter(list(chain.from_iterable(eligible_movies)))
    most_freq_N = max_movies
    print(f'interaction preservance rate at {most_freq_N} items')
    print(sum(i[1] for i in frequency.most_common(most_freq_N)) / sum(i[1] for i in frequency.items()))

    unique_movie_names = sorted(i[0] for i in frequency.most_common(most_freq_N))
    movie2id = {movie: idx for idx, movie in enumerate(unique_movie_names)}
    id2movie = {idx:movie for movie,idx in movie2id.items()}
    movie_vocab = unique_movie_names

    print('num items ', len(movie2id))
    positive_samples = []
    for post in eligible_movies:
        filtered_post = [movie for movie in post if movie in movie2id]
        positive_samples.append([movie2id[movie] for movie in filtered_post])

    reddit_data_train = {
        'context':[],
        'label':[]
    }
    reddit_data_validation = {
        'context':[],
        'label':[]
    }
    split_point = int(len(reddit_posts_train['full_situation'])*(1-heldout_portion))

    # full situation is the entire post (there is another field that has post title only
    contexts = list(reddit_posts_train['full_situation'])
    print('flattening posts into training data')

    for i in tqdm(range(split_point), total = split_point):
        context = contexts[i]

        for positive_sample in positive_samples[i]:
            reddit_data_train['context'].append(context)
            reddit_data_train['label'].append(positive_sample)

    for i in tqdm(range(split_point, len(reddit_posts_train)), total = len(reddit_posts_train)-split_point):
        context = contexts[i]

        for positive_sample in positive_samples[i]:
            reddit_data_validation['context'].append(context)
            reddit_data_validation['label'].append(positive_sample)

    return reddit_data_train, reddit_data_validation, movie_vocab