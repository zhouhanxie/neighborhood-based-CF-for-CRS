import os
import pandas as pd
from itertools import chain
from collections import defaultdict
from collections import Counter
import numpy as np
from tqdm import tqdm
from data_utils import get_reddit_data, get_reddit_data_with_heldout
from transformers import Trainer, TrainingArguments, AutoTokenizer
import torch
import datasets
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from modeling_nmf import NMFDataCollator, NMFConfig, NMF



def main(args=None):
    # relevant dirs, needs clean-up
    base_model_name = args.base_model_name
    pretrained_embedding_path = args.pretrained_embedding_path
    n_item_latent_factors = args.n_item_latent_factors
    output_dir = args.output_dir
    reddit_training_data_path = args.training_data_path
    max_movies = args.max_movies

    # # load data
    # reddit_data, movie_vocab = get_reddit_data(
    #     csv_path=reddit_training_data_path, 
    #     max_movies=max_movies
    #     )
    # hf_dataset = datasets.Dataset.from_dict(reddit_data).train_test_split(test_size=0.2)
    # train_dataset = hf_dataset['train']
    # val_dataset = hf_dataset['test']

    # load data
    reddit_data_train, reddit_data_validation, movie_vocab = get_reddit_data_with_heldout(
        csv_path=reddit_training_data_path, 
        max_movies=max_movies,
        heldout_portion=0.2
        )
    train_dataset = datasets.Dataset.from_dict(reddit_data_train)
    val_dataset = datasets.Dataset.from_dict(reddit_data_validation)

    # model init
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_config = NMFConfig(
        n_items=len(movie_vocab), 
        n_item_latent_factors=n_item_latent_factors, 
        mlm_config_or_model_name_or_path=base_model_name
        )
    model=NMF(model_config)
    # loading pre-initialized weight into the model
    movie_embeddings = torch.load(pretrained_embedding_path)
    model.item_embeddings.weight = torch.nn.Parameter(
        movie_embeddings[:,:model.config.n_item_latent_factors].to(torch.float32)
    )

    # reprocess
    def preprocess_function(example):
        context = example['context']
        inputs = tokenizer(context, return_tensors='pt', max_length=368, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'][0]
        token_type_ids = inputs['token_type_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
    train_dataset = train_dataset.map(preprocess_function, num_proc=16)
    val_dataset = val_dataset.map(preprocess_function, num_proc=16)

    # setup trainer and train
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=30,
        save_strategy = 'epoch',
        evaluation_strategy = 'epoch',
        logging_dir="./logs",
        logging_steps=1,
        save_total_limit=1,
        disable_tqdm=False,  # (do not) Disable tqdm progress bars
        report_to=[],  # Disable logging to any external service,
        load_best_model_at_end = True, # EarlyStoppingCallback requires load_best_model_at_end = True,
        learning_rate = 5e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=NMFDataCollator(tokenizer=tokenizer),
        eval_dataset=val_dataset,
    )

    from transformers import EarlyStoppingCallback

    early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience = 1,
            early_stopping_threshold=1e-2
        )
    trainer.add_callback(early_stopping_callback)

    print('trainer start')
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a NMF model on Reddit data")
    parser.add_argument('--base_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Base model name or path')
    parser.add_argument('--pretrained_embedding_path', type=str, default='semantic_embs_20000.pt', help='Path to pretrained embeddings')
    parser.add_argument('--n_item_latent_factors', type=int, default=16, help='Number of item latent factors')
    parser.add_argument('--output_dir', type=str, default='./saved_models', help='Directory to save the trained model')
    parser.add_argument('--training_data_path', type=str, default='datasets/reddit/reddit_large_train.csv', help='Path to the Reddit training data CSV')
    parser.add_argument('--max_movies', type=int, default=20000, help='Maximum number of movies to consider')
    args = parser.parse_args()

    main(args)