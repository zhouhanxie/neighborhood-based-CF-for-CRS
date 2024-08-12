# python -u train_knnlm.py \
#     --pretrained_embedding_path 'semantic_embs_inspired.pt' \
#     --n_item_latent_factors 16 \
#     --output_dir './models/inspired' \
#     --training_data_path 'datasets/inspired/inspired_train.csv' \
#     --max_movies 20000

# python -u train_knnlm.py \
#     --pretrained_embedding_path 'semantic_embs_redial.pt' \
#     --n_item_latent_factors 16 \
#     --output_dir ./models/redial \
#     --training_data_path 'datasets/redial/redial_train.csv' \
#     --max_movies 20000

python -u train_knnlm.py \
    --pretrained_embedding_path 'semantic_embs_reddit.pt' \
    --n_item_latent_factors 16 \
    --output_dir ./models/reddit \
    --training_data_path 'datasets/reddit/reddit_large_train.csv' \
    --max_movies 20000