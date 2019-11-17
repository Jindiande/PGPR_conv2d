from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import pickle
from easydict import EasyDict as edict
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import *
from transe_model import KnowledgeEmbedding,ConvE
using_method_transE=False
using_method='tranE' if using_method_transE else 'conv_2d'



def load_embedding(args):
    state_dict = torch.load(args.model_file,map_location='cuda:0')
    state_dict=state_dict['state_dict']
    user_embed = state_dict['user.weight'].cpu()
    product_embed = state_dict['product.weight'].cpu()
    purchase_embed = state_dict['purchase'].cpu()
    purchase_bias = state_dict['purchase_bias.weight'].cpu()
    #con2d_model=state_dict['conv2d_model']
    results = edict(
            user_embed=user_embed.data.numpy(),
            product_embed=product_embed.data.numpy(),
            purchase_embed=purchase_embed.data.numpy(),
            purchase_bias=purchase_bias.data.numpy(),

    )
    output_file = '{}/{}_embedding.pkl'.format(args.dataset_dir, args.dataset)
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)


def load_train_reviews(args):
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(args.train_review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    output_file = '{}/{}_train_label.pkl'.format(args.dataset_dir, args.dataset)
    with open(output_file, 'wb') as f:
        pickle.dump(user_products, f)


def load_test_reviews(args):
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(args.test_review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    output_file = '{}/{}_test_label.pkl'.format(args.dataset_dir, args.dataset)
    with open(output_file, 'wb') as f:
        pickle.dump(user_products, f)


def test(args, topk=10):
    #embed_file = '{}/{}_embedding.pkl'.format(args.dataset_dir, args.dataset)
    embed_file='/content/drive/My Drive/PGPR_conv2d/tmp/Amazon_Beauty/Amazon_Beauty_embedding.pkl'
    with open(embed_file, 'rb') as f:
        embeddings = pickle.load(f)

    #train_labels_file = '{}/{}_train_label.pkl'.format(args.dataset_dir, args.dataset)
    train_labels_file='/content/drive/My Drive/PGPR_conv2d/tmp/Amazon_Beauty/train_label.pkl'
    with open(train_labels_file, 'rb') as f:
        train_user_products = pickle.load(f)

    #test_labels_file = '{}/{}_test_label.pkl'.format(args.dataset_dir, args.dataset)
    test_labels_file='/content/drive/My Drive/PGPR_conv2d/tmp/Amazon_Beauty/test_label.pkl'
    with open(test_labels_file, 'rb') as f:
        test_user_products = pickle.load(f)
    test_user_idxs = list(test_user_products.keys())
    # print('Num of users:', len(user_idxs))
    # print('User:', user_idxs[0], 'Products:', user_products[user_idxs[0]])

    user_embed = embeddings['user_embed'][:-1]  # remove last dummy user[N1 b]
    purchase_embed = embeddings['purchase_embed']# [1 b]
    product_embed = embeddings['product_embed'][:-1]# [N2 b]
    print('user embed:', user_embed.shape, 'purchase_embed',purchase_embed.shape,'product embed:', product_embed.shape)

    scores_matrix=[]
    if(using_method=='conv_2d'):
        state_dict = torch.load(args.model_file, map_location='cuda:0')
        state_dict = state_dict['state_dict']
        conv2d_model = ConvE(args)
        print('state_dict', state_dict.keys())
        print('conv2d_model', conv2d_model.state_dict().keys())
        for name, param in conv2d_model.state_dict().items():
            pretrain_data = state_dict['conv2d_model.' + name].data
            conv2d_model.state_dict()[name].copy_(pretrain_data)

        print('load over')

        # test conv2d model
        user_embed = torch.from_numpy(user_embed)
        purchase_embed = torch.from_numpy(purchase_embed)  # [N1 b]
        product_embed = torch.from_numpy(product_embed)  # [N2 b]
        purchase_embed = purchase_embed.repeat(user_embed.size(0), 1)
        conv2d_model.eval()
        _, scores_matrix = conv2d_model(user_embed, purchase_embed, product_embed,e2=None)  # [N1 b]
       # multi_matrix = torch.mm(calulated_product_emb, product_embed.transpose(1, 0).contiguous())  # [N1 N2]
        scores_matrix=scores_matrix.detach().numpy()
        """
        calulated_product_emb_sum_sq = torch.mul(calulated_product_emb, calulated_product_emb).sum(dim=1).view(
            calulated_product_emb.size(0), 1).repeat(1, product_embed.size(0))  # [N1 N2]
        product_embed_sum_sq = torch.mul(product_embed, product_embed).sum(dim=1).view(product_embed.size(0), 1).repeat(
            1, calulated_product_emb.size(0))  # [N2 N1]
        print(multi_matrix.shape, calulated_product_emb_sum_sq.shape, product_embed_sum_sq.shape)
        scores_matrix = calulated_product_emb_sum_sq + product_embed_sum_sq.transpose(1, 0) - 2 * multi_matrix
        scores_matrix = -1 * scores_matrix.detach().numpy()
        """
    else:
        # transE
        calulated_product_emb = user_embed + purchase_embed
        scores_matrix = np.dot(calulated_product_emb, product_embed.T)








    print('scores_matrix.shape',scores_matrix.shape)
    #print('Max score:', np.max(scores_matrix),torch.max(calulated_product_emb),torch.max(user_embed),torch.max(product_embed),torch.max(purchase_embed))

    # normalize embeddings(TBD)
    # norm_calulated_product_emb = calulated_product_emb/LA.norm(calulated_product_emb, axis=1, keepdims=True)
    # norm_product_embed = product_embed/LA.norm(product_embed, axis=1, keepdims=True)
    # scores_matrix = np.dot(norm_calulated_product_emb, np.transpose(norm_product_embed))
    # print (scores_matrix.shape)

    # filter the test data item which trained in train data
    idx_list = []
    for uid in train_user_products:
        pids = train_user_products[uid]
        tmp = list(zip([uid] * len(pids), pids))
        idx_list.extend(tmp)
    idx_list = np.array(idx_list)
    scores_matrix[idx_list[:, 0], idx_list[:, 1]] = -99

    if scores_matrix.shape[1] <= 30000:
        top_matches = np.argsort(scores_matrix)  # sort row by row
        topk_matches = top_matches[:, -topk:]  # user-product matrix, from lowest rank to highest
    else:  # sort in batch way
        topk_matches = np.zeros((scores_matrix.shape[0], topk), dtype=np.int)
        i = 0
        while i < scores_matrix.shape[0]:
            start_row = i
            end_row = np.min([i + 100, scores_matrix.shape[0]])
            batch_scores = scores_matrix[start_row:end_row, :]
            matches = np.argsort(batch_scores)
            topk_matches[start_row:end_row] = matches[:, -topk:]
            i = end_row

    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    for uid in test_user_idxs:
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon_Beauty',
                        help='One of {Beauty, CDs_Vinyl, Cellphones_Accessories, Movies_TV, Clothing}')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    args = parser.parse_args()
    args.dataset_dir = '/content/drive/My Drive/PGPR_conv2d/tmp/{}'.format(args.dataset)
    args.dataset_file = args.dataset_dir + '/dataset.pkl'
    """
    model_files = {
        'Beauty': args.dataset_dir + '/train_embedding_final/transe_model_sd_epoch_2_0.050522819660864006.ckpt',
        
        'Cellphones_Accessories': args.dataset_dir + '/train_embedding_final/embedding_des_epoch_30.ckpt',
        'Clothing': args.dataset_dir + '/train_embedding_final/embedding_des_epoch_29.ckpt',
        'CDs_Vinyl': args.dataset_dir + '/train_embedding/embedding_epoch_20.ckpt',
        
    }
    """
    model_files = {
        'Amazon_Beauty': args.dataset_dir + '/train_transe_model/transe_model_sd_epoch_6_29.0.ckpt',

    }
    args.model_file = model_files[args.dataset]
    """
    review_dir = {
        'Beauty': './data/CIKM2017/reviews_Beauty_5.json.gz.stem.nostop/min_count5/query_split',
        'CDs_Vinyl': './data/CIKM2017/reviews_CDs_and_Vinyl_5.json.gz.stem.nostop/min_count5/query_split',
        'Cellphones_Accessories': './data/CIKM2017/reviews_Cell_Phones_and_Accessories_5.json.gz.stem.nostop/min_count5/query_split',
        'Movies_TV': './data/CIKM2017/reviews_Movies_and_TV_5.json.gz.stem.nostop/min_count5/query_split',
        'Clothing': './data/CIKM2017/reviews_Clothing_Shoes_and_Jewelry_5.json.gz.stem.nostop/min_count5/query_split',
    }
    """
    review_dir = {
        'Amazon_Beauty': './data/CIKM2017/reviews_Beauty_5.json.gz.stem.nostop/min_count5/query_split',

    }
    args.train_review_file = review_dir[args.dataset] + '/train.txt.gz'
    args.test_review_file = review_dir[args.dataset] + '/test.txt.gz'

    load_embedding(args)
    #load_train_reviews(args)
    #load_test_reviews(args)

    test(args)


if __name__ == '__main__':
    main()
