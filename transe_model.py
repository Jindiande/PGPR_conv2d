from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from data_utils import AmazonDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.conv2d_model=ConvE(args)

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            purchase=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            mentions=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            describe_as=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            produced_by=edict(
                et='brand',
                et_distrib=self._make_distrib(dataset.produced_by.et_distrib)),
            belongs_to=edict(
                et='category',
                et_distrib=self._make_distrib(dataset.belongs_to.et_distrib)),
            also_bought=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_bought.et_distrib)),
            also_viewed=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_viewed.et_distrib)),
            bought_together=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.bought_together.et_distrib)),
        )
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        user_idxs = batch_idxs[:, 0]
        product_idxs = batch_idxs[:, 1]
        word_idxs = batch_idxs[:, 2]
        brand_idxs = batch_idxs[:, 3]
        category_idxs = batch_idxs[:, 4]
        rproduct1_idxs = batch_idxs[:, 5]
        rproduct2_idxs = batch_idxs[:, 6]
        rproduct3_idxs = batch_idxs[:, 7]

        regularizations = []

        # user + purchase -> product
        up_loss, up_embeds = self.neg_loss('user', 'purchase', 'product', user_idxs, product_idxs)
        regularizations.extend(up_embeds)
        loss = up_loss

        # user + mentions -> word
        uw_loss, uw_embeds = self.neg_loss('user', 'mentions', 'word', user_idxs, word_idxs)
        regularizations.extend(uw_embeds)
        loss += uw_loss

        # product + describe_as -> word
        pw_loss, pw_embeds = self.neg_loss('product', 'describe_as', 'word', product_idxs, word_idxs)
        regularizations.extend(pw_embeds)
        loss += pw_loss

        # product + produced_by -> brand
        pb_loss, pb_embeds = self.neg_loss('product', 'produced_by', 'brand', product_idxs, brand_idxs)
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        # product + belongs_to -> category
        pc_loss, pc_embeds = self.neg_loss('product', 'belongs_to', 'category', product_idxs, category_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss

        # product + also_bought -> related_product1
        pr1_loss, pr1_embeds = self.neg_loss('product', 'also_bought', 'related_product', product_idxs, rproduct1_idxs)
        if pr1_loss is not None:
            regularizations.extend(pr1_embeds)
            loss += pr1_loss

        # product + also_viewed -> related_product2
        pr2_loss, pr2_embeds = self.neg_loss('product', 'also_viewed', 'related_product', product_idxs, rproduct2_idxs)
        if pr2_loss is not None:
            regularizations.extend(pr2_embeds)
            loss += pr2_loss

        # product + bought_together -> related_product3
        pr3_loss, pr3_embeds = self.neg_loss('product', 'bought_together', 'related_product', product_idxs, rproduct3_idxs)
        if pr3_loss is not None:
            regularizations.extend(pr3_embeds)
            loss += pr3_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):

        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        #print("entity_head_idxs",entity_head_idxs.size(),"entity_tail_idxs",entity_tail_idxs.size())
        entity_head_idxs=Variable(entity_head_idxs)
        entity_tail_idxs = Variable(entity_tail_idxs)
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]#[B 1]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding [vocab_size+1 embed_size]
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding [vocab_size+1 embed_size]
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding [vocab_size + 1, 1]
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss_conv2d(self.conv2d_model,entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, self.num_neg_samples,entity_tail_distrib)

def kg_neg_loss_conv2d(model,entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, num_samples, distrib):
    # conv2d method using 2d convolution operation to replace the plus operation between
    # the head vector and relation vector while not changing loss function in transe(distance
    # of both positive and negative data pair)


    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]

    relation_vec=relation_vec.repeat(batch_size,1)# [batch_size, embed_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    if(entity_head_vec.size(0)==1):
        loss_positive,_ = model(entity_head_vec[0,:].repeat(2,1), relation_vec[0,:].repeat(2,1), entity_tail_vec[0,:].repeat(2,1))
    else:
        loss_positive,_=model(entity_head_vec,relation_vec,entity_tail_vec)

    mindim=min(num_samples,entity_head_vec.size(0))
    if(mindim==1):
        loss_nege,_ = model(entity_head_vec[0:mindim, :].repeat(2,1), relation_vec[0:mindim, :].repeat(2,1), neg_vec[0:mindim, :].repeat(2,1))
    else:
        loss_nege,_=model(entity_head_vec[0:mindim,:],relation_vec[0:mindim,:],neg_vec[0:mindim,:])
    loss=loss_positive-0.1*loss_nege

    #loss=loss_positive
    return loss,[entity_head_vec, entity_tail_vec, neg_vec]
def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]
class ConvE(torch.nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        #self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        #self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.emb_dim1 = 10
        self.emb_dim2 = 100 // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embed_size)
        #self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(32*18*8,args.embed_size)



    def forward(self, e1, rel,e2):
        e1= e1.view(-1, 1, self.emb_dim1, self.emb_dim2)#[batch_size,1, emb_dim1,emb_dim2]
        rel = rel.view(-1, 1, self.emb_dim1, self.emb_dim2)#[batch_size,1, emb_dim1,emb_dim2]

        stacked_inputs = torch.cat([e1, rel], 2)#[batch_size,1, emb_dim1*2,emb_dim2]

        stacked_inputs = self.bn0(stacked_inputs)#[batch_size,1, emb_dim1*2,emb_dim2]
        x= self.inp_drop(stacked_inputs)#[batch_size,1, emb_dim1*2,emb_dim2]
        x= self.conv1(x)#[B 32 18 8]
        x= self.bn1(x)#[B 32 18 8]
        x= F.relu(x)#[B 32 18 8]
        x = self.feature_map_drop(x)#[B 32 18 8]
        x = x.view(x.shape[0], -1)#[B 32*18*8]
        x = self.fc(x)#[B 100]
        x = self.hidden_drop(x)#[B 100]
        x = self.bn2(x)#[B 100]
        x = F.relu(x)#[B 100]
        if self.training:
           loss=self.loss(x,e2)
        else:
            loss=0
        #x = torch.mm(x, self.emb_e.weight.transpose(1,0))#[b num_entities]
        #x += self.b.expand_as(x)#[]
        #pred = torch.sigmoid(x)

        return loss,x
