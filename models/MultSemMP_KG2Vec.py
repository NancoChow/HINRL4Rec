from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict


class MultSemMP_KG2Vec(nn.Module):

    def __init__(self, dataset, args):

        super(MultSemMP_KG2Vec, self).__init__()

        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda

        self.fusion_linear_layer1 = nn.Linear(100, 100)
        self.fusion_linear_layer2 = nn.Linear(100, 100)
        self.fusion_sigmoid_layer = nn.Sigmoid()

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )

        emb_dir_path = './data/Amazon_Beauty/small/embs'
        output_prod_emb_file_path = os.path.join(emb_dir_path, 'mp2vec_prod_embs.txt')
        output_prod_text_emb_file_path = os.path.join(emb_dir_path, 'bert-text2vec_prod_desc_embs.txt')
        output_user_emb_file_path = os.path.join(emb_dir_path, 'mp2vec_user_embs.txt')

        # load the meta-path-based embeddings
        input_dir_path = './data/Amazon_Beauty/small/rels'
        self.prod_dict = json.load(open(os.path.join(input_dir_path, 'products.json'), 'r', encoding='utf-8'))
        self.prod_reverse_dict = json.load(open(os.path.join(input_dir_path, 'products_reverse.json'), 'r',
                                                encoding='utf-8'))

        self.user_dict = json.load(open(os.path.join(input_dir_path, 'users.json'), 'r', encoding='utf-8'))
        self.user_reverse_dict = json.load(open(os.path.join(input_dir_path, 'users_reverse.json'), 'r',
                                                encoding='utf-8'))

        self.user_metapath_types = ['upu']
        self.prod_metapath_types = ['pcp', 'pbp', 'pp']

        print('Load meta-path-based embeddings for products...')
        self.prod_mp_embs = json.load(open(output_prod_emb_file_path, 'r', encoding='utf-8'))

        print('Load textual embeddings for products...')
        self.prod_text_embs = json.load(open(output_prod_text_emb_file_path, 'r', encoding='utf-8'))

        print('Load meta-path-based embeddings for users...')
        self.user_mp_embs = json.load(open(output_user_emb_file_path, 'r', encoding='utf-8'))

        # pcp meta-path
        self.pcp_embs = nn.Embedding(dataset.product.vocab_size, 100)
        self.pcp_emb_vectors = np.ones((dataset.product.vocab_size + 1, 100))
        self.pbp_emb_vectors = np.ones((dataset.product.vocab_size + 1, 100))
        self.pp_emb_vectors = np.ones((dataset.product.vocab_size + 1, 100))

        for prod_id in self.prod_mp_embs.keys():
            if self.prod_reverse_dict[prod_id] in dataset.product.vocab:
                self.pcp_emb_vectors[int(prod_id)] = self.prod_mp_embs[prod_id]['pcp']
                self.pbp_emb_vectors[int(prod_id)] = self.prod_mp_embs[prod_id]['pbp']
                self.pp_emb_vectors[int(prod_id)] = self.prod_mp_embs[prod_id]['pp']

        self.pcp_emb_vectors = torch.FloatTensor(self.pcp_emb_vectors)
        self.pbp_emb_vectors = torch.FloatTensor(self.pbp_emb_vectors)
        self.pp_emb_vectors = torch.FloatTensor(self.pp_emb_vectors)

        self.prod_emb_vectors = self.pcp_emb_vectors + self.pbp_emb_vectors + self.pp_emb_vectors

        # product textual embedding
        self.prod_text_vectors = np.ones((dataset.product.vocab_size + 1, 100))
        for prod_id in self.prod_mp_embs.keys():
            if self.prod_reverse_dict[prod_id] in dataset.product.vocab:
                if int(prod_id) in self.prod_text_embs:
                    self.prod_text_vectors[int(prod_id)] = self.prod_text_embs[int(prod_id)]

        self.prod_text_vectors = torch.FloatTensor(self.prod_text_vectors)

        # upu meta-path
        self.upu_emb_vectors = np.ones((dataset.user.vocab_size + 1, 100))

        for user_id in self.user_mp_embs.keys():
            if self.user_reverse_dict[user_id] in dataset.user.vocab:
                self.upu_emb_vectors[int(user_id)] = self.user_mp_embs[user_id]['upu']

        self.upu_emb_vectors = torch.FloatTensor(self.upu_emb_vectors)

        self.user_emb_vectors = self.upu_emb_vectors

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
        pr3_loss, pr3_embeds = self.neg_loss('product', 'bought_together', 'related_product', product_idxs,
                                             rproduct3_idxs)
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
        mask = entity_tail_idxs >= 0

        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]

        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding

        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding

        relation_vec = getattr(self, relation)  # [1, embed_size]

        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding

        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return self.kg_neg_loss(entity_head,
                                entity_tail,
                                entity_head_embedding,
                                entity_tail_embedding,
                                fixed_entity_head_idxs,
                                fixed_entity_tail_idxs,
                                relation_vec,
                                relation_bias_embedding,
                                self.num_neg_samples,
                                entity_tail_distrib)

    def kg_neg_loss(self,
                    entity_head,
                    entity_tail,
                    entity_head_embed,
                    entity_tail_embed,
                    entity_head_idxs,
                    entity_tail_idxs,
                    relation_vec,
                    relation_bias_embed,
                    num_samples,
                    distrib):
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
        entity_head_idxs = entity_head_idxs.type(torch.LongTensor)
        entity_tail_idxs = entity_tail_idxs.type(torch.LongTensor)

        batch_size = entity_head_idxs.size(0)

        entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
        if entity_head == 'product':
            mp_entity_head_embedding = self.prod_emb_vectors[entity_head_idxs.numpy().tolist()]
            text_entity_head_embedding = self.prod_text_vectors[entity_head_idxs.numpy().tolist()]
            prod_emb = self.fusion_linear_layer2(text_entity_head_embedding * mp_entity_head_embedding)
            example_vec = self.fusion_linear_layer1(entity_head_vec * prod_emb)
            example_vec = self.fusion_sigmoid_layer(example_vec)
            example_vec = example_vec + relation_vec  # [batch_size, embed_size]
        elif entity_head == 'user':
            mp_entity_head_embedding = self.user_emb_vectors[entity_head_idxs.numpy().tolist()]
            example_vec = self.fusion_linear_layer1(entity_head_vec * mp_entity_head_embedding)
            example_vec = self.fusion_sigmoid_layer(example_vec)
            example_vec = example_vec + relation_vec  # [batch_size, embed_size]
        else:
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
