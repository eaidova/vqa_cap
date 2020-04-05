import torch
import torch.nn as nn
from .attention import Att_0, Att_2, Att_3
from .language_model import WordEmbedding, QuestionEmbedding
from .classifier import SimpleClassifier
from .fc import FCNet
from .caption_model import CaptionRNN

import torch.nn.functional as F
from torch.autograd import Variable
# Dropout p: probability of an element to be zeroed. Default: 0.5


class Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class Model_2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,caption_w_emb, reference_caption_decoder, question_caption_decoder,caption_decoder,v2rc_net,v2qc_net):
        super(Model_2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reference_caption_decoder = reference_caption_decoder
        self.question_caption_decoder = question_caption_decoder
        self.caption_w_emb = caption_w_emb
        self.caption_decoder = caption_decoder
        self.v2rc_net = v2rc_net
        self.v2qc_net = v2qc_net

    def forward(self, v, b, q, labels, c):
        """Forward

        v: [batch,5, num_objs, obj_dim]
        b: [batch, 5,num_objs, b_dim]
        q: [batch, 5, seq_length]
        c: [batch, 5, 20 ]

        return: logits, not probs
        """
        
        batch = c.size(0)
        q = q.view(batch * 5, -1)
        c = c.view(batch * 5, -1)
        v = v.view(batch * 5, 36, -1)
        batch = c.size(0)
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]
        att_1 = self.v_att_1(v, q_emb) # [batch* 5, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch* 5, 1, v_dim]
        att = att_1 + att_2
        v_emb = (att * v).sum(1) # [batch, v_dim]
        
        q_repr = self.q_net(q_emb) #[batch * 5 ,hid_dim]
        v_repr = self.v_net(v_emb) #[batch *5, hid_dim]
        joint_repr = q_repr * v_repr #[batch *5,hid_dim ]

        logits = self.classifier(joint_repr)
        rc_w_emb = self.caption_w_emb(c)
        qc_w_emb = self.caption_w_emb(c) # [batch * 5, 20 , hid_dim]

        v_rc = self.v2rc_net(v)
        v_qc = self.v2qc_net(joint_repr)

        rc_emb = self.reference_caption_decoder(rc_w_emb, v_rc)
        qc_emb = self.question_caption_decoder(v_qc ,qc_w_emb)
        rc_repr = self.caption_decoder(rc_emb)
        qc_repr = self.caption_decoder(qc_emb)
        
        return logits, rc_repr, qc_repr


class Model_4(nn.Module):
    def __init__(self, w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,caption_w_emb, reference_caption_decoder, question_caption_decoder,caption_decoder,v2rc_net, v2qc_net):
        super(Model_4, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att_1 = v_att_1
        self.v_att_2 = v_att_2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reference_caption_rnn = reference_caption_decoder
        self.question_caption_rnn = question_caption_decoder
        self.caption_w_emb = caption_w_emb
        self.caption_decoder = caption_decoder
        self.v2rc_net = v2rc_net
        self.v2qc_net = v2qc_net

    def forward(self, v, b, q, labels, c):
        """Forward

        v: [batch,5, num_objs, obj_dim]
        b: [batch, 5,num_objs, b_dim]
        q: [batch, 5, seq_length]
        c: [batch, 5, 20 ]

        return: logits, not probs
        """
        batch = c.size(0)
        q = q.view(batch * 5, -1)
        c = c.view(batch * 5, -1)
        v = v.view(batch * 5, 36, -1)
        batch = c.size(0)
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]
        att_1 = self.v_att_1(v, q_emb) # [batch* 5, 1, v_dim]
        att_2 = self.v_att_2(v, q_emb)  # [batch* 5, 1, v_dim]
        att = att_1 + att_2
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb) #[batch * 5 ,hid_dim]
        v_repr = self.v_net(v_emb)#[batch *5, hid_dim]
        joint_repr = q_repr * v_repr #[batch *5,hid_dim ]

        logits = self.classifier(joint_repr)
        
        rc_w_emb = self.caption_w_emb(c)

        v_rc = self.v2rc_net(v.mean(1))
        v_qc = self.v2qc_net(joint_repr)

        rc_emb = self.reference_caption_rnn( v_rc,rc_w_emb)
        rc_repr = self.caption_decoder(rc_emb)

        pred_ans = F.sigmoid(logits).contiguous()
        pred_rc = F.sigmoid(rc_repr).contiguous()
        
        batch = batch / 5
        
        caption_from_ans =  pred_rc[:, : , : 3129 ]

        caption_from_ans = caption_from_ans.contiguous().view(batch, 1 ,5, 20, -1).repeat(1,5,1,1,1)
        
        similarities_ = (caption_from_ans * (pred_ans.view(batch, 5,1,1,-1).repeat(1, 1, 5, 20, 1))).sum(4)
        similarities, _ = similarities_.max(3)
        _, indices = similarities.max(2)
        indices = indices.view(-1,1 )
        target_qc_mask = torch.zeros(batch*5, 5)
        target_qc_mask.scatter_(1, indices.data.type(torch.LongTensor), 1)
        target_qc_mask = Variable(target_qc_mask.view(batch, 5, 5, 1).repeat(1,1,1,20).type(torch.LongTensor)).cuda()
        target_qc = c.view(batch,1,5,20).repeat(1,5,1,1)
        target_qc = target_qc * target_qc_mask
        target_qc = target_qc.sum(2).view(-1, 20)
        qc_w_emb = self.caption_w_emb(target_qc) # [batch * 5, 20 , hid_dim]
        qc_emb = self.question_caption_rnn(v_qc ,qc_w_emb)
        qc_repr = self.caption_decoder(qc_emb)
        pred_qc = F.sigmoid(qc_repr).contiguous()
        
        return logits, pred_rc, pred_qc, target_qc

# Attn: 1 layer attention, output layer, softmax
def build_baseline(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att = Att_0(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)

    return Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

# 2*Attn: 1 layer seperate, element-wise *, 1 layer, output layer, softmax
# our adopted model
# (self, in_dim, num_hid, v_dim, nlayers, bidirect, dropout, rnn_type='LSTM'):
# (self, embed_size, hidden_size, vocab_size, num_layers):
def build_model_A3x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_3(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)
    # num_hid = 1280 , dataset.v_dim = 2048
    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)
        
    v2rc_net = FCNet([dataset.v_dim, 300 ], dropout= dropL, norm= norm, act= activation)
    v2qc_net = FCNet([num_hid, 300], dropout= dropL, norm= norm, act= activation)

    caption_w_emb = WordEmbedding(dataset.caption_dictionary.ntoken, emb_dim=300, dropout=dropW)
    reference_caption_decoder = CaptionRNN(300, 512,  num_layers = 1 )
    question_caption_decoder = CaptionRNN(300, 512, num_layers = 1 )
    caption_decoder = SimpleClassifier( in_dim=512, hid_dim=2 * num_hid, out_dim= dataset.caption_dictionary.ntoken, dropout=dropC, norm= norm, act= activation)

    return Model_4(
        w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,
        caption_w_emb, reference_caption_decoder, question_caption_decoder, caption_decoder, v2rc_net, v2qc_net
    )


# 2*Attn: 1 layer seperate, element-wise *, output layer, softmax
def build_model_A2x2(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=dropW)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=dropG, rnn_type='GRU')

    v_att_1 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    v_att_2 = Att_2(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                    act=activation)
    q_net = FCNet([q_emb.num_hid, num_hid], dropout= dropL, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= dropL, norm= norm, act= activation)
    caption_w_emb = WordEmbedding(dataset.caption_dictionary.ntoken, emb_dim=300, dropout=dropW)
    reference_caption_decoder = CaptionRNN(300, 512, num_layers=1)
    question_caption_decoder = CaptionRNN(300, 512, num_layers=1)
    caption_decoder = SimpleClassifier(in_dim=512, hid_dim=2 * num_hid, out_dim=dataset.caption_dictionary.ntoken,
                                       dropout=dropC, norm=norm, act=activation)
    v2rc_net = FCNet([dataset.v_dim, 300 ], dropout= dropL, norm= norm, act= activation)
    v2qc_net = FCNet([num_hid, 300], dropout= dropL, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=dropC, norm= norm, act= activation)

    return Model_2(
        w_emb, q_emb, v_att_1, v_att_2, q_net, v_net, classifier,
        caption_w_emb, reference_caption_decoder, question_caption_decoder, caption_decoder, v2rc_net, v2qc_net
    )


MODELS = {
    'A2x2': build_model_A2x2,
    'A3x2': build_model_A3x2,
    'baseline': build_baseline
}


def build_model(model_config, dataset):
    model_type = model_config['type']
    model_builder = MODELS.get(model_type)
    if model_builder is None:
        raise ValueError('Wrong model type {}'.format(model_type))
    num_hid = model_config['num_hid']
    norm = model_config['norm']
    activation = model_config['activation']
    train_config = model_config['train']
    dropout = train_config['dropout']
    dropL = train_config['dropout_l']
    dropG = train_config['dropout_g']
    dropW = train_config['dropout_w']
    dropC = train_config['dropout_c']

    return model_builder(dataset, num_hid, dropout, norm, activation, dropL , dropG, dropW, dropC)