import os
import time
import datetime
from functools import partial
import torch
import torch.nn as nn
from utils import common_utils as utils
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

OPTIMIZERS = {
    'Adadelta': partial( torch.optim.Adadelta, rho=0.95, eps=1e-6),
    'RMSprop': partial(torch.optim.RMSprop, lr=0.01, alpha=0.99, eps=1e-08, momentum=0, centered=False),
    'Adam': partial(torch.optim.Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08),
    'Adamax': torch.optim.Adamax
}


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def select_optimizer(model_params, opt, weights_decay):
    optim = OPTIMIZERS[opt]
    return optim(model_params)


def resume_train(model, checkpoint, optim):
    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer'])


def train(model, train_loader, eval_loader, output, train_config, checkpoint=None):
    writer = SummaryWriter('{}/logs/{}'.format(output, datetime.datetime.now()))
    num_epochs = train_config['epochs']
    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    optim = select_optimizer(model.parameters(), train_config['optimizer'], train_config['weights_decay'])
    if checkpoint:
        resume_train(model, checkpoint, optim)

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        for i, (v, _, q, a, c, _) in enumerate(train_loader):
            model.train()
            v = Variable(v).cuda()
            a = Variable(a).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            c = Variable(c.type(torch.LongTensor)).cuda()
            pred, pred_rc , pred_qc ,target_qc= model(v,q, c)

            loss_ans = instance_bce_with_logits(pred.view(-1, pred.size(-1)), a.view(-1, a.size(-1)))

            loss_rc = nn.NLLLoss()(torch.log(pred_rc.view(-1, pred_rc.size(-1))), c.view(-1))
            loss_qc =  nn.NLLLoss()(torch.log(pred_qc.view(-1, pred_qc.size(-1))), target_qc.view(-1))
            
            loss = loss_ans + loss_rc + loss_qc
            if np.mod(i, 100) == 0:
                mini_batch_score = compute_score_with_logits(
                    pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data
                ).sum()
                logger.write('epoch {} step {} mini_batch_score {}'.format(epoch, i, mini_batch_score))

            writer.add_scalar('Train/loss', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Train/loss_answer', loss_ans.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Train/loss_question_captioning', loss_qc.item(), epoch * len(train_loader) + i)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits( pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data).sum()
            writer.add_scalar('Train/mini_batch_score', batch_score, epoch * len(train_loader) + i)
            total_loss += loss.data * v.size(0)
            train_score += batch_score

        total_loss /= (len(train_loader.dataset)*5)
        train_score = 100 * train_score / (len(train_loader.dataset)*5)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
        }, filename=output + '/epoch_{}_model.pth'.format(output))

        model.eval()
        eval_score, bound, V_loss = evaluate(model, eval_loader)
        logger.write('epoch {}, time: {.2f}'.format(epoch, time.time()-t))
        logger.write('\ttrain_loss: {.3f}, score: {.3f}'.format(total_loss, train_score))
        logger.write('\teval loss: {.3f}, score: {.3f} ({.3f})'.format(V_loss, 100 * eval_score, 100 * bound))
        writer.add_scalar('Train/total_loss', total_loss, epoch)
        writer.add_scalar('Train/total_score', train_score)
        writer.add_scalar('Eval/loss', V_loss, epoch)
        writer.add_scalar('Eval/score', 100 * eval_score, epoch)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
            }, model_path)
            
            best_eval_score = eval_score


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def evaluate(model, dataloader):
    score = 0
    V_loss = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad():
        for v, _, q, a , c, _ in iter(dataloader):
            v = Variable(v).cuda()
            a = Variable(a).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            c = Variable(c.type(torch.LongTensor)).cuda()

            pred, pred_rc, pred_qc, target_qc = model(v, q, c)
            loss = instance_bce_with_logits(pred.view(-1, pred.size(-1)), a.view(-1, a.size(-1)))
            V_loss += loss.data * v.size(0)
            batch_score = compute_score_with_logits( pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data).sum()
            score += batch_score
            upper_bound += (a.view(-1, a.size(-1)).max(1)[0]).sum()
            num_data += pred.size(0)

        score /= (len(dataloader.dataset) * 5)
        V_loss /= (len(dataloader.dataset) * 5)
        upper_bound /= (len(dataloader.dataset) * 5)

    return score, upper_bound, V_loss
