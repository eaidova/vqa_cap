import os
import time
from functools import partial
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
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
    return optim(model_params, weights_decay=weights_decay)


def resume_train(model, output, optim):
    resume_path = os.path.join(output, 'init_model.pth')
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optim.load_state_dict(checkpoint['optimizer'])


def train(model, train_loader, eval_loader, output, train_config):
    num_epochs = train_config['num_epochs']
    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    optim = select_optimizer(model.parameters(), train_config['optimizer'], train_config['weights_decay'])
    for epoch in range(num_epochs):

        total_loss = 0
        train_score = 0
        t = time.time()
        for i, (v, b, q, a, c) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            a = Variable(a).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            c = Variable(c.type(torch.LongTensor)).cuda()
            pred, pred_rc , pred_qc ,target_qc= model(v, b, q, a, c )

            loss_ans = instance_bce_with_logits(pred.view(-1, pred.size(-1)), a.view(-1, a.size(-1)))

            loss_rc = nn.NLLLoss()(torch.log(pred_rc.view(-1, pred_rc.size(-1))), c.view(-1))
            loss_qc =  nn.NLLLoss()(torch.log(pred_qc.view(-1, pred_qc.size(-1))), target_qc.view(-1))
            
            loss = loss_ans + loss_rc + loss_qc
            if np.mod(i, 100) == 0:
                
                mini_batch_score = compute_score_with_logits( pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data).sum()
                print(i, loss.data[0], mini_batch_score)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits( pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= (len(train_loader.dataset)*5)
        train_score = 100 * train_score / (len(train_loader.dataset)*5)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
        }, filename=output + '/epoch_%d_model.pth'%epoch  )

        model.train(False)
        eval_score, bound, V_loss = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.3f, score: %.3f' % (total_loss, train_score))
        logger.write('\teval loss: %.3f, score: %.3f (%.3f)' % (V_loss, 100 * eval_score, 100 * bound))

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

    for v, b, q, a , c in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        a = Variable(a, volatile=True).cuda()
        q = Variable(q.type(torch.LongTensor), volatile=True).cuda()
        c = Variable(c.type(torch.LongTensor), volatile=True).cuda()
        
        pred, pred_rc , pred_qc ,target_qc= model(v, b, q, a, c )
        loss = instance_bce_with_logits(pred.view(-1, pred.size(-1)), a.view(-1, a.size(-1)))
        V_loss += loss.data[0] * v.size(0)
        batch_score = compute_score_with_logits( pred.view(-1,pred.size(-1)), a.view(-1, a.size(-1)).data).sum()
        score += batch_score
        upper_bound += (a.view(-1, a.size(-1)).max(1)[0]).sum()
        num_data += pred.size(0)

    score /=  (len(dataloader.dataset ) * 5)
    V_loss /= (len(dataloader.dataset ) * 5)
    upper_bound /= (len(dataloader.dataset ) * 5)

    return score, upper_bound, V_loss
