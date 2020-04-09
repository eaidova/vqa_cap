import os
import json
import torch
from torch.autograd import Variable

def test(model, data_loader, output_dir, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    results = []
    with torch.no_grad():
        for i, (v, _, q, _, c, qids) in enumerate(data_loader):
            v = Variable(v).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            c = Variable(c.type(torch.LongTensor)).cuda()
            pred,  _, _ = model(v, q, c)
            _, ix = pred.data.max(1)
            for i, qid in enumerate(qids):
                results.append({
                    'question_id': qid,
                    'answer': data_loader.label2ans[ix[i]]
                })
    json.dump(results, open(os.path.join(output_dir, 'result.json'), 'w'))








