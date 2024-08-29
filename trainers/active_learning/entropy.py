import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL

class Entropy(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        
    def run(self, n_query):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.model.eval()
        with torch.no_grad():
            # selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            selection_loader = build_data_loader(
                self.cfg, 
                data_source=self.unlabeled_set, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            scores = np.array([])
            
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                
                preds = self.model(inputs, get_feature=False)
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                scores = np.append(scores, entropys)
                
        return scores

    def select(self, n_query, **kwargs):
        selected_indices, scores = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index

  
class BudgetSaving(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 

    def rank_uncertainty(self):
        self.model.eval()
        with torch.no_grad():
            # selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            selection_loader = build_data_loader(
                self.cfg, 
                data_source=self.unlabeled_dst, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            scores = np.array([])
            self.preds = []
            
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                
                preds = self.model(inputs, get_feature=False)
                preds = torch.nn.functional.softmax(preds, dim=1)
                #entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                entropys = torch.sum(torch.special.entr(preds), dim=1).cpu().numpy()
                scores = np.append(scores, entropys)
                self.preds = np.append(self.preds, np.argmax(preds.cpu().numpy(), axis=1))

        self.preds = self.preds.astype(int)
                
        return scores

    def budget_save(self, Q_index, p_thres=None, **kwargs):
        scores = self.rank_uncertainty()

        p_thres_ent = np.percentile(scores, p_thres * 100)
        P_index = [idx for idx in Q_index if scores[idx] < p_thres_ent]
        matched = 0
        # Pseudo Labeling
        for i in P_index:
            if self.unlabeled_dst[i]._label == self.preds[i]:
                matched += 1
            self.unlabeled_dst[i]._label = self.preds[i]
        self.unlabeled_dst
        if len(P_index) == 0:
            p_acc = None
        else:
            p_acc = matched / len(P_index)

        return P_index, p_acc
    