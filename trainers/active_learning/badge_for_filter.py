import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from tqdm import tqdm

from .AL import AL


class BADGE(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device 
        self.pred = []
        
    def get_grad_features(self):
        self.pred = []
        self.model.eval()

        if torch.cuda.device_count() > 1:
            self.n_class_desc = self.model.module.n_class_desc
            self.weighted_sum_weight = self.model.module.weighted_sum.w.detach()
        else:
            self.n_class_desc = self.model.n_class_desc
            self.weighted_sum_weight = self.model.weighted_sum.w.detach()

        softmaxed_weight = []
        start = 0
        for n in self.n_class_desc:
            same_class_softmaxed_weight = torch.nn.functional.softmax(self.weighted_sum_weight[start: start + n], dim=0)
            softmaxed_weight.extend(same_class_softmaxed_weight)
            start += n
        
        desc_label = []
        for i, n in enumerate(self.n_class_desc):
            desc_label.extend([i] * n)  # [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, ...]

        # embDim = self.model.image_encoder.attnpool.c_proj.out_features
        if self.cfg.MODEL.BACKBONE.NAME != "RN50":
            embDim = 512
        else:
            embDim = 1024
        num_unlabeled = len(self.U_index)
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"
        grad_embeddings = torch.zeros([num_unlabeled, embDim * sum(self.n_class_desc)]) # 
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            # unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            #                                                num_workers=self.cfg.DATALOADER.NUM_WORKERS)
            # generate entire unlabeled features set
            
            print("Getting grad features...")
            for i, batch in enumerate(tqdm(unlabeled_loader)):
                inputs = batch["img"].to(self.device)
                out, features = self.model(inputs, get_feature=True)
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)
                # _, preds = torch.max(out.data, 1)
                self.pred.append(maxInds.detach().cpu())
                
                for j in range(len(inputs)):
                    for c in range(sum(self.n_class_desc)):
                        #if c == maxInds[j]:
                        if desc_label[c] == maxInds[j]:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        1 - batchProbs[j][desc_label[c]]) * softmaxed_weight[c]
                        else:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        -1 * batchProbs[j][desc_label[c]]) * softmaxed_weight[c]
        return grad_embeddings.cpu().numpy()

    # kmeans ++ initialization
    def k_means_plus_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if len(mu) % 100 == 0:
                print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def select(self, n_query, **kwargs):
        unlabeled_features = self.get_grad_features()
        selected_indices = self.k_means_plus_centers(X=unlabeled_features, K=n_query)
        scores = list(np.ones(len(selected_indices))) # equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index
                