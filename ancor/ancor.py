import torch
from torch import nn

from ancor.util import concat_all_gather


class ANCOR(nn.Module):
    def __init__(self, encoder_q, encoder_k, k2q_mapping, queue_obj, dequeuer, logits_and_labels_calculators, m=0.999):
        super(ANCOR, self).__init__()
        self.m = m
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.k2q_mapping = k2q_mapping
        self.dequeuer = dequeuer
        self.logits_and_labels_calculators = logits_and_labels_calculators
        self.register_buffer("queue", queue_obj["queue"])
        self.register_buffer("queue_ptr", queue_obj["queue_ptr"])
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for name_k, name_q in self.k2q_mapping.items():
            param_q = state_dict_q[name_q]
            param_k = state_dict_k[name_k]
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, im_q, im_k, **kwargs):
        q_outs = self.encoder_q(im_q)
        self._momentum_update_key_encoder()
        keys = self._encoder_k_forward(im_k)
        detached_queue = self.queue.clone().detach() if self.queue is not None else None
        logits_and_labels_list = [
            calculator.calculate(q_outs, encoder_q=self.encoder_q, k=keys, queue=detached_queue, **kwargs)
            for calculator in self.logits_and_labels_calculators
        ]
        self.queue, self.queue_ptr = self.dequeuer.dequeue_and_enqueue(self.queue, self.queue_ptr, keys, **kwargs)
        return logits_and_labels_list

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        state_dict_q = self.encoder_q.state_dict()
        state_dict_k = self.encoder_k.state_dict()
        for name_k, name_q in self.k2q_mapping.items():
            param_k = state_dict_k[name_k]
            param_q = state_dict_q[name_q]
            param_k.data.copy_(param_k.data * self.m + param_q.data * (1. - self.m))

    @torch.no_grad()
    def _encoder_k_forward(self, img):
        try:
            img, idx_unshuffle = self._batch_shuffle_ddp(img)
        except:
            idx_unshuffle = None

        k = self.encoder_k(img)  # keys: NxC

        # undo shuffle
        if idx_unshuffle is not None:
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
