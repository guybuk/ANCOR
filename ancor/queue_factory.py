import torch
from torch import nn

from ancor.util import concat_all_gather

QUEUES = ['single', 'multi', 'none']


class QueueFactory(object):
    def create_queues(self, queue_type, K, num_classes, dim):
        if queue_type == 'single':
            queue = torch.randn(dim, K)
            queue = nn.functional.normalize(queue, dim=0)
            queue_ptr = torch.zeros(1, dtype=torch.long)
            dequeuer = SingleDequeuer(K)
        elif queue_type == 'multi':
            queue = torch.randn(num_classes, dim, K)
            queue = nn.functional.normalize(queue, dim=1)
            queue_ptr = torch.zeros(num_classes, dtype=torch.long)
            dequeuer = MultiDequeuer(K)
        elif queue_type is None or queue_type == 'none':
            return None, None, NullDequeuer(0)
        else:
            raise NotImplementedError
        return queue, queue_ptr, dequeuer


class Dequeuer(object):
    def __init__(self, K):
        self.K = K

    def dequeue_and_enqueue(self, queue, queue_ptr, keys):
        raise NotImplementedError


class NullDequeuer(Dequeuer):
    def dequeue_and_enqueue(self, queue, queue_ptr, keys, **kwargs):
        return None, None


class SingleDequeuer(Dequeuer):
    @torch.no_grad()
    def dequeue_and_enqueue(self, queue, queue_ptr, keys, **kwargs):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr
        return queue, queue_ptr


class MultiDequeuer(Dequeuer):
    def dequeue_and_enqueue(self, queue, queue_ptr, keys, cls_labels=None):
        keys = concat_all_gather(keys)
        cls_labels = concat_all_gather(cls_labels)
        for cls_id in torch.unique(cls_labels):
            cls_keys = keys[cls_labels == cls_id]
            batch_size = cls_keys.size(0)
            ptr = int(queue_ptr[cls_id])

            # replace the keys at ptr (dequeue and enqueue)
            if ptr + batch_size >= self.K:
                queue[cls_id][:, ptr:] = cls_keys.T[:, :self.K - ptr]
                queue[cls_id][:, :(ptr + batch_size) % self.K] = cls_keys.T[:, self.K - ptr:]
            else:
                queue[cls_id][:, ptr:ptr + batch_size] = cls_keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            queue_ptr[cls_id] = ptr
        return queue, queue_ptr
