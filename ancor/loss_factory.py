from torch import nn

LOSSES = ['ce', 'cosine']


class LossFactory(object):
    def create_criterion(self, loss_type, gpu):
        if loss_type == 'ce':
            return nn.CrossEntropyLoss().cuda(gpu)
        elif loss_type == 'cosine':
            def D(p, z):
                z = z.detach()
                p = nn.functional.normalize(p, dim=1)
                z = nn.functional.normalize(z, dim=1)
                return -(p * z).sum(dim=1).mean()

            return D
            # def loss_fn(x, y):
            #     x = nn.functional.normalize(x.to(gpu), dim=-1, p=2)
            #     y = nn.functional.normalize(y.to(gpu).detach(), dim=-1, p=2)
            #     return 2 - 2 * (x * y).sum(dim=-1)
            # return loss_fn

        else:
            raise NotImplementedError
