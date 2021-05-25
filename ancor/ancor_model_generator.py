from collections import OrderedDict

from ancor.ancor import ANCOR
from ancor.arch_factory import ArchFactory
from ancor.calculator_factory import CalculatorFactory
from ancor.head_factory import HeadFactory
from ancor.loss_factory import LossFactory
from ancor.queue_factory import QueueFactory


class ANCORModelGenerator(object):

    def __init__(self):
        self.arch_factory = ArchFactory()
        self.head_factory = HeadFactory()
        self.queue_factory = QueueFactory()
        self.calculator_factory = CalculatorFactory()

    def generate_ancor_model(self, arch, head_type, dim=128, K=65536, m=0.999, T=0.07, mlp=False, num_classes=1000,
                             queue_type='single', metric='angular', gpu=0, calc_types=[], loss_types=[]):
        assert len(calc_types) == len(loss_types)

        encoder_q, encoder_k, k2q_mapping = self.arch_factory.create_arch(arch)
        feature_dim = encoder_q.fc.weight.shape[1]
        fc_q, fc_k, mapping = self.head_factory.create_head(feature_dim, dim, head_type, num_classes, mlp)
        encoder_q.fc, encoder_k.fc = fc_q, fc_k
        if mapping is not None:
            mapping = {'fc.' + key: 'fc.' + value for key, value in mapping.items()}
            k2q_mapping.update(mapping)
        del k2q_mapping['fc.weight']
        del k2q_mapping['fc.bias']
        queue, queue_ptr, dequeuer = self.queue_factory.create_queues(queue_type, K, num_classes, dim)
        calculators = [
            self.calculator_factory.create_calculator(
                calc_type, metric, T=T, num_classes=num_classes
            ) for calc_type in calc_types
        ]
        loss_factory = LossFactory()
        criterions = OrderedDict(
            [(f"{calc_type}_{loss_type}", loss_factory.create_criterion(loss_type, gpu))
             for calc_type, loss_type in zip(calc_types, loss_types)])
        model = ANCOR(encoder_q, encoder_k, k2q_mapping, {'queue': queue, 'queue_ptr': queue_ptr}, dequeuer,
                      calculators, m)
        self.print_model_aspects(arch, head_type, dim, K, m, T, mlp, num_classes, queue_type, metric, calc_types,
                                 loss_types)
        return model, criterions

    def print_model_aspects(self, arch, head_type, dim, K, m, T, mlp, num_classes,
                            queue_type, metric, calc_types, loss_types):
        print(f"Arch: {arch}")
        print(f"Head type: {head_type}")
        print(f"Contrast dim: {dim}")
        print(f"Queue type: {queue_type}")
        print(f"Queue size: {K}")
        print(f"Encoder Momentum: {m}")
        print(f"Metric: {metric}")
        print(f"MLP: {mlp}")
        print(f"Temperature: {T}")
        print(f"Calculator types: {calc_types}")
        print(f"Loss types: {loss_types}")
        print(f"Num classes: {num_classes}")
