from models import model_pool


class ArchFactory(object):
    def create_arch(self, arch_name):
        encoder_q, encoder_k = model_pool[arch_name](), model_pool[arch_name]()
        k2q_mapping = {k_name: q_name for q_name, k_name in
                       zip(encoder_q.state_dict().keys(), encoder_k.state_dict().keys())}

        return encoder_q, encoder_k, k2q_mapping
