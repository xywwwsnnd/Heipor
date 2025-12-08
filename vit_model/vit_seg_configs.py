# vit_seg_configs.py

def get_b16_config():
    return {
        'patches': {'size': (16, 16)},
        'hidden_size': 768,
        'transformer': {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


def get_b32_config():
    return {
        'patches': {'size': (32, 32)},
        'hidden_size': 768,
        'transformer': {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


def get_l16_config():
    return {
        'patches': {'size': (16, 16)},
        'hidden_size': 1024,
        'transformer': {
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_layers': 24,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


def get_l32_config():
    return {
        'patches': {'size': (32, 32)},
        'hidden_size': 1024,
        'transformer': {
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_layers': 24,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


def get_h14_config():
    return {
        'patches': {'size': (14, 14)},
        'hidden_size': 1280,
        'transformer': {
            'mlp_dim': 5120,
            'num_heads': 16,
            'num_layers': 32,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


def get_r50_b16_config():
    config = get_b16_config()
    config.update({
        'patches': {'grid': (16, 16)},
        'resnet': {
            'num_layers': [3, 4, 6, 3],
            'width_factor': 1.0
        }
    })
    return config


def get_r50_l16_config():
    config = get_l16_config()
    config.update({
        'patches': {'grid': (16, 16)},
        'resnet': {
            'num_layers': [3, 4, 6, 3],
            'width_factor': 1.0
        }
    })
    return config


def get_testing():
    return {
        'patches': {'size': (16, 16)},
        'hidden_size': 256,
        'transformer': {
            'mlp_dim': 512,
            'num_heads': 4,
            'num_layers': 4,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'token',
        'representation_size': None
    }


# 总入口：提供统一调用接口
CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing()
}
