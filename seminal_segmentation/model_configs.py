from task_configs import PATCH_SIZE

MODEL_CONFIGS = [
    {
        'name': 'UNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'conv',
            'encoder_units': 2,
            'decoder_block': 'conv',
            'decoder_units': 2,
        }
    },
    {
        'name': 'ResUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'res',
            'encoder_units': 2,
            'decoder_block': 'res',
            'decoder_units': 2,
        }

    },
    {
        'name': 'DenseUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'dense',
            'encoder_units': 4,
            'decoder_block': 'conv',
            'decoder_units': 2,
        }
    },
    {
        'name': 'InceptionUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'inception',
            'encoder_units': 2,
            'decoder_block': 'inception',
            'decoder_units': 2,
        }
    },
    {
        'name': 'AxUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 4,
            'encoder_block': 'axial',
            'encoder_units': 3,
            'decoder_block': 'conv',
            'decoder_units': 2,
        }
    },
]

MODEL_CONFIGS = [
    {
        'name': 'UNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'conv',
            'encoder_units': 2,
            'decoder_block': 'conv',
            'decoder_units': 2,
        }
    },
    {
        'name': 'ResUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'res',
            'encoder_units': 2,
            'decoder_block': 'res',
            'decoder_units': 2,
        }

    },
    {
        'name': 'DenseUNet',
        'model_settings': {
            'in_channels': 1,
            'out_channels': 2,
            'img_size': PATCH_SIZE,
            'norm_name': "INSTANCE",
            'num_layers': 5,
            'encoder_block': 'dense',
            'encoder_units': 4,
            'decoder_block': 'conv',
            'decoder_units': 2,
        }
    },
]