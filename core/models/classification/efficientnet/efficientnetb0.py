from .efficientnet import EfficientNet


class EfficientNetB0(EfficientNet):
    config = {
        "name": "efficientnetb0",
        "image_size": 224,
        "width_scale_factor": 1.0,
        "depth_scale_factor": 1.0,
        "drop_connect_rate": 0.2,       # Stochastic dropout rate for each MBConv block
        "drop_rate": 0.2                # Drop out rate for final dense layer
    }


# class EfficientNetB1(EfficientNet):
#     config = {
#         "name": "efficientnetb1",
#         "image_size": 240,
#         "width_scale_factor": 1.0,
#         "depth_scale_factor": 1.1,
#         "drop_connect_rate": 0.2,
#         "drop_rate": 0.2
#     }
