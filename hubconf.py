# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_YOLOv8

Usage:
    import torch
    model = torch.hub.load('ultralytics/YOLOv8', 'YOLOv8s')  # official model
    model = torch.hub.load('ultralytics/YOLOv8:master', 'YOLOv8s')  # from branch
    model = torch.hub.load('ultralytics/YOLOv8', 'custom', 'YOLOv8s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'YOLOv8s.pt', source='local')  # local repo
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates or loads a YOLOv8 model

    Arguments:
        name (str): model name 'YOLOv8s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv8 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv8 model
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / 'requirements.txt', exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv8 ClassificationModel is not yet AutoShape compatible. '
                                       'You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).')
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv8 SegmentationModel is not yet AutoShape compatible. '
                                       'You will not be able to run inference with this model.')
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = 'https://docs.ultralytics.com/YOLOv8/tutorials/pytorch_hub_model_loading'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # YOLOv8 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def YOLOv8n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-nano model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8n', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-small model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8s', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-medium model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8m', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-large model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8l', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-xlarge model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8x', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-nano-P6 model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8n6', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-small-P6 model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8s6', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-medium-P6 model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8m6', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-large-P6 model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8l6', pretrained, channels, classes, autoshape, _verbose, device)


def YOLOv8x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv8-xlarge-P6 model https://github.com/ultralytics/YOLOv8
    return _create('YOLOv8x6', pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='YOLOv8s', help='model name')
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        'data/images/zidane.jpg',  # filename
        Path('data/images/zidane.jpg'),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
