import os
import re
import types
import torch
import contextlib
import numpy as np
import torch.nn as nn
from PIL import Image

from torchvision import models
from torchvision.transforms import functional as F

from alfred.gen import constants
from alfred.nn.transforms import Transforms

import supervision as sv

BOX_THRESHOLD = 0.4

class Resnet18(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''
    def __init__(self,
                 device,
                 checkpoint_path=None,
                 share_memory=False):
        super().__init__()
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        if checkpoint_path is not None:
            print('Loading ResNet checkpoint from {}'.format(checkpoint_path))
            model_state_dict = torch.load(checkpoint_path, map_location=device)
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'GU_' not in key and 'text_pooling' not in key}
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'fc.' not in key}
            model_state_dict = {
                key.replace('resnet.', ''): value
                for key, value in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(torch.device(device))
        # self.model = self.model.cuda()
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        self._transform = Transforms.get_transform('default')

    def extract(self, x):
        x = self._transform(x).to(torch.device(self.device))
        # x = self._transform(x).cuda()
        return self.model(x)


class RCNN(nn.Module):
    '''
    pretrained FasterRCNN or MaskRCNN from torchvision
    '''
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint_path=None,
                 share_memory=False,
                 load_heads=False):
        super().__init__()
        self.device = device
        self.feat_layer = '3'
        if archi == 'maskrcnn':
            self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=800)
        elif archi == 'fasterrcnn':
            self.model = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=(checkpoint_path is None),
                pretrained_backbone=(checkpoint_path is None),
                min_size=224)
        else:
            raise ValueError('Unknown model type = {}'.format(archi))

        if archi == 'maskrcnn':
            self._transform = self.model.transform
        else:
            self._transform = Transforms.get_transform('default')
        if not load_heads:
            for attr in ('backbone', 'body'):
                self.model = getattr(self.model, attr)

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, load_heads, device, archi, 'backbone.body')
        self.model = self.model.to(torch.device(device))
        # self.model = self.model.cuda()
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        if load_heads:
            # if the model is used for predictions, prepare a vocabulary
            # TODO: 词表 list of AI2THOR obj names
            self.vocab_pred = {
                i: class_name for i, class_name in
                enumerate(constants.OBJECTS_ACTIONS)}
            OBJECTS_ACTIONS = [
            'None', 'AlarmClock', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat',
            'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book', 'Bowl',
            'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet',
            'Candle', 'Cart', 'CellPhone', 'Cloth', 'CoffeeMachine', 'CoffeeTable',
            'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable',
            'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet', 'FloorLamp', 'Fork',
            'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'Kettle', 'KeyChain',
            'Knife', 'Ladle', 'Laptop', 'Lettuce', 'LettuceSliced', 'Microwave', 'Mug',
            'Newspaper', 'Ottoman', 'Pan', 'Pen', 'Pencil', 'PepperShaker',
            'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl',
            'Safe', 'SaltShaker', 'Shelf', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar',
            'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue',
            'StoveBurner', 'TVStand', 'TennisRacket', 'TissueBox', 'Toilet', 'ToiletPaper',
            'ToiletPaperHanger', 'Tomato', 'TomatoSliced', 'Vase', 'Watch', 'WateringCan',
            'WineBottle']

    def extract(self, images):
        if isinstance(
                self._transform, models.detection.transform.GeneralizedRCNNTransform):
            images_normalized = self._transform(
                torch.stack([F.to_tensor(img) for img in images]))[0].tensors
        else:
            images_normalized = torch.stack(
                [self._transform(img) for img in images])
        images_normalized = images_normalized.to(torch.device(self.device))
        # images_normalized = images_normalized.cuda()
        model_body = self.model
        if hasattr(self.model, 'backbone'):
            model_body = self.model.backbone.body
        features = model_body(images_normalized)
        return features[self.feat_layer]

    def load_from_checkpoint(self, checkpoint_path, load_heads, device, archi, prefix):
        print('Loading RCNN checkpoint from {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not load_heads:
            # load only the backbone
            state_dict = {
                k.replace(prefix + '.', ''): v
                for k, v in state_dict.items() if prefix + '.' in k}
        else:
            # load a full model, replace pre-trained head(s) with (a) new one(s)
            num_classes, in_features = state_dict[
                'roi_heads.box_predictor.cls_score.weight'].shape
            box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes)
            self.model.roi_heads.box_predictor = box_predictor
            if archi == 'maskrcnn':
                # and replace the mask predictor with a new one
                in_features_mask = \
                    self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes)
                self.model.roi_heads.mask_predictor = mask_predictor
        self.model.load_state_dict(state_dict)

    # TODO: pred = types.SimpleNamespace(label=label, box=box, score=score, mask=mask)
    def predict_objects(self, image, confidence_threshold=0.8, verbose=False):
        print('using threshold', confidence_threshold)
        image = F.to_tensor(image).to(torch.device(self.device))
        # image = F.to_tensor(image).cuda()
        output = self.model(image[None])[0]
        preds = []
        for pred_idx in range(len(output['scores'])):
            score = output['scores'][pred_idx].cpu().item()
            if score < confidence_threshold:
                continue
            box = output['boxes'][pred_idx].cpu().numpy()
            label = self.vocab_pred[output['labels'][pred_idx].cpu().item()]  # 'KeyChain' 'Sink'/'SinkBasin'?
            if verbose:
                print('{} at {} with {}'.format(label, box, score))
            pred = types.SimpleNamespace(
                label=label, box=box, score=score)
            if 'masks' in output:
                pred.mask = output['masks'][pred_idx].cpu().numpy()
            preds.append(pred)
        return preds

    
    def segmentation_for_map(self, env, verbose=False, classes_needed:list=None, confidence_threshold=0.8): 
        # FIXME: basin
        for idx, obj in enumerate(classes_needed):
            if 'Basin' in obj:
                classes_needed[idx] = obj.replace('Basin', '')
        print('classes_needed', classes_needed)
        image = Image.fromarray(env.last_event.frame) 
        preds = self.predict_objects(image, verbose=verbose, confidence_threshold=confidence_threshold)
        # semantic_seg = np.zeros((self.env_frame_height, self.env_frame_width, self.num_sem_categories))
        
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        detections = []
        for pred in preds:  # for each entities
            if pred.label in classes_needed:
                cat = classes_needed.index(pred.label)
                
                detection = sv.Detections(
                    xyxy=np.expand_dims(pred.box, axis=0),
                    confidence=np.array([pred.score]),
                    class_id=np.array([cat]),
                    mask=(pred.mask > 0) * 1,
                    # mask = pred.mask.astype(int)
                )
                detections.append(detection)
                
                # v = pred.score * pred.mask
                # semantic_seg[:, :, cat] += v.astype('float')
        
        detections = sv.Detections.merge(detections)
        
        '''
        # FIND LAMP
        desklamp_idx_ = classes_needed.index("DeskLamp")
        floorlamp_idx_ = classes_needed.index("FloorLamp")
        none_idx = classes_needed.index("None")
        if desklamp_idx_ > none_idx or floorlamp_idx_ > none_idx:
            # print('trying hard to find lamp')
            
            desklamp_idx = np.where(detections.class_id == classes_needed.index("DeskLamp"))[0]
            floorlamp_idx = np.where(detections.class_id == classes_needed.index("FloorLamp"))[0]
            non_desklamp_idx = np.where(detections.class_id != classes_needed.index("DeskLamp"))[0]
            non_floorlamp_idx = np.where(detections.class_id != classes_needed.index("FloorLamp"))[0]
            non_lamp_idx = np.hstack((non_desklamp_idx, non_floorlamp_idx))
            non_lamp_idx = list(set(non_lamp_idx))
            non_lamp_idx = [int(x) for x in non_lamp_idx]
            # print('desklamp_idx', desklamp_idx)
            # print('floorlamp_idx', floorlamp_idx)
            # print('non_lamp_idx', non_lamp_idx)
            non_lamp_high_idx = []
            if len(non_lamp_idx) != 0:
                for idx in non_lamp_idx:
                    # print('detections.confidence[idx]', detections.confidence[idx])
                    if detections.confidence[idx] > 0.8:
                        non_lamp_high_idx.append(idx)
            
            non_lamp_high_idx = np.array(non_lamp_high_idx).astype(int)
            save_idx = np.hstack((desklamp_idx, floorlamp_idx, non_lamp_high_idx))
            
            # print('save_idx', save_idx)
            detections.xyxy = detections.xyxy[save_idx]
            detections.confidence = detections.confidence[save_idx]
            detections.class_id = detections.class_id[save_idx]
            detections.mask = detections.mask[save_idx]
        '''
        
        labels = []
        for _, _, confidence, class_id, _ in detections:
            if class_id < len(classes_needed):
                labels.append(f"{classes_needed[class_id]} {confidence:0.2f}")
            else:
                labels.append(f"{classes_needed[len(classes_needed) - 1]} {confidence:0.2f}")  
        # image = np.array(image)
        annotated_image = mask_annotator.annotate(scene=env.last_event.cv2img.astype(np.uint8).copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return detections, annotated_image


def get_feat_shape(visual_archi, compress_type=None):
    '''
    Get feat shape depending on the training archi and compress type
    '''
    if visual_archi == 'fasterrcnn':
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == 'maskrcnn':
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == 'resnet18':
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError('Unknown archi {}'.format(visual_archi))

    if compress_type is not None:
        if not re.match(r'\d+x', compress_type):
            raise NotImplementedError('Unknown compress type {}'.format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0], feat_shape[1] // compress_times,
            feat_shape[2], feat_shape[3])
    return feat_shape


class FeatureExtractor(nn.Module):
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint=None,
                 share_memory=False,
                 compress_type=None,
                 load_heads=False):
        super().__init__()
        self.feat_shape = get_feat_shape(archi, compress_type)
        self.eval_mode = True
        if archi == 'resnet18':
            assert not load_heads
            self.model = Resnet18(device, checkpoint, share_memory)
        else:
            self.model = RCNN(
                archi, device, checkpoint, share_memory, load_heads=load_heads)
        self.compress_type = compress_type
        # load object class vocabulary
        # TODO: 词表 at 'files/obj_cls.vocab'
        vocab_obj_path = "../obj_cls.vocab"
        # vocab_obj_path = os.path.join(
        #     constants.ET_ROOT, constants.OBJ_CLS_VOCAB)
        self.vocab_obj = torch.load(vocab_obj_path)
        print('self.vocab_obj', self.vocab_obj)

    def featurize(self, images, batch=32):
        feats = []
        with (torch.set_grad_enabled(False) if not self.model.model.training
              else contextlib.nullcontext()):
            for i in range(0, len(images), batch):
                images_batch = images[i:i+batch]
                feats.append(self.model.extract(images_batch))
        feat = torch.cat(feats, dim=0)
        # TODO: 没有用到 feat_compress，可以不用
        # if self.compress_type is not None:
        #     feat = data_util.feat_compress(feat, self.compress_type)
        assert self.feat_shape[1:] == feat.shape[1:]
        return feat

    def predict_objects(self, image, verbose=False, confidence_threshold=0.8):
        with torch.set_grad_enabled(False):
            pred = self.model.predict_objects(image, verbose=verbose, confidence_threshold=confidence_threshold)
        return pred

    def train(self, mode):
        if self.eval_mode:
            return
        for module in self.children():
            module.train(mode)

    def segmentation_for_map(self, image, verbose=False, classes_needed:list=None, confidence_threshold=0.8):
        with torch.set_grad_enabled(False):
            pred = self.model.segmentation_for_map(image, verbose=verbose, classes_needed=classes_needed, confidence_threshold=confidence_threshold)
        return pred
    
    # def segmentation(self, image, verbose=False, classes_needed:list=None,confidence_threshold=0.8):
    #     with torch.set_grad_enabled(False):
    #         pred = self.model.segmentation_for_map(image, verbose=verbose, classes_needed=classes_needed, confidence_threshold=confidence_threshold)
    #     return pred
    
class FeatureFlat(nn.Module):
    '''
    a few conv layers to flatten features that come out of ResNet
    '''
    def __init__(self, input_shape, output_size):
        super().__init__()
        if input_shape[0] == -1:
            input_shape = input_shape[1:]
        layers, activation_shape = self.init_cnn(
            input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0])
        layers += [
            Flatten(), nn.Linear(np.prod(activation_shape), output_size)]
        self.layers = nn.Sequential(*layers)

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(planes_in, planes_out, kernel_size=kernel,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(planes_out), nn.ReLU(inplace=True)]
            planes_in = planes_out
            spatial = (spatial - kernel + 2 * padding) // stride + 1
        activation_shape = (planes_in, spatial, spatial)
        return layers, activation_shape

    def forward(self, frames):
        activation = self.layers(frames)
        return activation


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
