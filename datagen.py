import linecache
import os
from enum import Enum

import numpy as np

import cv2
import utils


widerface_images_dir = 'dataset/wider_images'
widerface_annos_dir = 'dataset'
widerface_annos_file = 'wider_face_train_bbx_gt.txt'

celeba_images_dir = 'dataset/celeba_images'
celeba_annos_dir = 'dataset'
celeba_annos_file = 'list_landmarks_align_celeba.txt'


DataState = Enum('DataState', ('stop', 'next'))


def widerface_data_loader(skip=0, **kwargs):
    """
        generator function

        load data from WIDER FACE dataset

        params from kwargs: widerface_images_dir, widerface_annos_dir, widerface_annos_file 
    """
    images_dir = kwargs.get('widerface_images_dir', widerface_images_dir)
    annos_dir = kwargs.get('widerface_annos_dir', widerface_annos_dir)
    annos_file = kwargs.get('widerface_annos_file', widerface_annos_file)

    annos_path = os.path.join(annos_dir, annos_file)
    lines = linecache.getlines(annos_path)
    n_lines = len(lines)

    while True:
        idx = 0
        while idx < n_lines:
            image_name = lines[idx].strip()
            assert '/' in image_name
            n_faces = int(lines[idx + 1])

            if skip > 0:
                idx += 1 + n_faces + 1
                if skip % 5000 == 0:
                    print('[WIDER FACE loader]: skipping. %d remaining.' % (skip))
                skip -= 1
                continue

            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)
            bboxes = []
            for i in range(n_faces):
                anno = lines[idx + 2 + i].strip().split()
                anno = list(map(int, anno))
                x1, y1, w, h = anno[0], anno[1], anno[2], anno[3]
                box = utils.convert_bbox((x1, y1, w, h), False)
                bboxes.append(box)

            response = yield image, bboxes, image_path
            if response == DataState.stop:
                return

            # idx += 1 + n_faces + 1
            if n_faces == 0:
                idx += 3
            else:
                idx += 1 + n_faces + 1


def celeba_data_loader(skip=0, **kwargs):
    """
        generator function

        load data from CelebA dataset

        params from kwargs: celeba_images_dir, celeba_annos_dir, celeba_annos_file 
    """
    images_dir = kwargs.get('celeba_images_dir', celeba_images_dir)
    annos_dir = kwargs.get('celeba_annos_dir', celeba_annos_dir)
    annos_file = kwargs.get('celeba_annos_file', celeba_annos_file)

    annos_path = os.path.join(annos_dir, annos_file)
    lines = linecache.getlines(annos_path)
    n_lines = len(lines)

    while True:
        idx = 2  # we start from the third line
        while idx < n_lines:
            line = lines[idx].strip().split()
            idx += 1

            if skip > 0:
                if skip % 5000 == 0:
                    print('[CelebA loader]: skipping. %d remaining.' % (skip))
                skip -= 1
                continue

            image_name = line[0]
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)

            landmarks = tuple(map(int, line[1:]))
            landmarks = [(landmarks[i], landmarks[i + 1])
                         for i in range(0, 10, 2)]

            response = yield image, landmarks, image_path
            if response == DataState.stop:
                return


class SampleType(Enum):
    negative = 0
    positive = 1
    partial = 2
    landmark = 3


def augmented_data_generator(dst_size, min_face):
    """
        training data generator for MTCNN

        @param dst_size: output image size (dst_size, dst_size)
        @param pos_cnt: expected count of positive samples in a batch
        @param part_cnt:  expected count of partial samples in a batch
        @param neg_cnt: expected count of negative samples in a batch
        @param ldmk_cnt: expected count of landmark samples in a batch
        @param double_aug: if set to True, the size of batches will double (using image augmentaion)
        @param skip: if nonzero, given number of images will be skipped (default zero)
        @param min_face: minimum face size

        according to the paper, pos_cnt : part_cnt : neg_cnt : ldmk_cnt should be 1 : 1 : 3 : 2

    """
    dst_size = dst_size
    pos_cnt = 10
    part_cnt = 10
    neg_cnt = 30
    ldmk_cnt = 20
    double_aug = False
    skip = 0
    min_face = min_face
    """
        record data format: 
        [ type, cls_0, cls1] [ type, bbox1, ..., bbox4 ] [ type, ldmk1, ... ldmk10 ]

        for positivie samples:
            type = SampleType.positive.value
            cls_0 = 0, cls_1 = 1
            bboxes: real
            ldmks: nan
        
        for negative samples:
            type = SampleType.negative.value
            cls_0 = 1, cls_1 = 0
            bboxes: real
            ldmks: nan
        
        for partial samples:
            type = SampleType.partial.value
            cls: nan
            bboxes: real
            ldmks: nan
        
        for landmark samples:
            type = SampleType.landmark.value
            cls: nan
            bboxes: nan
            ldmks: real
    """
    """
    # positive IoU threshold: 0.65+
    # partial IoU threshold: 0.4 - 0.65
    # negative IoU threshold: 0.3-
    """

    #pos_threshold_low = 0.65
    #neg_threshold_high = 0.3
    #part_threshold_low = 0.4

    widerface_loader = widerface_data_loader(skip=skip)
    celeba_loader = celeba_data_loader(skip=skip)

    loop_threshold = (pos_cnt + part_cnt + neg_cnt) * 10

    batch = 0
    while True:
        images, face_cls, bbox_reg, ldmk_reg = [], [], [], []

        # process images from WIDER FACE dataset

        img, boxes, _ = widerface_loader.send(None)
        h_img, w_img, _ = img.shape
        img_size = (w_img, h_img)
        boxes = np.array(boxes)

        n_pos, n_part, n_neg = 0, 0, 0
        loop_cnt = 0
        observe_flag = False

        pos_threshold_low = 0.65
        neg_threshold_high = 0.3
        part_threshold_low = 0.4

        no_proper_faces_found = True
        while n_pos < pos_cnt or n_part < part_cnt or n_neg < neg_cnt:

            def append_images_and_bboxes(im, gtbox, crbox, label):
                x1, y1, x2, y2, w, h = utils.unpack_bbox(crbox)
                dx1 = (gtbox[0] - x1) / w
                dy1 = (gtbox[1] - y1) / h
                dx2 = (gtbox[2] - x2) / w
                dy2 = (gtbox[3] - y2) / h

                # suppose it should be a one-hot vector
                cls_0, cls_1 = np.nan, np.nan
                if label == SampleType.negative:
                    cls_0, cls_1 = 1, 0
                elif label == SampleType.positive:
                    cls_0, cls_1 = 0, 1

                dummy_ldmks = [np.nan] * 10
                face_cls.append([label.value, cls_0, cls_1])
                bbox_reg.append([label.value, dx1, dy1, dx2, dy2])
                ldmk_reg.append([label.value] + dummy_ldmks)

                cropped = utils.crop_image(im, crbox)
                resized = cv2.resize(cropped, (dst_size, dst_size))
                images.append(resized)

            try:
                for box in boxes:
                    x1, y1, w, h = utils.convert_bbox(box, True)

                    if max(w, h) < min_face:  # bounding box too small, discard it
                        continue
                    no_proper_faces_found = False

                    if n_pos < pos_cnt or n_part < part_cnt:
                        crop_box = utils.bbox_positive_sampling(box)
                        if utils.is_valid_bbox(crop_box, img_size):
                            iou = utils.IoU(crop_box, boxes)
                            iou = np.max(iou)
                            # if observe_flag:
                            #    print(iou)
                            if iou >= pos_threshold_low and n_pos < pos_cnt:
                                n_pos += 1
                                #cv2.imshow('positive', utils.crop_image(img, crop_box))
                                append_images_and_bboxes(
                                    img, box, crop_box, SampleType.positive)

                            elif iou >= part_threshold_low and n_part < part_cnt:
                                n_part += 1
                                #cv2.imshow('partial', utils.crop_image(img, crop_box))
                                append_images_and_bboxes(
                                    img, box, crop_box, SampleType.partial)

                    if n_neg < neg_cnt:
                        crop_box = utils.bbox_global_negative_sampling(
                            box, img_size, dst_size)
                        if utils.is_valid_bbox(crop_box, img_size):
                            iou = utils.IoU(crop_box, boxes)
                            iou = np.max(iou)
                            # if observe_flag:
                            #    print(iou)
                            if iou < neg_threshold_high:
                                n_neg += 1
                                #cv2.imshow('negative', utils.crop_image(img, crop_box))
                                append_images_and_bboxes(
                                    img, box, crop_box, SampleType.negative)

                    if n_neg < neg_cnt:
                        crop_box = utils.bbox_local_negative_sampling(
                            box, dst_size)
                        if utils.is_valid_bbox(crop_box, img_size):
                            iou = utils.IoU(crop_box, boxes)
                            iou = np.max(iou)
                            # if observe_flag:
                            #    print(iou)
                            if iou < neg_threshold_high:
                                n_neg += 1
                                append_images_and_bboxes(
                                    img, box, crop_box, SampleType.negative)

                    loop_cnt += 1
                    if loop_cnt > loop_threshold * 2:
                        # we can't handle these bounding boxes, skip
                        observe_flag = False
                        no_proper_faces_found = True
                        break

                    elif loop_cnt > loop_threshold and not observe_flag:
                        # adjust IoU threshold
                        pos_threshold_low = 0.55
                        neg_threshold_high = 0.4
                        observe_flag = True
            except:
                # there might be a few exceptions, try to ignore them?
                continue

            if no_proper_faces_found:
                break

        if no_proper_faces_found:
            continue

        # process images from CelebA dataset

        img, ldmks, _ = celeba_loader.send(None)
        h_img, w_img, _ = img.shape
        img_size = (h_img, w_img)

        n_ldmk = 0
        while n_ldmk < ldmk_cnt:
            try:
                box = utils.crop_bbox_for_facial_landmarks(ldmks)
                if utils.is_valid_bbox(box, img_size):
                    n_ldmk += 1

                    angle = np.random.random_integers(-15, 15)
                    aug_img, ldmks = utils.rotate_facial_landmarks(
                        img, ldmks, box, angle)
                    aug_img = utils.crop_image(aug_img, box)
                    aug_img = utils.adjust_hue_and_saturation(aug_img)
                    aug_img = utils.adjust_lighting_naive(aug_img)
                    resized = cv2.resize(aug_img, (dst_size, dst_size))
                    images.append(resized)

                    x1, y1, w, h = utils.convert_bbox(box, True)
                    d_ldmks = []
                    for x, y in ldmks:
                        dx = (x - x1) / w
                        dy = (y - y1) / h
                        d_ldmks.extend((dx, dy))

                    dummy_cls_bbox = [np.nan] * 4
                    face_cls.append(
                        [SampleType.landmark.value, np.nan, np.nan])
                    bbox_reg.append(
                        [SampleType.landmark.value] + dummy_cls_bbox)
                    ldmk_reg.append([SampleType.landmark.value] + d_ldmks)
            except:
                continue

        if double_aug:
            def aug_fn(im): return utils.adjust_lighting_naive(
                utils.adjust_hue_and_saturation(im))
            images.extend(list(map(aug_fn, images)))
            face_cls.extend(face_cls)
            bbox_reg.extend(bbox_reg)
            ldmk_reg.extend(ldmk_reg)

        # do shuffle ? (we may left that to .fit_generator)
        images = list(map(utils.normalize_image, images))
        yield np.array(images), {
            'face_cls': np.array(face_cls),
            'bbox_reg': np.array(bbox_reg),
            'ldmk_reg': np.array(ldmk_reg)
        }

        batch += 1
        #print('batch %d' %(batch))


def load_and_display_widerface(max_count=10, skip=0):
    """
        show annotated images from WIDER FACE
    """
    loader = widerface_data_loader(skip=skip)

    count = 0
    for img, bboxes, path in loader:
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(path, img)
        cv2.waitKey(0)

        count += 1
        if count >= max_count:
            loader.send(DataState.stop)


def load_and_display_celeba(max_count=10, skip=0):
    """
        show annotated facial landmarks from CelebA
    """
    loader = celeba_data_loader(skip=skip)

    count = 0
    for img, landmarks, path in loader:
        for x, y in landmarks:
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)
        x1, y1, x2, y2 = utils.generate_bbox_from_landmarks(img, landmarks)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow(path, img)
        cv2.waitKey(0)

        count += 1
        if count >= max_count:
            loader.send(DataState.stop)


if __name__ == '__main__':
    pass
