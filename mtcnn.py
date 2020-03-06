import tensorflow.keras as keras
import numpy as np

import cv2
import models
import utils


class MTCNN(object):
    def __init__(self, min_face_size: int = 24, scale: float = 0.709):
        self.min_face_size = min_face_size
        self.scale_factor = scale

    @staticmethod
    def generate_bboxes_with_scores(cls_map, scale, threshold=0.5, size=12, stride=2):
        """
            generate bounding boxes from score map

            @param cls_map: PNet's output feature map for classification
            @param scale: the scale of the image feed to PNet, used to 
                convert bbox coordinates of the resized image into coordinates
                of the original image 
            @param threshold: classification score threshold
            @param size: bbox size (size, size) (default 12)
            @param stride: the stride for bbox generation (default 2)
        """
        assert len(cls_map.shape) == 2

        indices = np.where(cls_map >= threshold)
        bboxes = np.concatenate((
            ((indices[1] * stride) / scale).reshape(-1, 1),
            ((indices[0] * stride) / scale).reshape(-1, 1),
            ((indices[1] * stride + size) / scale).reshape(-1, 1),
            ((indices[0] * stride + size) / scale).reshape(-1, 1),
            cls_map[indices].reshape(-1, 1)
        ), axis=1)
        return bboxes, indices

    def get_image_pyramid_scales(self, min_size: int, img_size: tuple):
        m = min(img_size)
        scales = []
        scale = 1

        while m >= min_size:
            scales.append(scale)
            scale *= self.scale_factor
            m *= self.scale_factor
        return scales

    @staticmethod
    def NMS(rectangles, threshold, type):
        if len(rectangles) == 0:
            return rectangles
        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort())
        pick = []
        while len(I) > 0:
            # I[-1] have hightest prob score, I[0:-1]->others
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'iom':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(o <= threshold)[0]]
        result_rectangle = boxes[pick].tolist()
        return result_rectangle

    @staticmethod
    def non_maximum_suppression(boxes, threshold: float, mode: str = 'union'):
        """
            Non Maximum Suppression

            @return: indices of remained boxes
        """
        assert boxes.shape[1] == 5
        assert mode in ('union', 'minimum')

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        result = []
        while order.size > 0:
            i = order[0]
            others = order[1:]
            result.append(i)

            ix1 = np.maximum(x1[i], x1[others])
            iy1 = np.maximum(y1[i], y1[others])
            ix2 = np.minimum(x2[i], x2[others])
            iy2 = np.minimum(y2[i], y2[others])

            w = np.maximum(ix2 - ix1 + 1, 0)
            h = np.maximum(iy2 - iy1 + 1, 0)
            intersect = w * h

            if mode == 'union':
                iou = intersect / (areas[i] + areas[others] - intersect)
            elif mode == 'minimum':
                iou = intersect / (np.minimum(areas[i], areas[others]))

            i = np.where(iou <= threshold)[0]
            order = order[i + 1]
        return np.array(result, dtype=np.int32)

    @staticmethod
    def refine_bboxes(boxes, reg_offsets, transform=lambda x: x.astype(np.int32).reshape(-1, 1)):
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1

        refined_boxes = np.concatenate((
            transform(boxes[:, 0] + reg_offsets[:, 0] * w),
            transform(boxes[:, 1] + reg_offsets[:, 1] * h),
            transform(boxes[:, 2] + reg_offsets[:, 2] * w),
            transform(boxes[:, 3] + reg_offsets[:, 3] * h)
        ), axis=1)
        return refined_boxes

    @staticmethod
    def rect2square(rectangles):
        w = rectangles[:, 2] - rectangles[:, 0]
        h = rectangles[:, 3] - rectangles[:, 1]
        l = np.maximum(w, h).T
        rectangles[:, 0] = rectangles[:, 0] + w*0.5 - l*0.5
        rectangles[:, 1] = rectangles[:, 1] + h*0.5 - l*0.5
        rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
        return rectangles

    @staticmethod
    def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2-x1
        h = y2-y1
        x1 = np.array([(x1+dx1*w)[0]]).T
        y1 = np.array([(y1+dx2*h)[0]]).T
        x2 = np.array([(x2+dx3*w)[0]]).T
        y2 = np.array([(y2+dx4*h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
        rectangles = MTCNN.rect2square(rectangles)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])

        pick = np.array(pick)
        return MTCNN.NMS(pick, 0.3, 'iou')

    @staticmethod
    def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
        prob = cls_prob[:,1]
        pick = np.where(prob>=threshold)
        rectangles = np.array(rectangles)
        x1  = rectangles[pick,0]
        y1  = rectangles[pick,1]
        x2  = rectangles[pick,2]
        y2  = rectangles[pick,3]
        sc  = np.array([prob[pick]]).T
        dx1 = roi[pick,0]
        dx2 = roi[pick,1]
        dx3 = roi[pick,2]
        dx4 = roi[pick,3]
        w   = x2-x1
        h   = y2-y1
        pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
        pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
        pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
        pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
        pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
        pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
        pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
        pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
        pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
        pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
        # pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        # pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
        # pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        # pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
        # pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        # pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        # pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
        # pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        # pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
        # pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
        x1  = np.array([(x1+dx1*w)[0]]).T
        y1  = np.array([(y1+dx2*h)[0]]).T
        x2  = np.array([(x2+dx3*w)[0]]).T
        y2  = np.array([(y2+dx4*h)[0]]).T
        rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0     ,rectangles[i][0]))
            y1 = int(max(0     ,rectangles[i][1]))
            x2 = int(min(width ,rectangles[i][2]))
            y2 = int(min(height,rectangles[i][3]))
            if x2>x1 and y2>y1:
                pick.append([x1,y1,x2,y2,rectangles[i][4],
                    rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
        return MTCNN.NMS(pick,0.3,'iom')

    def stage_PNet(self, model, img):
        h, w, _ = img.shape
        img_size = (w, h)

        boxes_tot = np.empty((0, 5))
        reg_offsets = np.empty((0, 4))

        scales = self.get_image_pyramid_scales(self.min_face_size, img_size)

        print(scales)

        for scale in scales:
            resized = utils.scale_image(img, scale)
            normalized = utils.normalize_image(resized)
            net_input = np.expand_dims(normalized, 0)

            cls_map, reg_map, _ = model.predict(net_input)
            cls_map = cls_map.squeeze()[:, :, 1]  # here
            reg_map = reg_map.squeeze()

            boxes, indices = self.generate_bboxes_with_scores(
                cls_map, scale, threshold=0.7)
            reg_deltas = reg_map[indices]

            indices = self.non_maximum_suppression(boxes, 0.5, 'union')
            boxes_tot = np.append(boxes_tot, boxes[indices], axis=0)
            reg_offsets = np.append(reg_offsets, reg_deltas[indices], axis=0)

        indices = self.non_maximum_suppression(boxes_tot, 0.7, 'union')
        boxes_tot = boxes_tot[indices]
        reg_offsets = reg_offsets[indices]

        # refine bounding boxes
        refined_boxes = self.refine_bboxes(boxes_tot, reg_offsets)
        return refined_boxes

    def stage_RNet(self, model, img, refined_boxes):
        h, w, _ = img.shape
        input_imgs = []
        for box in refined_boxes:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 <0 :
                continue
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, (24, 24))
            input_img = utils.normalize_image(crop_img)
            input_imgs.append(input_img)

        input_imgs = np.array(input_imgs)
        cls_list, reg_list, _ = model.predict(input_imgs)
        cls_list = np.array(cls_list)
        reg_list = np.array(reg_list)

        refined_boxes = self.filter_face_24net(
            cls_list, reg_list, refined_boxes, w, h, 0.6)

        return np.array(refined_boxes)

    def stage_ONet(self, model,  img, refined_boxes):
        h, w, _ = img.shape
        input_imgs = []
        for box in refined_boxes:
            x1, y1, x2, y2 = box
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, (48, 48))
            input_img = utils.normalize_image(crop_img)
            input_imgs.append(input_img)
        
        input_imgs = np.array(input_imgs)
        cls_list, reg_list, lmk_list = model.predict(input_imgs)

        refined_boxes = self.filter_face_48net(cls_list,reg_list,lmk_list,refined_boxes,w,h,0.6)
        return refined_boxes


if __name__ == '__main__':

    img = cv2.imread('006.jpg')

    mtcnn = MTCNN()

    model_pent = models.make_pnet(train=False)
    model_pent.load_weights('savedmodel/model_pnet.h5')

    model_rnet = models.make_rnet(train=False)
    model_rnet.load_weights('savedmodel/model_rnet.h5')

    model_onet = models.make_onet(train=False)
    model_onet.load_weights('savedmodel/model_onet.h5')

    bboxes = mtcnn.stage_PNet(model_pent, img)

    if bboxes.shape[0] == 0:
        print("Empty")
        exit(0)

    bboxes = mtcnn.stage_RNet(model_rnet, img, bboxes)

    bboxes = bboxes[:,:4]
    bboxes = bboxes.astype(dtype=np.int32)

    results = mtcnn.stage_ONet(model_onet, img, bboxes)

    results = np.array(results)
    print('bbox count: ', results.shape[0])
    for ret in results:
        x1, y1, x2, y2 = ret[:4]
        socre = ret[4]
        landmark = ret[5:]
        landmark = landmark.astype(dtype=np.int32)        
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
        cv2.circle(img,(landmark[0],landmark[1]),1,(255,0,0))
        cv2.circle(img,(landmark[2],landmark[3]),1,(255,0,0))
        cv2.circle(img,(landmark[4],landmark[5]),1,(255,0,0))
        cv2.circle(img,(landmark[6],landmark[7]),1,(255,0,0))
        cv2.circle(img,(landmark[8],landmark[9]),1,(255,0,0))
        

    print(results)
    cv2.imshow('detection', img)
    cv2.waitKey(0)
