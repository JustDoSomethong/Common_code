from Detection import *
from trace_classifier_mobilenet import *
from gesture_classifier_mobilnet import *



def genGestureSample(origimg, bbox):
    img_shape = origimg.shape
    img_h, img_w = img_shape[0], img_shape[1]
    box_h, box_w = bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]
    if box_h > box_w:
        delta = (box_h - box_w) / 2
        x_min = max(0, bbox[0][0] - delta)
        x_max = min(img_w, bbox[1][0] + delta)
        y_min = bbox[0][1]
        y_max = bbox[1][1]
    elif box_h <= box_w:
        delta = (box_w - box_h) / 2
        x_min = bbox[0][0]
        x_max = bbox[1][0]
        y_min = max(0, bbox[0][1] - delta)
        y_max = min(img_h, bbox[1][1] + delta)

    delta = int((y_max - y_min) * 0.15)
    y_min = max(0, y_min - delta)
    y_max = min(img_h, y_max + delta)
    x_min = max(0, x_min - delta)
    x_max = min(img_w, x_max + delta)

    return origimg[y_min:y_max, x_min:x_max, :]


def Finger(gesture_classify_result_quene):
    return 1
    gesture = 0
    cnt = 0
    for result in gesture_classify_result_quene:
        if result == gesture:
            cnt += 1
    return float(cnt) / len(gesture_classify_result_quene) > 0.6
    return gesture_classify_result_quene[-1] == 1


def genTraceSample(im):
    im_shape = im.shape
    for i in range(im_shape[0]):
        if 0 in im[i, :, :]:
            x_min = i
            break
    for i in range(im_shape[0] - 1, -1, -1):
        if 0 in im[i, :, :]:
            x_max = i
            break
    for j in range(im_shape[1]):
        if 0 in im[:, j, :]:
            y_min = j
            break
    for j in range(im_shape[1] - 1, -1, -1):
        if 0 in im[:, j, :]:
            y_max = j
            break
    im = im[x_min:x_max, y_min:y_max, :]

    im_shape = im.shape
    h, w = im_shape[0], im_shape[1]
    if h > w:
        delta = (h - w) / 2
        delta = np.zeros(shape=(h, delta, 3), dtype=np.uint8) + 255
        im = np.hstack([delta, im, delta])
    elif w > h:
        delta = (w - h) / 2
        delta = np.zeros(shape=(delta, w, 3), dtype=np.uint8) + 255
        im = np.vstack([delta, im, delta])
    # im = cv2.resize(im, (224, 224))
    return im




class yk_models():
    def __init__(self):
        self.detecter = Detecter(DeployFile='./yk/detect.prototxt', CaffeModel='./yk/detect.caffemodel',
                        Class=('background', 'hand'), UseGPU=True)
        self.trace_classifier = TraceClassifier(DeployFile='./yk/trace.prototxt', CaffeModel='./yk/trace.caffemodel',
                                       LabelMap='./yk/trace.txt', MeanFile='./yk/trace.npy', UseGPU=True)
        self.gesture_classifier = GestureClassifier(DeployFile='./yk/gesture.prototxt', CaffeModel='./yk/gesture.caffemodel',
                                           LabelMap='./yk/gesture.txt', MeanFile='./yk/gesture.npy', UseGPU=True)

        # trace
        self.trace_length = 1000
        self.trace = [(0, 0) for i in range(self.trace_length)]
        self.trace_valid_point_cnt = 0.0
        self.trace_classify_frame = 0
        self.trace_classify_interval = 20
        self.trace_classify_result = ''

        # gesture
        self.gesture_classify_frame = 0
        self.gesture_classify_interval = 2
        self.gesture_classify_result = ''
        self.gesture_classify_conf_thre = 0.5
        self.gesture_classify_result_quene_len = 10
        self.gesture_classify_result_quene =  [-1 for i in range(self.gesture_classify_result_quene_len)]

        self.frame = 0


    def refresh(self):
        # trace
        self.trace_length = 1000
        self.trace = [(0, 0) for i in range(self.trace_length)]
        self.trace_valid_point_cnt = 0.0
        self.trace_classify_frame = 0
        self.trace_classify_interval = 20
        self.trace_classify_result = ''

        # gesture
        self.gesture_classify_frame = 0
        self.gesture_classify_interval = 2
        self.gesture_classify_result = ''
        self.gesture_classify_conf_thre = 0.5
        self.gesture_classify_result_quene_len = 10
        self.gesture_classify_result_quene = [-1 for i in range(self.gesture_classify_result_quene_len)]

        self.frame = 0

    def main(self, origimg):
        if self.frame > 100000:
            self.frame = 0
        else:
            self.frame += 1

        origimg = cv2.flip(origimg, 1)

        self.detecter.detect_v2(origimg)

        # gesture classify
        """
        if self.frame % self.gesture_classify_interval == 0:
            if self.detecter.tracking_mode:
                # hand_contour, hand_finger_loc, hand_contour_bbox = LocFinger(origimg, detecter.tracking_box)
                # if len(hand_contour):
                if 1:
                    # gesture_sample = genGestureSample(origimg, hand_contour_bbox)
                    gesture_sample = genGestureSample(origimg, ((self.detecter.tracking_box[0], self.detecter.tracking_box[1]),
                                                                (self.detecter.tracking_box[2], self.detecter.tracking_box[3])))
                    hand_finger_loc = (self.detecter.tracking_box[0] + self.detecter.tracking_box[2]) / 2, \
                                      (self.detecter.tracking_box[1] + self.detecter.tracking_box[3]) / 2,
                    # hand_finger_loc = (detecter.tracking_box[0] + detecter.tracking_box[2]) / 2, detecter.tracking_box[
                    #     1]

                    cv2.imshow('gesture_sample', gesture_sample)
                    self.gesture_classifier.forward(gesture_sample)
                    if self.gesture_classifier.conf > self.gesture_classify_conf_thre:
                        self.gesture_classify_result = self.gesture_classifier.label_map[self.gesture_classifier.pred]
                        self.gesture_classify_result_quene.pop(0)
                        self.gesture_classify_result_quene.append(self.gesture_classifier.pred)
                    else:
                        self.gesture_classify_result = 'unidentified.'
                        self.gesture_classify_result_quene.pop(0)
                        self.gesture_classify_result_quene.append(-1)
                else:
                    self.gesture_classify_result = ''
                    self.gesture_classify_result_quene.pop(0)
                    self.gesture_classify_result_quene.append(-1)
            else:
                self.gesture_classify_result = ''
                self.gesture_classify_result_quene.pop(0)
                self.gesture_classify_result_quene.append(-1)
        """

        # trace

        if self.detecter.tracking_mode and Finger(self.gesture_classify_result_quene):
            hand_finger_loc = (self.detecter.tracking_box[0] + self.detecter.tracking_box[2]) / 2, \
                              (self.detecter.tracking_box[1] + self.detecter.tracking_box[3]) / 2,
            new_trace_point_x, new_trace_point_y = hand_finger_loc[0], hand_finger_loc[1]
            cv2.circle(origimg, hand_finger_loc, 5, (0, 255, 0), 3)
            self.trace.pop(0)
            self.trace.append((int(new_trace_point_x), int(new_trace_point_y)))
            self.trace_valid_point_cnt += 1
        else:
            if self.trace[0] != (0, 0):
                self.trace_valid_point_cnt -= 1
            self.trace.pop(0)
            self.trace.append((0, 0))
        if self.detecter.tracking_drift_frames > 40:
            self.trace = [(0, 0) for i in range(self.trace_length)]
            self.trace_valid_point_cnt = 0.0
            self.trace_classify_result = ''

        # gen trace map
        trace_map = np.zeros(shape=origimg.shape, dtype=np.uint8) + 255
        trace_str = ''
        for i, point in enumerate(self.trace):
            if point == (0, 0):
                trace_str += 'x'
            else:
                trace_str += str(i) + '_'
        trace_str_list = re.split(r'x{5,}', trace_str)
        trace_str_list = [s.replace('x', '')[:-1] for s in trace_str_list]
        trace_str_list = [re.split(r'_', s) for s in trace_str_list]
        trace_str_list = [s for s in trace_str_list if len(s) > 2]
        for trace_str in trace_str_list:
            for i in range(1, len(trace_str)):
                cv2.line(trace_map, self.trace[int(trace_str[i])], self.trace[int(trace_str[i - 1])], (0, 0, 0), 6)
                cv2.line(origimg, self.trace[int(trace_str[i])], self.trace[int(trace_str[i - 1])], (0, 255, 0), 4)

        if self.trace_valid_point_cnt > 50 and abs(self.trace_classify_frame - self.frame) > self.trace_classify_interval:
            self.trace_classify_frame = self.frame
            trace_sample = genTraceSample(trace_map)
            # cv2.imshow('trace_sample', trace_sample)
            self.trace_classifier.forward(trace_sample)
            if self.trace_classifier.conf > 0.6:
                self.trace_classify_result = self.trace_classifier.label_map[self.trace_classifier.pred]
            else:
                self.trace_classify_result = 'unidentified.'

        # show origimg
        for i in range(len(self.detecter.box)):
            if not self.detecter.deleted[i]:
                p1 = (self.detecter.box[i][0], self.detecter.box[i][1])
                p2 = (self.detecter.box[i][2], self.detecter.box[i][3])
                if self.detecter.cls[i] == 1:
                    cv2.rectangle(origimg, p1, p2, (255, 0, 0))
                elif self.detecter.cls[i] == 2:
                    cv2.rectangle(origimg, p1, p2, (0, 255, 0))
        if self.detecter.tracking_mode and self.detecter.tracking_box:
            cv2.rectangle(origimg, (self.detecter.tracking_box[0], self.detecter.tracking_box[1]),
                          (self.detecter.tracking_box[2], self.detecter.tracking_box[3]), (0, 0, 255))
        cv2.putText(origimg, 'gesture : %s' % (self.gesture_classify_result), (20, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
        cv2.putText(origimg, 'trace : %s' % (self.trace_classify_result), (20, 100), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
        cv2.imshow('origimg', origimg)

        # show tracemap
        # cv2.putText(trace_map, 'trace : %s' % (self.trace_classify_result), (20, 60), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
        # cv2.imshow('tracemap', trace_map)

