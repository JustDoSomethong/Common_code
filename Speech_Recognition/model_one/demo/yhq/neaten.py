import sys
import cv2
import numpy as np
import math
caffe_root = '/home/pro/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import gc
class FaceDetection:
    def init(self):
        self.net = caffe.Net('./yhq/MobileNetSSD_deploy.prototxt', './yhq/mobilenet_iter_145000.caffemodel', caffe.TEST)
    def preprocess(self,src):
        img = cv2.resize(src, (300, 300))
        img = img - 127.5
        img = img * 0.007843
        return img

    def postprocess(self,img, out):
        h = img.shape[0]
        w = img.shape[1]
        box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0, 0, :, 1]
        conf = out['detection_out'][0, 0, :, 2]
        return (box.astype(np.int32), conf, cls)

    def detect(self,origimg):
        img = self.preprocess(origimg)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = img
        out = self.net.forward()
        box, conf, cls = self.postprocess(origimg, out)
        for i in range(len(box)):
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            p3 = (max(p1[0], 15), max(p1[1], 15))
        return box
class Landmark:
    def init(self):
        caffe_model_path = "./yhq"
        self.LNet = caffe.Net(caffe_model_path + "/SE-BN-Inception.prototxt",
                         caffe_model_path + "/landmark_iter_40000.caffemodel", caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': [1, 3, 224, 224]})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.load('./yhq/mean.npy').mean(1).mean(1))

    def detectlandmark(self,image):
        # image=image*0.00392157
        # imga=transformer.preprocess('data', image)
        # print "imga",transformer.preprocess('data', image)
        # print transformer.preprocess('data', image)[0][0]
        self.LNet.blobs['data'].data[0, ...] = self.transformer.preprocess('data', image)
        out = self.LNet.forward()
        return self.LNet.blobs['classifier'].data[0]
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R
def headpose(shape,frame):
    c_x = 640 / 2
    c_y = 480 / 2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    landmarks_3D = np.float32([[-69.69999130857637, -33.25332101835264, 33.25537187146924],#0
   [ -69.1295285520978, -13.03442942699502, 35.25333161282494],#1
   [ -66.31939952592323, 7.250066692386407, 38.16297541867183],#2
   [ -62.09589991670153, 26.50001871014885, 39.02784810084581],#3
   [ -55.4191014304546, 43.87911993210747, 32.0574059883217],#4
   [ -44.9492457391497, 58.13746018875266, 21.99590368295328],#5
   [ -33.1003068046949, 69.33053061741228, 11.4134393463586],#6
   [ -18.06223537868813, 76.87427185837321, 0.9148196698774846],#7
   [ -1.115106793569509, 78.15516164669779, -4.17664167636272],#8
   [ 15.27169625973218, 76.15088941107855, 1.306175189584364],#9
   [ 29.49848916017946, 67.12602984035345, 11.6627097113174],#10
   [ 42.23619778223933, 55.25836430996625, 22.74079689458464],#11
   [ 52.52386307496216, 40.67325517544981, 28.07411654313706],#12
   [ 59.50792137673068, 23.54050054666086, 30.31291106318122],#13
   [ 64.17044379251567, 5.044870954649348, 31.00594689363578],#14
   [ 67.39603256136819, -14.2487902316755, 31.05914294668592],#15
   [ 68.84029923684656, -33.58565695252281, 32.43679441843813],#16
   [ -59.60493043687602, -58.08938832280548, 11.38848778068019],#17
   [ -50.39343020740523, -67.06114299848011, 6.290881706961114],#18
   [ -37.38816142458451, -70.22859625496686, 0.6518277549407706],#19
   [ -24.09721462894306, -69.20179733379294, -4.693403773139972],#20
   [ -12.26378850858996, -64.95354026374189, -7.35811121169569],#21
   [ 13.64271935738131, -65.26850810612247, -4.688649802408184],#22
   [ 26.64840404442134, -70.21553024075385, -1.329881756461543],#23
   [ 39.04358426379751, -71.36267413891478, 3.123080944675039],#24
   [ 51.01093778012456, -68.58961315746562, 6.857393384226505],#25
   [ 59.09369604931143, -60.51627106094444, 11.1412417492051],#26
   [ 0.6538192141320591, -46.98922766952771, -8.806696412452027],#27
   [ 0.6968809148100985, -34.16134733000608, -15.48242022291193],#0
   [ 0.7848205799347088, -21.39480540552694, -22.18137154824895],#0
   [ 0.8870293951743842, -8.314337623138483, -28.73027236534417],#0
   [ -13.52446701693809, 4.631325611046393, -14.68318521716817],#0
   [ -6.772890755991296, 6.813964229800471, -17.34494685213242],#0
   [ 0.1693427293909304, 8.602149928596781, -18.66979978381017],#0
   [ 7.655233516544413, 6.931474037428265, -17.05613683675121],#0
   [ 14.29882476872998, 4.624492314402197, -14.76131473843574],#0
   [ -45.67509123535952, -41.31566212029464, 6.00155188682348],#0
   [-38.00651375208525, -47.84197347174543, 2.887423130392905],#0
   [ -27.65671143381301, -47.84611632854482, 1.008777557479486],#0
   [ -20.0117140918183, -41.30493814141874, 0.1455417331919856],#0
   [ -28.24470582299646, -38.34701411595412, 0.4457148093352578],#0
   [-38.44548151285451, -38.02889992341376, 2.450849040273283],#0
   [ 20.4317232901593, -42.07789260676977, 0.9985515902990141],#0
   [ 27.90565902987759, -48.98266148106386, 1.568027446446173],#0
   [ 37.94010058890506, -48.80605053283252, 2.882052325715922],#0
   [ 44.8454721891369, -42.81769644495365, 6.377259167103603],#0
   [ 38.73913372878778, -39.32084752921348, 2.284832485144917],#0
   [29.1175109712184, -39.44475481945177, 0.9213344748759047],#0
   [ -24.79004681023952, 38.39556803546339, -6.963534013623576],#0
   [ -15.23713981426707, 30.13079025207708, -14.85327398842435],#0
   [-6.210256776506763, 25.79470707746639, -17.62069059468883],#0
   [ 0.6612304712681285, 27.71790406594837, -18.43547642206583],#0
   [ 7.726109971130182, 25.85925172528223, -17.64440029561435],#0
   [ 16.90636739488539, 30.17954383031108, -13.12267077343517],
   [ 24.36425916722986, 37.26555011168971, -6.643746501422994],
   [ 17.04658514565728, 43.78408736049108, -12.49560696940935],
   [ 8.170348970020413, 46.30090707884659, -15.78154105922123],
   [ 0.5174181829877942, 46.85682050684967, -17.25622213462919],
   [ -7.109655822594155, 46.28709349735755, -16.52601132365958],
   [-16.05509521849268, 43.96421857745796, -13.77914373450479],
   [-19.89116529983598, 38.00716496852058, -10.24378005129918],
   [ -6.48727002459057, 34.45392476649366, -15.7608193851608],
   [ 0.599755014891602, 34.76159245705267, -16.5484633265928],
   [7.83735470574369, 34.20866945008358, -15.71571549156865],
   [ 20.16951482940803, 37.41373973401485, -9.034526023388059],
   [ 7.444716416839573, 34.99371508147625, -16.05637962051822],
   [ 0.1972194058077559, 35.65813448067756, -17.15441875513363],
   [-6.89415979090469, 35.04617052928307, -16.5052765312243]]#67
)
    landmarks_2D= np.empty((68,2))
    for i in range(68):
        landmarks_2D[i][0]=shape[2*i]
        landmarks_2D[i][1] = shape[2 * i+1]
    axis = np.float32([[100, 0, 0],
                    [0, 100, 0],
                    [0.6538192141320591, -46.98922766952771, -58.806696412452027]])
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix, camera_distortion)

    z_x=math.sqrt(pow(tvec[0],2)+pow(tvec[2],2))
    eul_x=math.atan2(tvec[0],z_x)
    z_y=math.sqrt(pow(tvec[1],2)+pow(tvec[2],2))
    eul_y = math.atan2(tvec[1], z_y)
    eul=np.array([eul_x,eul_y,0.0])
    camera_rotation=eulerAnglesToRotationMatrix(eul)
    #camera_rotation, _ = cv2.Rodrigues(rvec)
    #rotationMatrixToEulerAngles(rmat)
    #print eul_x,eul_y
    scale=f_x/tvec[2]


    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
    sellion_xy = (int(landmarks_2D[27][0]), int(landmarks_2D[27][1]))
    if rvec[2]>0:
        rvec=0-rvec
    if abs(rvec[2])>1:
        R=eulerAnglesToRotationMatrix(rvec)
        ori=[0,0,1]
        R=np.dot(camera_rotation,R)
        headpose=np.dot(R,ori)

        trix=tvec[2]/headpose[2]*(headpose[1])
        triy=tvec[2]/headpose[2]*headpose[0]

        pox=trix+tvec[0]
        poy=triy-tvec[1]

        if abs(pox)<200 and abs(poy)<400:
            print "1",pox,poy
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # RED
        else:
            print "0",pox,poy
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # RED

    return frame


def gazenet_init():
    caffe.set_mode_gpu()
    facedetec=FaceDetection()
    facedetec.init()
    landmark=Landmark()
    landmark.init()
    return facedetec,landmark
def gazedetect(img,facedetec,landmark):
    boundingboxes = facedetec.detect(img)
    maxlen = 0
    detmax = []
    for det in boundingboxes:
        x1 = int(det[0])
        y1 = int(det[1])
        x2 = int(det[2])
        y2 = int(det[3])
        ##
        wid = int(max(x2 - x1, y2 - y1) / 2)
        if maxlen < wid:
            detmax = det
            maxlen = wid
    det = detmax
    # print det
    if len(det) != 0:
        x1 = int(det[0])
        y1 = int(det[1])
        x2 = int(det[2])
        y2 = int(det[3])
        ##
        wid = int(max(x2 - x1, y2 - y1) / 2)
        xmid = (x1 + x2) / 2
        ymid = (y1 + y2) / 2
        x1 = xmid - wid
        x2 = xmid + wid
        y1 = ymid - wid
        y2 = ymid + wid
        ##
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img0 = img[y1:y2 + 1, x1:x2 + 1, ]
        img0 = cv2.resize(img0, (224, 224))
        img0 = img0.astype(np.float32)
        points = landmark.detectlandmark(img0)
        shape = []
        point_pair_l = len(points)
        for i in range(point_pair_l / 2):
            x = int(points[2 * i] / 112.0 * wid) + xmid - wid
            y = int(points[2 * i + 1] / 112.0 * wid) + ymid - wid
            shape.append(x)
            shape.append(y)
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 1)
        shape = []
        point_pair_l = len(points)
        for i in range(point_pair_l / 2):
            x = int(points[2 * i] / 112.0 * wid) + xmid - wid
            y = int(points[2 * i + 1] / 112.0 * wid) + ymid - wid
            shape.append(x)
            shape.append(y)
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 1)
        img = headpose(shape, img)
    cv2.imshow("1", img)
 


    
if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    caffe.set_mode_gpu()
    facedetec=FaceDetection()
    facedetec.init()
    Landmark=Landmark()
    Landmark.init()
    while (1):
        ret, img = cap.read()
        boundingboxes = facedetec.detect(img)
        maxlen = 0
        detmax = []
        for det in boundingboxes:
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])
            ##
            wid = int(max(x2 - x1, y2 - y1) / 2)
            if maxlen < wid:
                detmax = det
                maxlen = wid
        det = detmax
        # print det
        if len(det) != 0:
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])
            ##
            wid = int(max(x2 - x1, y2 - y1) / 2)
            xmid = (x1 + x2) / 2
            ymid = (y1 + y2) / 2
            x1 = xmid - wid
            x2 = xmid + wid
            y1 = ymid - wid
            y2 = ymid + wid
            ##
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img.shape[1]: x2 = img.shape[1]
            if y2 > img.shape[0]: y2 = img.shape[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            img0 = img[y1:y2 + 1, x1:x2 + 1, ]
            img0 = cv2.resize(img0, (224, 224))
            img0 = img0.astype(np.float32)
            points = Landmark.detectlandmark(img0)
            shape = []
            point_pair_l = len(points)
            for i in range(point_pair_l / 2):
                x = int(points[2 * i] / 112.0 * wid) + xmid - wid
                y = int(points[2 * i + 1] / 112.0 * wid) + ymid - wid
                shape.append(x)
                shape.append(y)
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 1)
            shape = []
            point_pair_l = len(points)
            for i in range(point_pair_l / 2):
                x = int(points[2 * i] / 112.0 * wid) + xmid - wid
                y = int(points[2 * i + 1] / 112.0 * wid) + ymid - wid
                shape.append(x)
                shape.append(y)
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 1)
            img = headpose(shape, img)
        cv2.imshow("1", img)
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break
    del facedetec
    del Landmark
    gc.colloct()
