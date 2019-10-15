## OpenCV学习

- MSER检测(最大稳定极值区域)![mser](https://github.com/liangjin2007/data_liangjin/blob/master/opencv_mser.jpg?raw=true)
  - 仿射不变性
  - 区域增长
  - 类似于分水岭算法
  - 多级灰度threshold
```
import numpy as np
import cv2 as cv
import video
import sys

if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = video.create_capture(video_src)
    mser = cv.MSER_create()

    while True:
        ret, img = cam.read()
        if ret == 0:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        vis = img.copy()

        regions, _ = mser.detectRegions(gray)
        hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv.polylines(vis, hulls, 1, (0, 255, 0))

        cv.imshow('img', vis)
        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
```

- Blob检测 用于几何形状提取和分析
  - 圆度
  - 

- Edge Detection
  - 求偏导数
    - Scharr
    - Sobel
  - Canny Detection
    - threshold
  
- dft
  "An Intuitive Explanation of Fourier Theory"
  - 离散傅立叶变换
    - 任何信号（我们的case是可视图像）都可以表示成一系列sinusoids函数的和。
    - 输入信号包含三个部分： 空间频率frequency，振幅magnitude，相位phase
    - 从简单到复杂地去理解dft:
        - 1D Fourier Transform: 如果输入是一个sinusoid, 那么输出是一个单个的peak at point f。
        - 2D Fourier Transform: 输出图像中 横轴为原来图像中沿着x方向的频率， 纵轴为原图中沿着y方向的频率， 亮度为原图中的亮度对比度。
    - DC Term:对应于零频率，代表平均亮度
    - nyquist frequency

- facial_features
  可以指定眼睛等位置

- facedetect
  脸部检测
  
- pyramids
  生成图像的金字塔
  
- convexhull
  算一堆点的凸包
  
- morphology
腐蚀/膨胀/开/闭
  - 开运算：先腐蚀再膨胀，用周围原色填补白色小洞。清楚物体外的小孔洞，闭运算填补物体内的小孔洞。
```
st = cv.getStructuringElement(getattr(cv, str_name), (sz, sz))
res = cv.morphologyEx(img, getattr(cv, oper_name), st, iterations=iters)
```
- mouse_and_match
模版匹配，鼠标交互
```
patch = gray[sel[1]:sel[3],sel[0]:sel[2]]
result = cv.matchTemplate(gray,patch,cv.TM_CCOEFF_NORMED)
result = np.abs(result)**3
_val, result = cv.threshold(result, 0.01, 0, cv.THRESH_TOZERO)
result8 = cv.normalize(result,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
cv.imshow("result", result8)
```
    
- video_threaded
  - 线程池创建
  - 线程池消费和生产
  - 循环怎么写，异步操作
```
from multiprocessing.pool import ThreadPool
from collections import deque
from common import clock, draw_str, StatValue
import video

if __name__ == '__main__':
    import sys
    cap = video.create_capture(0)

    def process_frame(frame, t0):
        # some intensive computation...
        frame = cv.medianBlur(frame, 19)
        frame = cv.medianBlur(frame, 19)
        return frame, t0

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()
    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        while len(pending) > 0 and pending[0].ready(): # 获取队列第一个。
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value*1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv.imshow('threaded video', res)
        if len(pending) < threadn:
            ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
            pending.append(task)
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break
cv.destroyAllWindows()
```

- watershed
```
from common import Sketcher
sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)
sketch.show()
sketch.dirty = True 

m = self.markers.copy()
cv.watershed(self.img, m)
overlay = self.colors[np.maximum(m, 0)]
vis = cv.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
cv.imshow('watershed', vis)
```

- turing
https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/

- floodfill
```
cv.floodFill(flooded, mask, seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
cv.circle(flooded, seed_pt, 2, (0, 0, 255), -1)
cv.imshow('floodfill', flooded)
```

- kmeans
```
def make_gaussians(cluster_n, img_size):
    points = []
    ref_distrs = []
    for _i in xrange(cluster_n):
        mean = (0.1 + 0.8*random.rand(2)) * img_size
        a = (random.rand(2, 2)-0.5)*img_size*0.1
        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
        n = 100 + random.randint(900)
        pts = random.multivariate_normal(mean, cov, n)
        points.append( pts )
        ref_distrs.append( (mean, cov) )
    points = np.float32( np.vstack(points) )
    return points, ref_distrs
    
def draw_gaussain(img, mean, cov, color):
    x, y = np.int32(mean)
    w, u, _vt = cv.SVDecomp(cov)
    ang = np.arctan2(u[1, 0], u[0, 0])*(180/np.pi)
    s1, s2 = np.sqrt(w)*3.0
    cv.ellipse(img, (x, y), (s1, s2), ang, 0, 360, color, 1, cv.LINE_AA)

points, _ = make_gaussians(cluster_n, img_size)
        
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
ret, labels, centers = cv.kmeans(points, cluster_n, None, term_crit, 10, 0)

img = np.zeros((img_size, img_size, 3), np.uint8)
for (x, y), label in zip(np.int32(points), labels.ravel()):
    c = list(map(int, colors[label]))

    cv.circle(img, (x, y), 1, c, -1)

cv.imshow('gaussian mixture', img)
```

- edge
Canny Detection
```
edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
vis = img.copy()
vis = np.uint8(vis/2.)
vis[edge != 0] = (0, 255, 0)
```

- letter_recog.py 
这个是训练模型的例子。使用cv.ml中的RTree, KNeaerest, Boost, SVM, MLP，默认是训练Random Trees classifier.
model=cv2.ml.RTrees_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

- browse.py
这个是图片放大器。
small = cv.pyrDown(small)
cv.getRectSubPix(img, (800,600),(x+0.5, y+0.5))

- tst_scene_render.py
演示cv.fillConvexPoly，非常快的一个接口

- peopledetect.py
```
hog = cv.HOGDescriptor()
hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )  
found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
```

- hist.py
```
hist_item = cv.calcHist([im],[ch],None,[256],[0,256])
print(hist_item.shape) # (256,1)
cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
hist=np.int32(np.around(hist_item))
pts = np.int32(np.column_stack((bins,hist)))
cv.polylines(h,[pts],False,col)
```

- contours.py
levels设成7能显示所有大小的轮廓
```
contours0, hierarchy = cv.findContours( img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

def update(levels):
    vis = np.zeros((h, w, 3), np.uint8)
    levels = levels - 3
    cv.drawContours( vis, contours, (-1, 2)[levels <= 0], (128,255,255),
        3, cv.LINE_AA, hierarchy, abs(levels) )
    cv.imshow('contours', vis)
```

- find_obj.py
给定图片，根据图片特征去另一张图片中寻找图片

- fitline.py
DT_L2效果最差
```
func = getattr(cv, cur_func_name)
vx, vy, cx, cy = cv.fitLine(np.float32(points), func, 0, 0.01, 0.01)
cv.line(img, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255))
```

- squares.py
```
squares = find_squares(img)
cv.drawContours( img, squares, -1, (0, 255, 0), 3 )

if thrs == 0:
    bin = cv.Canny(gray, 0, 50, apertureSize=5)
    bin = cv.dilate(bin, None)
else:
    _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    cnt_len = cv.arcLength(cnt, True)
    cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
    if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
        cnt = cnt.reshape(-1, 2)
        max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
        if max_cos < 0.1:
            squares.append(cnt)
```

- dft.py
  - 计算dft得到实部和虚部，计算magnitude, cv.log(1+magnitude)将dft图像移到中心，cv.normalize归一化。
  - 实际图像x,y两个方向都表示频率。具体的dft值表示

- calibrate.py？
相机标定：输入棋盘格扭曲图像，计算参数矩阵，undistort扭曲图像。求的是内参？还是外参？
接口：
```
found, corners = cv.findChessboardCorners(img, pattern_size)
cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
cv.drawChessboardCorners(vis, pattern_size, corners, found)
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
```

- camera_calibration_show_extrinsics.py
用matplotlib ax.plot3D()把矩阵画出来

- digits.py
err怎么定义
confusion matrix怎么定义









