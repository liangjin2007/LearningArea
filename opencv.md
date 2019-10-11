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


