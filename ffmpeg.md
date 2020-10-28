# 
- 视频转图片 ffmpeg -i .\SHGN7_S001_S001_T156.MOV -r 1 -q 0 .\%08d.jpg
- 逆时针旋转且视频转图片 ffmpeg -i .\SHGN7_S001_S001_T156.MOV -r 1 -q 0 -vf transpose=2 .\%08d.jpg
