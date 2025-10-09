## NeuroSync 
- https://github.com/AnimaVR/NeuroSync_Player?tab=readme-ov-file

```
创建环境
conda create -n neurosync python=3.10
conda activate neurosync
安装cuda
 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121    
 pip install flask   
pip install librosa


VSCode配置NeuroSync_Player-main
 Ctrl+Shift+P配置Python为neurosync对应的python
 可创建launch.json调试当前文件
 选择play_generated_files.py为当前文件
 按F5启动会出现如下提示信息：
   (neurosync) PS D:\UE\Expressions\NeuroSync\NeuroSync_Player-main>  & 'd:\Anaconda3\envs\neurosync\python.exe' 'c:\Users\liang\.vscode\extensions\ms-python.debugpy-2025.10.0-win32-x64\bundled\libs\debugpy\launcher' '60365' '--' 'D:\UE\Expressions\NeuroSync\NeuroSync_Player-main\play_generated_files.py' 
   pygame 2.6.1 (SDL 2.28.4, Python 3.10.18)
   Hello from the pygame community. https://www.pygame.org/contribute.html
   Loaded 3 animations for emotion 'Angry'
   Loaded 3 animations for emotion 'Disgusted'
   Loaded 3 animations for emotion 'Fearful'
   Loaded 1 animations for emotion 'Happy'
   Loaded 1 animations for emotion 'Neutral'
   Loaded 3 animations for emotion 'Sad'
   Loaded 1 animations for emotion 'Surprised'
   Available generated files:
   1: Audio: generated\30c1e660-973e-4551-9af8-f18b4c850ef3\audio.wav, Shapes: generated\30c1e660-973e-4551-9af8-f18b4c850ef3\shapes.csv
   2: Audio: generated\6d49e0e6-d8ef-4fb2-afcf-aca72450ac9d\audio.wav, Shapes: generated\6d49e0e6-d8ef-4fb2-afcf-aca72450ac9d\shapes.csv
   Enter the number of the file to play, or 'q' to quit: 
 按1或者2 UE中可看到口型讲话
 修改play_generated_files.py中line 20, ENABLE_EMOTE_CALLS = False -> ENABLE_EMOTE_CALLS = True



文档和代码 https://github.com/AnimaVR/NeuroSync_Player?tab=readme-ov-file
有个视频演示怎么配置 https://www.youtube.com/watch?v=qN-CSqNEhmk

3部分代码：
NeuroSync_Local_API
NeuroSync_Player
NEUROSYNC_Demo_Project - ue示例工程

NeuroSync_Local_API为本地服务可以audio to facial blendshapes。 那么推理部分在这里。
   python neuro_sync_local_api.py
NeuroSync_Player可以发faical blendshapes数据给UE， UE通过Livelink驱动角色表情。
  python play_generated_files.py

play_generated_files.py line 20 有个ENABLE_EMOTE_CALLS的全局变量，似乎可以还可以整合emotion。
  emotion试了一下API调的是127.0.0.1:7777，将NeuroSync_Local_API端口改为7777， NeuroSync_Player 那边按1发送会得到如下结果
  127.0.0.1 - - [09/Oct/2025 10:14:39] code 400, message Bad request syntax ('startspeaking')
  127.0.0.1 - - [09/Oct/2025 10:14:39] "startspeaking" HTTPStatus.BAD_REQUEST -
  127.0.0.1 - - [09/Oct/2025 10:15:02] code 400, message Bad request syntax ('stopspeaking')
  127.0.0.1 - - [09/Oct/2025 10:15:02] "stopspeaking" HTTPStatus.BAD_REQUEST -

model.pth大概900多M有点大
目前的示例是通过Livelink方式（貌似只能编辑器模式用），需要
还有其他几个开源代码没去试
https://github.com/its-DeFine/NeuroSync-Core
https://github.com/admin-noosphere/NeuroSync_Real-Time_API




```



- Runtime-metahuman-lip-sync https://docs.georgy.dev/runtime-metahuman-lip-sync

- HumanAudio2Face https://www.fab.com/listings/f0f53566-4e15-43ce-9bef-fcc68a4dd07f
