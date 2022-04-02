@echo off
for %%i in (*.txt) do (
   echo %%i
   echo %%~fi
   echo %%~di
   echo %%~pi
   echo %%~ni
   echo %%~xi
   
   echo %%~ti
   echo %%~zi
)

那么，此时就可以通过一些特殊命令来取得文件的相关信息，比如：

%%~fi：表示获取该文件的绝对路径信息
%%~di：表示获取该文件所在的盘符
%%~pi：表示获取该文件的路径，不包含盘符的信息
%%~ni：表示获取该文件的文件名，不包含扩展名信息
%%~xi：表示获取该文件的扩展名
%%~ti：表示获取该文件的上次修改时间
%%~zi：表示获取该文件的大小



set va=5
echo %variable%



xcopy /R /Y source dest


::@echo off  
rem 正在搜索...  
rem 删除文件  
for /f "delims=" %%i in ('dir /b /a-d /s "*.pb.cc"') do del %%i  
rem 删除完毕  
pause
