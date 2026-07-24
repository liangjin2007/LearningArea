# 一键启动带所有选项的磁盘清理
cleanmgr /d C:

# 1. Dism 清理 WinSxS（推荐，安全）
Dism.exe /online /Cleanup-Image /StartComponentCleanup /ResetBase

# 2. 清理临时文件
$paths = @(
    "$env:TEMP\*",
    "$env:WINDIR\Temp\*",
    "$env:WINDIR\Prefetch\*",
    "C:\Windows\SoftwareDistribution\Download\*"
)
Remove-Item $paths -Recurse -Force -ErrorAction SilentlyContinue

# 3. 清理用户缓存
$cachePaths = @(
    "$env:LOCALAPPDATA\Microsoft\Windows\INetCache\*",
    "$env:LOCALAPPDATA\Microsoft\TerminalServerClient\Cache\*",
    "$env:LOCALAPPDATA\Microsoft\OneDrive\logs\*",
    "$env:APPDATA\Microsoft\Teams\Cache\*",
    "$env:APPDATA\Microsoft\Teams\blob_storage\*"
)
Remove-Item $cachePaths -Recurse -Force -ErrorAction SilentlyContinue

