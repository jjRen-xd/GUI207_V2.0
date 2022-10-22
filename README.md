~~~
# 文件夹结构说明
---
PATH_ROOT
|
|___api/           <!-- 存放接口文件，接口主要用于于业务逻辑提供数据操作 -->
|   |___... **预留,暂时不管**
|
|___build/         <!-- Qmake编译路径 -->
|   |___build-GUI_V1.0-Desktop_Qt_6_2_4_MinGW_64_bit-Debug/
|   |___... **编译路径位置**
|
|___conf/          <!-- 存放GUI美化,qss等配置文件 -->
|   |___QRibbon/
|   |___QRibbon_yue_CN.ts
|
|___core/          <!-- 存放QT GUI业务核心逻辑代码 -->
|   |___MainWindow.h
|   |___MainWindow.cpp      **对MainWindow的修改尽可能小**
|   |___datasetsWindow/     **数据预览界面区域的功能逻辑（比如按钮的逻辑）通过自定义.h引入MainWindow**
|   |   |___XXX.h
|   |   |___XXX.cpp
|   |___modelsWindow/       **模型预览界面区域的功能逻辑（比如按钮的逻辑）通过自定义.h引入MainWindow**
|   |   |___XXX.h
|   |   |___XXX.cpp
|
|___db/            <!-- 存放数据集及模型 -->
|   |___datasets/
|   |   |___HRRP_20220508/  **数据集的命名一定要规范,形式自定,注意txt**
|   |   |   |___HRRP_XXX_20220508.txt
|   |   |   |___...
|   |   |___...
|   |___models/             **模型的命名一定要规范,形式自定,注意txt**
|       |___XXX__20220508.pt
|       |___XXX__20220508.txt
|       |___...
|
|___lib/           <!-- 存放程序中的自定义模块 -->
|   |___guiLogic/   **code here,例如曲线绘制的核心代码,留好接口**
|   |___algorithm/  **code here,例如模型调用的核心代码,留好接口**
|
|___sources/       <!-- 存放QT图像资源文件 -->
|   |___icon/       **图标位置**
|   |   |___...
|   |___MainWindow_sources.qrc
|
|___uis/           <!-- 存放QT UI文件 -->
|   |___MainWindow.ui
|
|___GUI_V1.0.pro   <!-- QT Creator工程配置文件 -->
|
|___main.cpp       <!-- 主程序入口 -->
|
|___.gitignore     <!-- gitignore -->
|
|___README.md      <!-- readme -->

~~~
