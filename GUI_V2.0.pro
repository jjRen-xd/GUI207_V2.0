QT       += core gui charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
CONFIG += exceptions #open try catch
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += _CRT_SECURE_NO_WARNINGS #avoid monitorPage.cpp's localtime waring
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    build/conf/QRibbon/QRibbon.h \
    core/MainWindow.h \
    core/datasetsWindow/chart.h \
    core/datasetsWindow/datasetDock.h \
    core/modelCAMPage.h \
    core/modelChoicePage.h \
    core/modelEvalPage.h \
    core/modelTrainPage.h \
    core/modelVisPage.h \
    core/modelsWindow/modelDock.h \
    core/monitorPage.h \
    core/sensePage.h \
    build/lib/algorithm/inferthread.h \
    build/lib/algorithm/logging.h \
    build/lib/algorithm/trtinfer.h \
    build/lib/dataprocess/matdataprocess.h \
    build/lib/dataprocess/customdataset.h \
    build/lib/dataprocess/matdataprocess_abfc.h \
    build/lib/guiLogic/bashTerminal.h \
    build/lib/guiLogic/customWidget/imagewidget.h \
    build/lib/guiLogic/datasetInfo.h \
    build/lib/guiLogic/modelInfo.h \
    build/lib/guiLogic/tinyXml/tinystr.h \
    build/lib/guiLogic/tinyXml/tinyxml.h \
    build/lib/guiLogic/tools/convertTools.h \
    build/lib/guiLogic/tools/searchFolder.h \
    build/lib/guiLogic/tools/guithreadrun.h \
    build/lib/guiLogic/tools/searchFolder.h \
    build/lib/TRANSFER/ToHrrp.h \
    build/lib/guiLogic/tools/socketclient.h \
    build/lib/guiLogic/tools/socketserver.h

SOURCES += \
    core/modelCAMPage.cpp \
    core/modelVisPage.cpp \
    build/lib/guiLogic/customWidget/imagewidget.cpp \
    build/lib/algorithm/inferthread.cpp \
    build/lib/guiLogic/tools/socketclient.cpp \
    build/lib/guiLogic/tools/socketserver.cpp \
    main.cpp \
    build/conf/QRibbon/QRibbon.cpp \
    core/MainWindow.cpp \
    core/datasetsWindow/chart.cpp \
    core/datasetsWindow/datasetDock.cpp \
    core/modelChoicePage.cpp \
    core/modelEvalPage.cpp \
    core/modelTrainPage.cpp \
    core/monitorPage.cpp \
    core/modelsWindow/modelDock.cpp \
    core/sensePage.cpp \
    build/lib/dataprocess/matdataprocess.cpp \
    build/lib/dataprocess/matdataprocess_abfc.cpp \
    build/lib/algorithm/trtinfer.cpp \
    build/lib/guiLogic/tools/guithreadrun.cpp \
    build/lib/guiLogic/bashTerminal.cpp \
    build/lib/guiLogic/datasetInfo.cpp \
    build/lib/guiLogic/modelInfo.cpp \
    build/lib/guiLogic/tinyXml/tinystr.cpp \
    build/lib/guiLogic/tinyXml/tinyxml.cpp \
    build/lib/guiLogic/tinyXml/tinyxmlerror.cpp \
    build/lib/guiLogic/tinyXml/tinyxmlparser.cpp \
    build/lib/guiLogic/tools/searchFolder.cpp

FORMS += \
    ./build/conf/QRibbon/qribbon.ui \
    ./uis/MainWindow.ui

TRANSLATIONS += \
    ./build/conf/QRibbon_yue_CN.ts

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    ./build/conf/QRibbon/QRibbon.qrc \
    ./sources/MainWindow_sources.qrc \
    build/conf/QRibbon/QRibbon.qrc \
    sources/MainWindow_sources.qrc

include("./build/conf/libtorch.pri")

RC_ICONS = "./sources/LOGO.ico"

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRP
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRPd
#else:unix: LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRP

#INCLUDEPATH += $$PWD/lib/TRANSFER
#DEPENDPATH += $$PWD/lib/TRANSFER
