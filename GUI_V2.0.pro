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
    conf/QRibbon/QRibbon.h \
    core/MainWindow.h \
    core/datasetsWindow/chart.h \
    core/datasetsWindow/datasetDock.h \
    core/modelChoicePage.h \
    core/modelEvalPage.h \
    core/modelsWindow/modelDock.h \
    core/modelTrainPage.h \
    core/monitorPage.h \
    core/sensePage.h \
    core/modelTrain.h \
    #lib/algorithm/libtorchTest.h \
    #lib/algorithm/onnxinfer.h \
    lib/algorithm/logging.h \
    lib/algorithm/trtinfer.h \
    lib/guiLogic/bashTerminal.h \
    lib/guiLogic/datasetInfo.h \
    lib/guiLogic/modelInfo.h \
    lib/guiLogic/tinyXml/tinystr.h \
    lib/guiLogic/tinyXml/tinyxml.h \
    lib/guiLogic/tools/guithreadrun.h \
    lib/guiLogic/tools/realtimeinferencebuffer.h \
    lib/guiLogic/tools/searchFolder.h \
    lib/TRANSFER/ToHRRP.h \
    lib/guiLogic/tools/socketserver.h

SOURCES += \
    core/monitorPage.cpp \
    lib/guiLogic/tools/realtimeinferencebuffer.cpp \
    lib/guiLogic/tools/socketserver.cpp \
    main.cpp \
    core/modeltrainpage.cpp \
    conf/QRibbon/QRibbon.cpp \
    core/MainWindow.cpp \
    core/datasetsWindow/chart.cpp \
    core/datasetsWindow/datasetDock.cpp \
    core/modelChoicePage.cpp \
    core/modelEvalPage.cpp \
    core/modelTrainPage.cpp \
    core/modelTrain.cpp \
    core/modelsWindow/modelDock.cpp \
    core/sensePage.cpp \
    #lib/algorithm/libtorchTest.cpp \
    #lib/algorithm/onnxinfer.cpp \
    lib/algorithm/trtinfer.cpp \
    lib/guiLogic/tools/guithreadrun.cpp \
    lib/guiLogic/bashTerminal.cpp \
    lib/guiLogic/datasetInfo.cpp \
    lib/guiLogic/modelInfo.cpp \
    lib/guiLogic/tinyXml/tinystr.cpp \
    lib/guiLogic/tinyXml/tinyxml.cpp \
    lib/guiLogic/tinyXml/tinyxmlerror.cpp \
    lib/guiLogic/tinyXml/tinyxmlparser.cpp \
    lib/guiLogic/tools/searchFolder.cpp

FORMS += \
    ./conf/QRibbon/qribbon.ui \
    ./uis/MainWindow.ui

TRANSLATIONS += \
    ./conf/QRibbon_yue_CN.ts

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    ./conf/QRibbon/QRibbon.qrc \
    ./sources/MainWindow_sources.qrc \
    conf/QRibbon/QRibbon.qrc \
    sources/MainWindow_sources.qrc

include("./conf/libtorch.pri")

RC_ICONS = "./sources/LOGO.ico"

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRP
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRPd
#else:unix: LIBS += -L$$PWD/lib/TRANSFER/ -lToHRRP

#INCLUDEPATH += $$PWD/lib/TRANSFER
#DEPENDPATH += $$PWD/lib/TRANSFER
