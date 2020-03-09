QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

#PKGCONFIG += opencv

SOURCES += \
        CsvReader.cpp \
        CsvWriter.cpp \
        Mesh.cpp \
        Model.cpp \
        ModelRegistration.cpp \
        PnPProblem.cpp \
        RobustMatcher.cpp \
        Utils.cpp \
        main.cpp


#INCLUDEPATH+= /home/parallels/libs/opencv/include

#LIBS += -L/home/parallels/libs/opencv/build/lib/*
#  LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_imgcodecs -lopencv_videoio -lopencv_objdetect -lopencv_calib3d

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv

HEADERS += \
    CsvReader.h \
    CsvWriter.h \
    Mesh.h \
    Model.h \
    ModelRegistration.h \
    PnPProblem.h \
    RobustMatcher.h \
    Utils.h
