TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    Utils.cpp \
    RobustMatcher.cpp \
    PnPProblem.cpp \
    ModelRegistration.cpp \
    Model.cpp \
    Mesh.cpp \
    CsvWriter.cpp \
    CsvReader.cpp

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv

HEADERS += \
    Utils.h \
    RobustMatcher.h \
    PnPProblem.h \
    ModelRegistration.h \
    Model.h \
    Mesh.h \
    CsvWriter.h \
    CsvReader.h
