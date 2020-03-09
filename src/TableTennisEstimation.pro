QT += core network websockets widgets
QT -= gui
QMAKE_CXXFLAGS += -fopenmp

CONFIG += c++11

TARGET = TableTennisEstimation
CONFIG += console
CONFIG -= app_bundle
CONFIG += no_keywords

INCLUDEPATH += kcf-tracker

TEMPLATE = app

SOURCES += detect.cpp \
            kcf-tracker/fhog.cpp \
            kcf-tracker/kcftracker.cpp \
            PosEstimation.cpp \
            geometryCalculation.cpp

HEADERS += kcf-tracker/ffttools.hpp \
            kcf-tracker/fhog.hpp \
            kcf-tracker/kcftracker.hpp \
            kcf-tracker/labdata.hpp \
            kcf-tracker/recttools.hpp \
            PosEstimation.h \
            kcf-tracker/trackpool.hpp \
            geometryCalculation.hpp

CONFIG += link_pkgconfig
PKGCONFIG += opencv

