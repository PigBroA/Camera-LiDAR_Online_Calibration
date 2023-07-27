#pragma once

#include <iostream>
#include <fstream>

struct IntrinsicData {
    int binTypes; //1: Intrinsic, 2: Extrinsic, 3: LUT
    int fov;
    int lensModelTypes; //1: Perspective, 2: Equidistance, 3: Catadioptric
    int doubleValueCounts;
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double k4;
    double p1;
    double p2;
    double xi;
};

std::ostream& operator<<(std::ostream& os, const IntrinsicData& intrinsinData);
void loadIntrinsic(const std::string binName, IntrinsicData& intrinsicData, int width=1920, int height=1080, bool verbose=true);
void saveIntrinsic(const std::string binName, const IntrinsicData intrinsicData, int width, int height);

struct ExtrinsicData {
    int binTypes; //1: Intrinsic, 2: Extrinsic, 3: LUT
    int rotationTypes; //1: Roll/Pitch/Yaw, 2: Rotation Matrix
    double roll;
    double pitch;
    double yaw;
    double r11;
    double r12;
    double r13;
    double r21;
    double r22;
    double r23;
    double r31;
    double r32;
    double r33;
    double t1;
    double t2;
    double t3;
};

std::ostream& operator<<(std::ostream& os, const ExtrinsicData& extrinsicData);
void loadExtrinsic(const std::string binName, ExtrinsicData& extrinsicData, bool verbose=true);
void saveExtrinsic(const std::string binName, const ExtrinsicData extrinsicData);

int distinguishBinType(const std::string binName);
