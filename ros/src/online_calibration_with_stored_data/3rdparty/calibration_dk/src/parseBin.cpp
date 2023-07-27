#include "parseBin.h"

std::ostream& operator<<(std::ostream& os, const IntrinsicData& intrinsicData) {
    os << "<Intrinsic Data>" << std::endl;
    os << "Bin Type: " << intrinsicData.binTypes << " | 1: Intrinsic, 2: Extrinsic, 3: LUT" << std::endl;
    os << "FoV: " << intrinsicData.fov << std::endl;
    os << "Lens Model Type: " << intrinsicData.lensModelTypes << " | 1: Perspective, 2: Equidistance, 3: Catadioptric" << std::endl;
    os << "Double Value Counts: " << intrinsicData.doubleValueCounts << std::endl;
    os << "fx: " << intrinsicData.fx << std::endl;
    os << "fy: " << intrinsicData.fy << std::endl;
    os << "cx: " << intrinsicData.cx << std::endl;
    os << "cy: " << intrinsicData.cy << std::endl;
    os << "k1: " << intrinsicData.k1 << std::endl;
    os << "k2: " << intrinsicData.k2 << std::endl;
    if(intrinsicData.lensModelTypes == 1) { //Perspective
        os << "p1: " << intrinsicData.p1 << std::endl;
        os << "p2: " << intrinsicData.p2 << std::endl;
        os << "k3: " << intrinsicData.k3 << std::endl;
    }
    else if(intrinsicData.lensModelTypes == 2) { //Equidistance
        os << "k3: " << intrinsicData.k3 << std::endl;
        os << "k4: " << intrinsicData.k4 << std::endl;
    }
    else if(intrinsicData.lensModelTypes == 3) { //Catadioptric
        os << "p1: " << intrinsicData.p1 << std::endl;
        os << "p2: " << intrinsicData.p2 << std::endl;
        os << "xi: " << intrinsicData.xi << std::endl;
    }
    os << "--------------------------------------------------" << std::endl;
    return os;
}

void loadIntrinsic(const std::string binName, IntrinsicData& intrinsicData, int width, int height, bool verbose) {
    std::ifstream inBin;
    inBin.open(binName);
    inBin.read((char*)&intrinsicData.binTypes, sizeof(int));
    inBin.read((char*)&intrinsicData.fov, sizeof(int));
    inBin.read((char*)&intrinsicData.lensModelTypes, sizeof(int));
    inBin.read((char*)&intrinsicData.doubleValueCounts, sizeof(int));
    inBin.read((char*)&intrinsicData.fx, sizeof(double));
    intrinsicData.fx *= (double)width;
    inBin.read((char*)&intrinsicData.fy, sizeof(double));
    intrinsicData.fy *= (double)height;
    inBin.read((char*)&intrinsicData.cx, sizeof(double));
    intrinsicData.cx *= (double)width;
    inBin.read((char*)&intrinsicData.cy, sizeof(double));
    intrinsicData.cy *= (double)height;
    inBin.read((char*)&intrinsicData.k1, sizeof(double));
    inBin.read((char*)&intrinsicData.k2, sizeof(double));
    if(intrinsicData.lensModelTypes == 1) { //Perspective
        inBin.read((char*)&intrinsicData.p1, sizeof(double));
        inBin.read((char*)&intrinsicData.p2, sizeof(double));
        inBin.read((char*)&intrinsicData.k3, sizeof(double));
    }
    else if(intrinsicData.lensModelTypes == 2) { //Equidistance
        inBin.read((char*)&intrinsicData.k3, sizeof(double));
        inBin.read((char*)&intrinsicData.k4, sizeof(double));
    }
    else if(intrinsicData.lensModelTypes == 3) { //Catadioptric
        inBin.read((char*)&intrinsicData.p1, sizeof(double));
        inBin.read((char*)&intrinsicData.p2, sizeof(double));
        inBin.read((char*)&intrinsicData.xi, sizeof(double));
    }
    else {
        std::cerr << "Check lensModelTypes, There is wrong value." << std::endl;
        inBin.close();
        exit(0);
    }
    inBin.close();

    if(verbose) {
        std::cout << "<Intrinsic Data>" << std::endl;
        std::cout << "Bin Type: " << intrinsicData.binTypes << " | 1: Intrinsic, 2: Extrinsic, 3: LUT" << std::endl;
        std::cout << "FoV: " << intrinsicData.fov << std::endl;
        std::cout << "Lens Model Type: " << intrinsicData.lensModelTypes << " | 1: Perspective, 2: Equidistance, 3: Catadioptric" << std::endl;
        std::cout << "Double Value Counts: " << intrinsicData.doubleValueCounts << std::endl;
        std::cout << "fx: " << intrinsicData.fx << " with width(" << width << ")" << std::endl;
        std::cout << "fy: " << intrinsicData.fy << " with height(" << height << ")" << std::endl;
        std::cout << "cx: " << intrinsicData.cx << " with width(" << width << ")" << std::endl;
        std::cout << "cy: " << intrinsicData.cy << " with height(" << height << ")" << std::endl;
        std::cout << "k1: " << intrinsicData.k1 << std::endl;
        std::cout << "k2: " << intrinsicData.k2 << std::endl;
        if(intrinsicData.lensModelTypes == 1) { //Perspective
            std::cout << "p1: " << intrinsicData.p1 << std::endl;
            std::cout << "p2: " << intrinsicData.p2 << std::endl;
            std::cout << "k3: " << intrinsicData.k3 << std::endl;
        }
        else if(intrinsicData.lensModelTypes == 2) { //Equidistance
            std::cout << "k3: " << intrinsicData.k3 << std::endl;
            std::cout << "k4: " << intrinsicData.k4 << std::endl;
        }
        else if(intrinsicData.lensModelTypes == 3) { //Catadioptric
            std::cout << "p1: " << intrinsicData.p1 << std::endl;
            std::cout << "p2: " << intrinsicData.p2 << std::endl;
            std::cout << "xi: " << intrinsicData.xi << std::endl;
        }
        std::cout << "--------------------------------------------------" << std::endl << std::endl;
    }
}

void saveIntrinsic(const std::string binName, const IntrinsicData intrinsicData, int width, int height) {
    if(binName.size() <= 0) {
        std::cerr << "Check parentPath, the size is zero" << std::endl;
        exit(0);
    }
    std::ofstream outBin(binName, std::ios::out | std::ios::binary);
    int binTypes = intrinsicData.binTypes;
    int fov = intrinsicData.fov;
    int lensModelTypes = intrinsicData.lensModelTypes;
    int doubleValueCounts = intrinsicData.doubleValueCounts;
    double fxPerPixel = intrinsicData.fx/(double)width;
    double fyPerPixel = intrinsicData.fy/(double)height;
    double cxPerPixel = intrinsicData.cx/(double)width;
    double cyPerPixel = intrinsicData.cy/(double)height;
    double k1 = intrinsicData.k1;
    double k2 = intrinsicData.k2;
    outBin.write((char*)&binTypes, sizeof(int));
    outBin.write((char*)&fov, sizeof(int));
    outBin.write((char*)&lensModelTypes, sizeof(int));
    outBin.write((char*)&doubleValueCounts, sizeof(int));
    outBin.write((char*)&fxPerPixel, sizeof(double));
    outBin.write((char*)&fyPerPixel, sizeof(double));
    outBin.write((char*)&cxPerPixel, sizeof(double));
    outBin.write((char*)&cyPerPixel, sizeof(double));
    outBin.write((char*)&k1, sizeof(double));
    outBin.write((char*)&k2, sizeof(double));
    if(intrinsicData.lensModelTypes == 1) { //Perspective
        double p1 = intrinsicData.p1;
        double p2 = intrinsicData.p2;
        double k3 = intrinsicData.k3;
        outBin.write((char*)&p1, sizeof(double));
        outBin.write((char*)&p2, sizeof(double));
        outBin.write((char*)&k3, sizeof(double));
    }
    else if(intrinsicData.lensModelTypes == 2) { //Equidistance
        double k3 = intrinsicData.k3;
        double k4 = intrinsicData.k4;
        outBin.write((char*)&k3, sizeof(double));
        outBin.write((char*)&k4, sizeof(double));
    }
    else if(intrinsicData.lensModelTypes == 3) { //Catadioptric
        double p1 = intrinsicData.p1;
        double p2 = intrinsicData.p2;
        double xi = intrinsicData.xi;
        outBin.write((char*)&p1, sizeof(double));
        outBin.write((char*)&p2, sizeof(double));
        outBin.write((char*)&xi, sizeof(double));
    }
    else {
        std::cerr << "Check lensModelTypes, There is wrong value." << std::endl;
        outBin.close();
        exit(0);
    }
    outBin.close();
}

std::ostream& operator<<(std::ostream& os, const ExtrinsicData& extrinsicData) {
    os << "<Extrinsic Data>" << std::endl;
    os << "Bin Type: " << extrinsicData.binTypes << " | 1: Intrinsic, 2: Extrinsic, 3: LUT" << std::endl;
    os << "Rotation Type: " << extrinsicData.rotationTypes << " | 1: Roll/Pitch/Yaw, 2: Rotation Matrix" << std::endl;
    if(extrinsicData.rotationTypes == 1) { //Roll/Pitch/Yaw
        os << "roll: " << extrinsicData.roll << std::endl;
        os << "pitch: " << extrinsicData.pitch << std::endl;
        os << "yaw: " << extrinsicData.yaw << std::endl;
    }
    else if(extrinsicData.rotationTypes == 2) { //Rotation Matrix
        os << "r11: " << extrinsicData.r11 << std::endl;
        os << "r12: " << extrinsicData.r12 << std::endl;
        os << "r13: " << extrinsicData.r13 << std::endl;
        os << "r21: " << extrinsicData.r21 << std::endl;
        os << "r22: " << extrinsicData.r22 << std::endl;
        os << "r23: " << extrinsicData.r23 << std::endl;
        os << "r31: " << extrinsicData.r31 << std::endl;
        os << "r32: " << extrinsicData.r32 << std::endl;
        os << "r33: " << extrinsicData.r33 << std::endl;
    }
    os << "t1: " << extrinsicData.t1 << std::endl;
    os << "t2: " << extrinsicData.t2 << std::endl;
    os << "t3: " << extrinsicData.t3 << std::endl;
    os << "--------------------------------------------------" << std::endl;
    return os;
}

void loadExtrinsic(const std::string binName, ExtrinsicData& extrinsicData, bool verbose) {
    std::ifstream inBin;
    inBin.open(binName);
    inBin.read((char*)&extrinsicData.binTypes, sizeof(int));
    inBin.read((char*)&extrinsicData.rotationTypes, sizeof(int));
    if(extrinsicData.rotationTypes == 1) { //Roll/Pitch/Yaw
        inBin.read((char*)&extrinsicData.roll, sizeof(double));
        inBin.read((char*)&extrinsicData.pitch, sizeof(double));
        inBin.read((char*)&extrinsicData.yaw, sizeof(double));
    }
    else if(extrinsicData.rotationTypes == 2) { //Rotation Matrix
        inBin.read((char*)&extrinsicData.r11, sizeof(double));
        inBin.read((char*)&extrinsicData.r12, sizeof(double));
        inBin.read((char*)&extrinsicData.r13, sizeof(double));
        inBin.read((char*)&extrinsicData.r21, sizeof(double));
        inBin.read((char*)&extrinsicData.r22, sizeof(double));
        inBin.read((char*)&extrinsicData.r23, sizeof(double));
        inBin.read((char*)&extrinsicData.r31, sizeof(double));
        inBin.read((char*)&extrinsicData.r32, sizeof(double));
        inBin.read((char*)&extrinsicData.r33, sizeof(double));
    }
    else {
        std::cerr << "Check rotationTypes, There is wrong value." << std::endl;
        inBin.close();
        exit(0);
    }
    inBin.read((char*)&extrinsicData.t1, sizeof(double));
    inBin.read((char*)&extrinsicData.t2, sizeof(double));
    inBin.read((char*)&extrinsicData.t3, sizeof(double));
    inBin.close();

    if(verbose) {
        std::cout << "<Extrinsic Data>" << std::endl;
        std::cout << "Bin Type: " << extrinsicData.binTypes << " | 1: Intrinsic, 2: Extrinsic, 3: LUT" << std::endl;
        std::cout << "Rotation Type: " << extrinsicData.rotationTypes << " | 1: Roll/Pitch/Yaw, 2: Rotation Matrix" << std::endl;
        if(extrinsicData.rotationTypes == 1) { //Roll/Pitch/Yaw
            std::cout << "roll: " << extrinsicData.roll << std::endl;
            std::cout << "pitch: " << extrinsicData.pitch << std::endl;
            std::cout << "yaw: " << extrinsicData.yaw << std::endl;
        }
        else if(extrinsicData.rotationTypes == 2) { //Rotation Matrix
            std::cout << "r11: " << extrinsicData.r11 << std::endl;
            std::cout << "r12: " << extrinsicData.r12 << std::endl;
            std::cout << "r13: " << extrinsicData.r13 << std::endl;
            std::cout << "r21: " << extrinsicData.r21 << std::endl;
            std::cout << "r22: " << extrinsicData.r22 << std::endl;
            std::cout << "r23: " << extrinsicData.r23 << std::endl;
            std::cout << "r31: " << extrinsicData.r31 << std::endl;
            std::cout << "r32: " << extrinsicData.r32 << std::endl;
            std::cout << "r33: " << extrinsicData.r33 << std::endl;
        }
        std::cout << "t1: " << extrinsicData.t1 << std::endl;
        std::cout << "t2: " << extrinsicData.t2 << std::endl;
        std::cout << "t3: " << extrinsicData.t3 << std::endl;
        std::cout << "--------------------------------------------------" << std::endl << std::endl;
    }
}

void saveExtrinsic(const std::string binName, const ExtrinsicData extrinsicData) {
    if(binName.size() <= 0) {
        std::cerr << "Check parentPath, the size is zero" << std::endl;
        exit(0);
    }
    std::ofstream outBin(binName, std::ios::out | std::ios::binary);
    int binTypes = extrinsicData.binTypes;
    int rotationTypes = extrinsicData.rotationTypes;
    outBin.write((char*)&binTypes, sizeof(int));
    outBin.write((char*)&rotationTypes, sizeof(int));
    if(extrinsicData.rotationTypes == 1) { //Roll/Pitch/Yaw
        double roll = extrinsicData.roll;
        double pitch = extrinsicData.pitch;
        double yaw = extrinsicData.yaw;
        outBin.write((char*)&roll, sizeof(double));
        outBin.write((char*)&pitch, sizeof(double));
        outBin.write((char*)&yaw, sizeof(double));
    }
    else if(extrinsicData.rotationTypes == 2) { //Rotation Matrix
        double r11 = extrinsicData.r11;
        double r12 = extrinsicData.r12;
        double r13 = extrinsicData.r13;
        double r21 = extrinsicData.r21;
        double r22 = extrinsicData.r22;
        double r23 = extrinsicData.r23;
        double r31 = extrinsicData.r31;
        double r32 = extrinsicData.r32;
        double r33 = extrinsicData.r33;
        outBin.write((char*)&r11, sizeof(double));
        outBin.write((char*)&r12, sizeof(double));
        outBin.write((char*)&r13, sizeof(double));
        outBin.write((char*)&r21, sizeof(double));
        outBin.write((char*)&r22, sizeof(double));
        outBin.write((char*)&r23, sizeof(double));
        outBin.write((char*)&r31, sizeof(double));
        outBin.write((char*)&r32, sizeof(double));
        outBin.write((char*)&r33, sizeof(double));
    }
    else {
        std::cerr << "Check rotationTypes, There is wrong value." << std::endl;
        outBin.close();
        exit(0);
    }
    double t1 = extrinsicData.t1;
    double t2 = extrinsicData.t2;
    double t3 = extrinsicData.t3;
    outBin.write((char*)&t1, sizeof(double));
    outBin.write((char*)&t2, sizeof(double));
    outBin.write((char*)&t3, sizeof(double));
    outBin.close();
}

int distinguishBinType(const std::string binName) {
    std::ifstream inBin;
    inBin.open(binName);
    int binType;
    inBin.read((char*)&binType, sizeof(int));
    inBin.close();

    return binType;
}
