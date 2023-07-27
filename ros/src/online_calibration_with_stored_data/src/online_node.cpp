#include <online.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "online_calibration_with_stored_data");
    ros::NodeHandle nh("~");

    OnlineCalibration onlineCalibration(&nh);

    return 0;
}