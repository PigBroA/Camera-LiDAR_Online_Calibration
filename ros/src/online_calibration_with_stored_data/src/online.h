#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <deque>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <eigen3/Eigen/Eigen>
#include <casadi/casadi.hpp>

#include <3rdparty/fullyHardNet/src/fullyHardNet.h>
#include <3rdparty/bts/src/bts.h>
#include <3rdparty/superglue/src/super_point.h>
#include <3rdparty/superglue/src/super_glue.h>
#include <3rdparty/superglue/src/utils.h>
#include <3rdparty/glob/src/glob.h>
#include <3rdparty/calibration_dk/src/parseBin.h>
#include <3rdparty/calibration_dk/src/util.h>
#include <3rdparty/pclomp/src/ndt_omp.h>
#include <3rdparty/patchworkpp/src/patchworkpp.h>


class OnlineCalibration {
    ros::NodeHandle nh;

    ros::Publisher cameraFloorCloudPub;
    ros::Publisher cameraFloorpolygonPub;
    ros::Publisher cameraPosePub;
    ros::Publisher lidarFloorCloudPub;
    ros::Publisher lidarFloorpolygonPub;
    ros::Publisher lidarPosePub;
    ros::Publisher flattenLidarPub;
    ros::Publisher gnssPosePub;

    std::string parentPath;
    std::string dbPath;
    std::string imgNameFormat;
    std::string pcdNameFormat;
    std::string poseNameFormat;
    std::vector<std::filesystem::path> imgFilesystemPathLet;
    std::vector<std::filesystem::path> pcdFilesystemPathLet;
    std::vector<std::filesystem::path> poseFilesystemPathLet;

    std::string sceneName;

    std::string offlineExtrinsicName;
    ExtrinsicData offlineExtrinsic;
    Eigen::Matrix4f offlineLidarToCameraTransformationMatrix;
    Eigen::Vector3f decodedOfflineLidarToCameraOrientation;
    float offlineLidarToCameraRoll;
    float offlineLidarToCameraPitch;
    float offlineLidarToCameraYaw;

    int cudaId = 0;
    torch::Device device = torch::Device(cv::format("cuda:%d", cudaId));
    std::string hardnetWeightName;
    std::string btsWeightName;
    HarDNet hardnet;
    BTS bts;
    torch::NoGradGuard noGrad;

    std::string superglueModelPath;

    int width;
    int height;

    int lensModelIdx;
    std::string intrinsicName;
    IntrinsicData intrinsicData;
    cv::Mat cameraMat;
    cv::Mat distCoeffs;
    cv::Mat xi;
    double focalLength;
    cv::Point2d principalPoint;

    int du;
    int dv;
    std::vector<cv::Point2d> distortedPtLet;
    std::vector<cv::Point2d> normalizedUndistortedPtLet;
    double uNUMin;
    double uNUMax;
    double vNUMin;
    double vNUMax;
    double realFovU;
    double realFovV;
    std::vector<cv::Point3d> fovVertexPtLet;

    cv::Mat lidarToBevMat;
    cv::Mat offlineBev;
    cv::Mat offlineRtMat;
    cv::Mat offlineHomography;
    cv::Mat onlineBev;
    cv::Mat onlineRtMat;
    cv::Mat onlineHomography;

    patchwork::Params patchworkParameters;
    std::shared_ptr<patchwork::PatchWorkpp> pPatchworkpp;

    Eigen::Matrix3f currImuRotationMatrix;
    Eigen::Matrix3f prevImuRotationMatrix;
    Eigen::Matrix3f initImuRotationMatrix;
    Eigen::Matrix<float, 3, 1> currGnssTranslationMatrix;
    Eigen::Matrix<float, 3, 1> prevGnssTranslationMatrix;
    Eigen::Matrix<float, 3, 1> initGnssTranslationMatrix;
    Eigen::Matrix3f gnssImuOdomRotationMatrix0; //origin<->local frame 
    Eigen::Matrix<float, 3, 1> gnssImuOdomTranslationMatrix0; //origin<->local frame

    pcl::PointCloud<pcl::PointXYZ>::Ptr pCurrCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pPrevCloud;
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr lidarOdomNdt;

    cv::Mat prevImg;
    cv::Mat currImg;
    Configs configs;
    std::shared_ptr<SuperPoint> superpoint;
    std::shared_ptr<SuperGlue> superglue;

    std::deque<Eigen::Matrix4f> cameraOdomTransformationMatrixLet;
    std::deque<Eigen::Matrix4f> flattenCameraOdomTransformationMatrixLet;
    std::deque<Eigen::Matrix4f> lidarOdomTransformationMatrixLet;
    std::deque<Eigen::Matrix4f> flattenLidarOdomTransformationMatrixLet;

    int dequeMaxSize;

    double bestLoss;
    double bestRLoss;
    double bestTLoss;
    Eigen::Matrix4f bestOdomLidarToCameraTransformationMatrix;

    int cnt;
    int sequenceGap;
public:
    OnlineCalibration(ros::NodeHandle* pNH);
    void loadPose(std::string poseName, geometry_msgs::PoseStamped& pose);
    void estimateCameraFloor(cv::Mat img, cv::Mat cameraMatrix, cv::Mat distCoeffs,
                            HarDNet hardnet, BTS bts,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pCameraFloorCloud,
                            std::vector<double>& cameraFloorCoeffs);
    void estimateLidarFloor(pcl::PointCloud<pcl::PointXYZI>::Ptr pCloud, std::shared_ptr<patchwork::PatchWorkpp> pPatchworkpp,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr& pLidarFloorCloud,
                            std::vector<double>& lidarFloorCoeffs);
    void makePlanePolygon(std::vector<double> planeCoeffs, int drawOption,
                          std::string frameId, ros::Time stamp,
                          geometry_msgs::PolygonStamped &polygon);
    void calcRotationVector3d(const cv::Point3d fromV, const cv::Point3d toV, cv::Mat& rotationMatrix);
    casadi::SX axisAngleToMatrix(casadi::SX rotvec);
    Eigen::Matrix3f axisAngleToMatrix(Eigen::Vector3f rotationVector);
    casadi::SX asTransitionMatrix(Eigen::Matrix3f r, Eigen::Matrix<float, 3, 1> t, bool inv=false);
    casadi::SX onlineCalibrateCost(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                                   casadi::SX rotvec, casadi::SX t);
    Eigen::Matrix4f onlineCalibrate(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                                    Eigen::Matrix<float, 3, 1> offlineTranslationMatrix, double& loss);
    void calcSolverLoss(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                        Eigen::Matrix4f transformationMatrix, double &loss, double &rLoss, double &tLoss);
    void calcComparisonLoss(Eigen::Matrix4f transformationMatrix, Eigen::Matrix4f referenceTransformationMatrix, double& loss, double& rLoss, double& tLoss);
    cv::Scalar scalarHSV2BGR(uchar h, uchar s, uchar v);
};