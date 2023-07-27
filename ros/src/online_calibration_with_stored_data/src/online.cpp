#include <online.h>

OnlineCalibration::OnlineCalibration(ros::NodeHandle* pNH) {
    nh = *pNH;
    
    cameraFloorCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/camera/floor_cloud", 1);
    cameraFloorpolygonPub = nh.advertise<geometry_msgs::PolygonStamped>("/camera/floor_polygon", 1);
    cameraPosePub = nh.advertise<geometry_msgs::PoseStamped>("/camera_pose", 1);
    lidarFloorCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/lidar/floor_cloud", 1);
    lidarFloorpolygonPub = nh.advertise<geometry_msgs::PolygonStamped>("/lidar/floor_polygon", 1);
    lidarPosePub = nh.advertise<geometry_msgs::PoseStamped>("/lidar_pose", 1);
    flattenLidarPub = nh.advertise<sensor_msgs::PointCloud2>("/lidar/flatten", 1);
    gnssPosePub = nh.advertise<geometry_msgs::PoseStamped>("/gnss_pose", 1);

    parentPath = CMAKE_PATH;
    nh.param<std::string>("dbPath", dbPath, "/sample_db/20230614_10");
    dbPath = parentPath + dbPath;

    imgNameFormat = dbPath + "/png/60front/*.png";
    imgFilesystemPathLet = glob::glob(imgNameFormat);
    std::sort(imgFilesystemPathLet.begin(), imgFilesystemPathLet.end());
    pcdNameFormat = dbPath + "/pcd/*.pcd";
    pcdFilesystemPathLet = glob::glob(pcdNameFormat);
    std::sort(pcdFilesystemPathLet.begin(), pcdFilesystemPathLet.end());
    poseNameFormat = dbPath + "/pose/*.txt";
    poseFilesystemPathLet = glob::glob(poseNameFormat);
    std::sort(poseFilesystemPathLet.begin(), poseFilesystemPathLet.end());

    sceneName = poseFilesystemPathLet[0].parent_path().parent_path().filename();
    nh.param<std::string>("offlineExtrinsicName", offlineExtrinsicName, "/pre_cali/extrinsic_parameters.bin");
    offlineExtrinsicName = parentPath + offlineExtrinsicName;
    
    loadExtrinsic(offlineExtrinsicName, offlineExtrinsic, false);
    offlineLidarToCameraTransformationMatrix << offlineExtrinsic.r11, offlineExtrinsic.r12, offlineExtrinsic.r13, offlineExtrinsic.t1,
        offlineExtrinsic.r21, offlineExtrinsic.r22, offlineExtrinsic.r23, offlineExtrinsic.t2,
        offlineExtrinsic.r31, offlineExtrinsic.r32, offlineExtrinsic.r33, offlineExtrinsic.t3,
        0.0, 0.0, 0.0, 1.0;
    decodedOfflineLidarToCameraOrientation = offlineLidarToCameraTransformationMatrix.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
    offlineLidarToCameraRoll = decodedOfflineLidarToCameraOrientation[2];
    offlineLidarToCameraPitch = decodedOfflineLidarToCameraOrientation[1];
    offlineLidarToCameraYaw = decodedOfflineLidarToCameraOrientation[0];
    std::cout << "Offline LiDAR to Camera Roll: " << offlineLidarToCameraRoll*(180.0/M_PI) << "°" << std::endl;
    std::cout << "Offline LiDAR to Camera Pitch: " << offlineLidarToCameraPitch*(180.0/M_PI) << "°" << std::endl;
    std::cout << "Offline LiDAR to Camera Yaw: " << offlineLidarToCameraYaw*(180.0/M_PI) << "°" << std::endl;

    currImuRotationMatrix = Eigen::Matrix3f::Identity();
    prevImuRotationMatrix = Eigen::Matrix3f::Identity();
    gnssImuOdomRotationMatrix0 = Eigen::Matrix3f::Identity();
    gnssImuOdomTranslationMatrix0 << 0.0,
        0.0,
        0.0;

    pCurrCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pPrevCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    lidarOdomNdt.reset(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
    lidarOdomNdt->setTransformationEpsilon(0.001);
    lidarOdomNdt->setResolution(4.0);
    lidarOdomNdt->setStepSize(40.0);
    lidarOdomNdt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    lidarOdomNdt->setNumThreads(32);

    hardnet = HarDNet(3);
    bts = BTS("Own_S", 120.0,
              std::vector<int64_t>({96, 96, 192, 384, 2208}), 512,
              48, std::vector<int64_t>({6, 12, 36, 24}),
              96, 4, 0.0, 1000, false,
              std::vector<std::string>({"relu0", "pool0", "transition1", "transition2", "norm5"}));
    nh.param<std::string>("hardnetWeightName", hardnetWeightName, "/model/hardnet/hardnet_outdoor.pt");
    hardnetWeightName = parentPath + hardnetWeightName;
    nh.param<std::string>("btsWeightName", btsWeightName, "/model/bts/bts_outdoor.pt");
    btsWeightName = parentPath + btsWeightName;
    hardnet->to(device);
    bts->to(device);
    torch::load(hardnet, hardnetWeightName);
    torch::load(bts, btsWeightName);
    hardnet->eval();
    bts->eval();

    nh.param<std::string>("superglueModelPath", superglueModelPath, "/model/superglue/");
    superglueModelPath = parentPath + superglueModelPath;
    configs = Configs(superglueModelPath + "config.yaml", superglueModelPath);
    superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        exit(1);
    }
    superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        exit(1);
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

    width = configs.superglue_config.image_width;
    height = configs.superglue_config.image_height;

    nh.param<int>("lensModelIdx", lensModelIdx, 1);
    nh.param<std::string>("intrinsicName", intrinsicName, "/pre_cali/intrinsic_paremeters.bin");
    intrinsicName = parentPath + intrinsicName;
    loadIntrinsic(intrinsicName, intrinsicData, width, height, true);
    cameraMat = (cv::Mat_<double>(3, 3) << intrinsicData.fx, 0.0, intrinsicData.cx,
                 0.0, intrinsicData.fy, intrinsicData.cy,
                 0.0, 0.0, 1.0);
    distCoeffs = (cv::Mat_<double>(4, 1) << intrinsicData.k1,
                  intrinsicData.k2,
                  intrinsicData.p1,
                  intrinsicData.p2);
    focalLength = (intrinsicData.fx + intrinsicData.fy) / 2.0;
    principalPoint = cv::Point2d(intrinsicData.cx, intrinsicData.cy);

    du = 5;
    dv = 5;
    for(int j = 0; j < height; j += dv) {
        for(int i = 0; i < width; i += du) {
            distortedPtLet.push_back(cv::Point2d(i, j));
        }
    }
    if(lensModelIdx == 3) {
        if(intrinsicData.lensModelTypes != 3) {
            std::cerr << "Check Lens Model Type!!!" << std::endl;
            exit(0);
        }
        xi = (cv::Mat_<double>(1, 1) << intrinsicData.xi);
        cv::omnidir::undistortPoints(distortedPtLet, normalizedUndistortedPtLet, cameraMat, distCoeffs, xi, cv::Mat::eye(3, 3, CV_64F));
    }
    else {
        if (intrinsicData.lensModelTypes != 1) {
            std::cerr << "Check Lens Model Type!!!" << std::endl;
            exit(0);
        }
        distCoeffs.push_back(intrinsicData.k3);
        cv::undistortPoints(distortedPtLet, normalizedUndistortedPtLet, cameraMat, distCoeffs);
    }
    uNUMin = std::numeric_limits<double>::max();
    uNUMax = -std::numeric_limits<double>::max();
    vNUMin = std::numeric_limits<double>::max();
    vNUMax = -std::numeric_limits<double>::max();
    for(int i = 0; i < normalizedUndistortedPtLet.size(); i++) {
        double uD = distortedPtLet[i].x;
        double vD = distortedPtLet[i].y;
        double uNU = normalizedUndistortedPtLet[i].x;
        double vNU = normalizedUndistortedPtLet[i].y;
        uNUMin = std::min(uNUMin, uNU);
        uNUMax = std::max(uNUMax, uNU);
        vNUMin = std::min(vNUMin, vNU);
        vNUMax = std::max(vNUMax, vNU);
    }
    realFovU = (std::atan2(std::abs(uNUMin), 1.0) + std::atan2(std::abs(uNUMax), 1.0))*180.0/CV_PI;
    realFovV = (std::atan2(std::abs(vNUMin), 1.0) + std::atan2(std::abs(vNUMax), 1.0))*180.0/CV_PI;
    
    fovVertexPtLet.push_back(cv::Point3d(-std::sin(realFovU/2.0), 0.0, std::cos(realFovU/2.0)));
    fovVertexPtLet.push_back(cv::Point3d(std::sin(realFovU/2.0), 0.0, std::cos(realFovU/2.0)));
    fovVertexPtLet.push_back(cv::Point3d(0.0, -std::sin(realFovV/2.0), std::cos(realFovV/2.0)));
    fovVertexPtLet.push_back(cv::Point3d(0.0, std::sin(realFovV/2.0), std::cos(realFovV/2.0)));

    nh.param<int>("dequeMaxSize", dequeMaxSize, 100);

    lidarToBevMat = (cv::Mat_<double>(3, 3) << 0.0, 50.0, 250.0,
                     50.0, 0.0, 500.0,
                     0.0, 0.0, 1.0);
    offlineBev = cv::Mat::zeros(500, 500, CV_8UC3);
    offlineRtMat = (cv::Mat_<double>(3, 3) << offlineExtrinsic.r11, offlineExtrinsic.r12, offlineExtrinsic.t1,
                    offlineExtrinsic.r21, offlineExtrinsic.r22, offlineExtrinsic.t2,
                    offlineExtrinsic.r31, offlineExtrinsic.r32, offlineExtrinsic.t3);
    offlineHomography = lidarToBevMat * offlineRtMat.inv() * cameraMat.inv();
    onlineBev = cv::Mat::zeros(500, 500, CV_8UC3);

    pPatchworkpp.reset(new patchwork::PatchWorkpp(patchworkParameters));

    bestLoss = std::numeric_limits<double>::max();
    bestRLoss = std::numeric_limits<double>::max();
    bestTLoss = std::numeric_limits<double>::max();
    bestOdomLidarToCameraTransformationMatrix = Eigen::Matrix4f::Identity();

    cnt = 0;
    sequenceGap = 1;

    while(ros::ok()) {
        std::chrono::system_clock::time_point startTick = std::chrono::system_clock::now();

        if(cnt >= imgFilesystemPathLet.size()) {
            cnt = imgFilesystemPathLet.size() - sequenceGap;
        }
        cv::Mat img = cv::imread(imgFilesystemPathLet[cnt]);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(pcdFilesystemPathLet[cnt], *pCloud);
        pCloud->header.frame_id = "show";
        geometry_msgs::PoseStamped pose;
        loadPose(poseFilesystemPathLet[cnt], pose);
        pose.header.frame_id = "show";

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCameraFloorCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pCameraFloorCloud->header.frame_id = "show";
        std::vector<double> cameraFloorCoeffs;
        estimateCameraFloor(img, cameraMat, distCoeffs,
                            hardnet, bts,
                            pCameraFloorCloud,
                            cameraFloorCoeffs);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pLidarFloorCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pLidarFloorCloud->header.frame_id = "show";
        std::vector<double> lidarFloorCoeffs;
        estimateLidarFloor(pCloud, pPatchworkpp,
                        pLidarFloorCloud,
                        lidarFloorCoeffs);

        currGnssTranslationMatrix(0, 0) = pose.pose.position.x;
        currGnssTranslationMatrix(1, 0) = pose.pose.position.y;
        currGnssTranslationMatrix(2, 0) = pose.pose.position.z;
        Eigen::Quaternionf currImuQuaternion;
        currImuQuaternion.w() = pose.pose.orientation.w;
        currImuQuaternion.x() = pose.pose.orientation.x;
        currImuQuaternion.y() = pose.pose.orientation.y;
        currImuQuaternion.z() = pose.pose.orientation.z;
        currImuRotationMatrix = currImuQuaternion;
        if(cnt == 0) {
            prevImuRotationMatrix = currImuRotationMatrix;
            prevGnssTranslationMatrix = currGnssTranslationMatrix;
            initImuRotationMatrix = currImuRotationMatrix;
            initGnssTranslationMatrix = currGnssTranslationMatrix;
        }
        Eigen::Matrix3f gnssImuOdomRotationMatrix = prevImuRotationMatrix.inverse()*currImuRotationMatrix;
        Eigen::Matrix<float, 3, 1> gnssImuOdomTranslationMatrix = currImuRotationMatrix.inverse()*(currGnssTranslationMatrix - prevGnssTranslationMatrix);
        Eigen::Matrix4f initTransformationMatrix = Eigen::Matrix4f::Identity();
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                initTransformationMatrix(row, col) = gnssImuOdomRotationMatrix(row, col);
            }
            initTransformationMatrix(row, 3) = gnssImuOdomTranslationMatrix(row, 0);
        }
        gnssImuOdomRotationMatrix0 = gnssImuOdomRotationMatrix*gnssImuOdomRotationMatrix0;
        gnssImuOdomTranslationMatrix0 = gnssImuOdomTranslationMatrix0 + (gnssImuOdomRotationMatrix0*gnssImuOdomTranslationMatrix);
        Eigen::Matrix<float, 3, 1> gnssDifferential = gnssImuOdomRotationMatrix0*gnssImuOdomTranslationMatrix;
        double scaleFromGnssImu = std::sqrt(std::pow(gnssDifferential(0, 0), 2) + std::pow(gnssDifferential(1, 0), 2) + std::pow(gnssDifferential(2, 0), 2));
        prevImuRotationMatrix = currImuRotationMatrix;
        prevGnssTranslationMatrix = currGnssTranslationMatrix;

        boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZ>> voxelgrid(new pcl::VoxelGrid<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr pVoxeledCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*pCloud, *pVoxeledCloud);
        voxelgrid->setLeafSize(1.0, 1.0, 1.0);
        voxelgrid->setInputCloud(pVoxeledCloud);
        voxelgrid->filter(*pVoxeledCloud);
        if(cnt == 0) {
            pcl::copyPointCloud(*pVoxeledCloud, *pPrevCloud);
        }
        pcl::copyPointCloud(*pVoxeledCloud, *pCurrCloud);
        lidarOdomNdt->setInputSource(pCurrCloud);
        lidarOdomNdt->setInputTarget(pPrevCloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pLidarOdomAligned(new pcl::PointCloud<pcl::PointXYZ>());
        // lidarOdomNdt->align(*pLidarOdomAligned, initTransformationMatrix);
        lidarOdomNdt->align(*pLidarOdomAligned, Eigen::Matrix4f::Identity());
        Eigen::Matrix4f lidarOdomTransformationMatrix = lidarOdomNdt->getFinalTransformation();
        *pPrevCloud = *pCurrCloud;

        img.copyTo(currImg);
        if(cnt == 0) {
            currImg.copyTo(prevImg);
        }
        cv::Mat prevGrayImg;
        cv::Mat currGrayImg;
        cv::cvtColor(prevImg, prevGrayImg, cv::COLOR_BGR2GRAY);
        cv::cvtColor(currImg, currGrayImg, cv::COLOR_BGR2GRAY);
        Eigen::Matrix<double, 259, Eigen::Dynamic> prevFeaturesMat;
        Eigen::Matrix<double, 259, Eigen::Dynamic> currFeaturesMat;
        superpoint->infer(prevGrayImg, prevFeaturesMat);
        superpoint->infer(currGrayImg, currFeaturesMat);
        std::vector<cv::KeyPoint> prevKeypoints;
        std::vector<cv::KeyPoint> currKeypoints;
        for(int i = 0; i < prevFeaturesMat.cols(); i++) {
            double score = prevFeaturesMat(0, i);
            double x = prevFeaturesMat(1, i);
            double y = prevFeaturesMat(2, i);
            prevKeypoints.emplace_back(std::move(x), std::move(y), 8, -1, std::move(score));
        }
        for(int i = 0; i < currFeaturesMat.cols(); i++) {
            double score = currFeaturesMat(0, i);
            double x = currFeaturesMat(1, i);
            double y = currFeaturesMat(2, i);
            currKeypoints.emplace_back(std::move(x), std::move(y), 8, -1, std::move(score));
        }
        std::vector<cv::DMatch> matchLet;
        superglue->matching_points(prevFeaturesMat, currFeaturesMat, matchLet);
        std::vector<cv::Point2f> prevFeatures;
        std::vector<cv::Point2f> currFeatures;
        // cv::Mat currVis;
        // currImg.copyTo(currVis);
        for(int i = 0; i < matchLet.size(); i++) {
            prevFeatures.push_back(prevKeypoints[matchLet[i].queryIdx].pt);
            currFeatures.push_back(currKeypoints[matchLet[i].trainIdx].pt);
            // cv::circle(currVis, prevFeatures[i], 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            // cv::line(currVis, prevFeatures[i], currFeatures[i], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            // cv::circle(currVis, currFeatures[i], 1, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }
        // cv::imshow("currVis", currVis);
        std::vector<cv::Point2f> prevUndistortedFeatures;
        std::vector<cv::Point2f> currUndistortedFeatures;
        cv::undistortPoints(prevFeatures, prevUndistortedFeatures, cameraMat, distCoeffs);
        cv::undistortPoints(currFeatures, currUndistortedFeatures, cameraMat, distCoeffs);
        cv::perspectiveTransform(prevUndistortedFeatures, prevUndistortedFeatures, cameraMat);
        cv::perspectiveTransform(currUndistortedFeatures, currUndistortedFeatures, cameraMat);
        cv::Mat cameraOdomEssentialMat;
        cv::Mat cameraOdomRotationMat;
        cv::Mat cameraOdomTranslationMat;
        cv::Mat outlierMask;
        cameraOdomEssentialMat = cv::findEssentialMat(currUndistortedFeatures, prevUndistortedFeatures, focalLength, principalPoint, cv::RANSAC, 0.999, 1.0, outlierMask);
        cv::recoverPose(cameraOdomEssentialMat, currUndistortedFeatures, prevUndistortedFeatures, cameraOdomRotationMat, cameraOdomTranslationMat, focalLength, principalPoint, outlierMask);
        prevFeatures.clear();
        currFeatures.clear();
        prevUndistortedFeatures.clear();
        currUndistortedFeatures.clear();
        Eigen::Matrix4f cameraOdomTransformationMatrix = Eigen::Matrix4f::Identity();
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                cameraOdomTransformationMatrix(row, col) = cameraOdomRotationMat.at<double>(row, col);
            }
            cameraOdomTransformationMatrix(row, 3) = scaleFromGnssImu*cameraOdomTranslationMat.at<double>(row, 0);
        }
        currImg.copyTo(prevImg);

        cv::Mat lidarFlattenRotationMat;
        calcRotationVector3d(cv::Point3d(lidarFloorCoeffs[0], lidarFloorCoeffs[1], lidarFloorCoeffs[2]),
                             cv::Point3d(0, 0, 1),
                             lidarFlattenRotationMat);
        Eigen::Matrix4f lidarFlattenTransformationMatrix = Eigen::Matrix4f::Identity();
        for(int j = 0; j < 3; j++) {
            for(int i = 0; i < 3; i++) {
                lidarFlattenTransformationMatrix(j, i) = lidarFlattenRotationMat.at<double>(j, i);
            }
        }
        pcl::transformPointCloud(*pLidarFloorCloud, *pLidarFloorCloud, lidarFlattenTransformationMatrix);
        pcl::SACSegmentation<pcl::PointXYZ> flattenLidarSacSeg;
        pcl::PointIndices::Ptr flattenLidarInliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr pFlattenLidarModelCoefficients(new pcl::ModelCoefficients());
        flattenLidarSacSeg.setOptimizeCoefficients(true);
        flattenLidarSacSeg.setModelType(pcl::SACMODEL_PLANE);
        flattenLidarSacSeg.setMethodType(pcl::SAC_RANSAC);
        flattenLidarSacSeg.setMaxIterations(100);
        flattenLidarSacSeg.setDistanceThreshold(0.025);
        flattenLidarSacSeg.setInputCloud(pLidarFloorCloud);
        flattenLidarSacSeg.segment(*flattenLidarInliers, *pFlattenLidarModelCoefficients);
        std::vector<double> flattenLidarFloorCoeffs = {pFlattenLidarModelCoefficients->values[0],
                                                       pFlattenLidarModelCoefficients->values[1],
                                                       pFlattenLidarModelCoefficients->values[2],
                                                       pFlattenLidarModelCoefficients->values[3]};

        cv::Mat cameraFlattenRotationMat;
        calcRotationVector3d(cv::Point3d(cameraFloorCoeffs[0], cameraFloorCoeffs[1], cameraFloorCoeffs[2]),
                             cv::Point3d(flattenLidarFloorCoeffs[0], flattenLidarFloorCoeffs[1], flattenLidarFloorCoeffs[2]),
                             cameraFlattenRotationMat);
        Eigen::Matrix4f cameraFlattenTransformationMatrix = Eigen::Matrix4f::Identity();
        for(int j = 0; j < 3; j++) {
            for(int i = 0; i < 3; i++) {
                cameraFlattenTransformationMatrix(j, i) = cameraFlattenRotationMat.at<double>(j, i);
            }
        }
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, cameraFlattenTransformationMatrix);
        pcl::SACSegmentation<pcl::PointXYZRGB> flattenCameraSacSeg;
        pcl::PointIndices::Ptr flattenCameraInliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr pFlattenCameraModelCoefficients(new pcl::ModelCoefficients());
        flattenCameraSacSeg.setOptimizeCoefficients(true);
        flattenCameraSacSeg.setModelType(pcl::SACMODEL_PLANE);
        flattenCameraSacSeg.setMethodType(pcl::SAC_RANSAC);
        flattenCameraSacSeg.setMaxIterations(100);
        flattenCameraSacSeg.setDistanceThreshold(0.025);
        flattenCameraSacSeg.setInputCloud(pCameraFloorCloud);
        flattenCameraSacSeg.segment(*flattenCameraInliers, *pFlattenCameraModelCoefficients);
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, cameraFlattenTransformationMatrix.inverse());
        cameraFlattenTransformationMatrix(2, 3) += -(pFlattenLidarModelCoefficients->values[3] - pFlattenCameraModelCoefficients->values[3]);
        pFlattenCameraModelCoefficients->values[3] = pFlattenLidarModelCoefficients->values[3];             
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, cameraFlattenTransformationMatrix);
        std::vector<double> flattenCameraFloorCoeffs = {pFlattenCameraModelCoefficients->values[0],
                                                        pFlattenCameraModelCoefficients->values[1],
                                                        pFlattenCameraModelCoefficients->values[2],
                                                        pFlattenCameraModelCoefficients->values[3]};
        
        
        Eigen::Matrix4f flattenCameraOdomTransformationMatrix = cameraFlattenTransformationMatrix*cameraOdomTransformationMatrix*cameraFlattenTransformationMatrix.inverse();
        Eigen::Matrix4f flattenLidarOdomTransformationMatrix = lidarFlattenTransformationMatrix*lidarOdomTransformationMatrix*lidarFlattenTransformationMatrix.inverse();
        if(flattenCameraOdomTransformationMatrix.size() == dequeMaxSize) {
            cameraOdomTransformationMatrixLet.pop_front();
            flattenCameraOdomTransformationMatrixLet.pop_front();
            lidarOdomTransformationMatrixLet.pop_front();
            flattenLidarOdomTransformationMatrixLet.pop_front();
        }
        cameraOdomTransformationMatrixLet.push_back(std::move(cameraOdomTransformationMatrix));
        flattenCameraOdomTransformationMatrixLet.push_back(std::move(flattenCameraOdomTransformationMatrix));
        lidarOdomTransformationMatrixLet.push_back(std::move(lidarOdomTransformationMatrix));
        flattenLidarOdomTransformationMatrixLet.push_back(std::move(flattenLidarOdomTransformationMatrix));

        geometry_msgs::PoseStamped cameraPose;
        geometry_msgs::PoseStamped lidarPose;
        geometry_msgs::PoseStamped gnssPose;
        cameraPose.header.frame_id = "show";
        lidarPose.header.frame_id = "show";
        gnssPose.header.frame_id = "show";
        Eigen::Quaternionf cameraPoseQuaternion;
        cameraPoseQuaternion = flattenCameraOdomTransformationMatrix.block<3, 3>(0, 0);
        cameraPose.pose.orientation.w = cameraPoseQuaternion.w();
        cameraPose.pose.orientation.x = cameraPoseQuaternion.x();
        cameraPose.pose.orientation.y = cameraPoseQuaternion.y();
        cameraPose.pose.orientation.z = cameraPoseQuaternion.z();
        cameraPose.pose.position.x = flattenCameraOdomTransformationMatrix(0, 3);
        cameraPose.pose.position.y = flattenCameraOdomTransformationMatrix(1, 3);
        cameraPose.pose.position.z = flattenCameraOdomTransformationMatrix(2, 3);
        Eigen::Quaternionf lidarPoseQuaternion;
        lidarPoseQuaternion = flattenLidarOdomTransformationMatrix.block<3, 3>(0, 0);
        lidarPose.pose.orientation.w = lidarPoseQuaternion.w();
        lidarPose.pose.orientation.x = lidarPoseQuaternion.x();
        lidarPose.pose.orientation.y = lidarPoseQuaternion.y();
        lidarPose.pose.orientation.z = lidarPoseQuaternion.z();
        lidarPose.pose.position.x = flattenLidarOdomTransformationMatrix(0, 3);
        lidarPose.pose.position.y = flattenLidarOdomTransformationMatrix(1, 3);
        lidarPose.pose.position.z = flattenLidarOdomTransformationMatrix(2, 3);
        Eigen::Quaternionf gnssPoseQuaternion;
        gnssPoseQuaternion = gnssImuOdomRotationMatrix;
        gnssPose.pose.orientation.w = gnssPoseQuaternion.w();
        gnssPose.pose.orientation.x = gnssPoseQuaternion.x();
        gnssPose.pose.orientation.y = gnssPoseQuaternion.y();
        gnssPose.pose.orientation.z = gnssPoseQuaternion.z();
        gnssPose.pose.position.x = gnssImuOdomTranslationMatrix(0, 0);
        gnssPose.pose.position.y = gnssImuOdomTranslationMatrix(1, 0);
        gnssPose.pose.position.z = gnssImuOdomTranslationMatrix(2, 0);
        cameraPosePub.publish(cameraPose);
        lidarPosePub.publish(lidarPose);
        gnssPosePub.publish(gnssPose);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pVisCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*pCurrCloud, *pVisCloud, lidarFlattenTransformationMatrix);
        flattenLidarPub.publish(*pVisCloud);

        double onlineInitLoss;
        Eigen::Matrix4f flattenLidarToflattenCameraTransformationMatrix = onlineCalibrate(flattenCameraOdomTransformationMatrixLet, flattenLidarOdomTransformationMatrixLet,
                                                                                          offlineLidarToCameraTransformationMatrix.block<3, 1>(0, 3), onlineInitLoss);
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, flattenLidarToflattenCameraTransformationMatrix.inverse());

        pcl::SACSegmentation<pcl::PointXYZRGB> zFittingSacSeg;
        pcl::PointIndices::Ptr zFittingInliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr pZFittingModelCoefficients(new pcl::ModelCoefficients());
        zFittingSacSeg.setOptimizeCoefficients(true);
        zFittingSacSeg.setModelType(pcl::SACMODEL_PLANE);
        zFittingSacSeg.setMethodType(pcl::SAC_RANSAC);
        zFittingSacSeg.setMaxIterations(100);
        zFittingSacSeg.setDistanceThreshold(0.025);
        zFittingSacSeg.setInputCloud(pCameraFloorCloud);
        zFittingSacSeg.segment(*zFittingInliers, *pZFittingModelCoefficients);
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, flattenLidarToflattenCameraTransformationMatrix);
        flattenLidarToflattenCameraTransformationMatrix(2, 3) += -(pZFittingModelCoefficients->values[3] - flattenLidarFloorCoeffs[3]);
        pcl::transformPointCloud(*pCameraFloorCloud, *pCameraFloorCloud, flattenLidarToflattenCameraTransformationMatrix.inverse());

        Eigen::Matrix4f onlineLidarToCameraTransformationMatrix = cameraFlattenTransformationMatrix.inverse()*flattenLidarToflattenCameraTransformationMatrix*lidarFlattenTransformationMatrix;

        double onlineSolverLoss;
        double onlineSolverRLoss;
        double onlineSolverTLoss;
        calcSolverLoss(cameraOdomTransformationMatrixLet, lidarOdomTransformationMatrixLet, onlineLidarToCameraTransformationMatrix, onlineSolverLoss, onlineSolverRLoss, onlineSolverTLoss);
        double offlineSolverLoss;
        double offlineSolverRLoss;
        double offlineSolverTLoss;
        calcSolverLoss(cameraOdomTransformationMatrixLet, lidarOdomTransformationMatrixLet, offlineLidarToCameraTransformationMatrix, offlineSolverLoss, offlineSolverRLoss, offlineSolverTLoss);
        double comparisonLoss;
        double comparisonRLoss;
        double comparisonTLoss;
        calcComparisonLoss(onlineLidarToCameraTransformationMatrix, offlineLidarToCameraTransformationMatrix, comparisonLoss, comparisonRLoss, comparisonTLoss);
        
        if((onlineSolverLoss < bestLoss)
            && (onlineSolverLoss > 0.1)
            // && (cameraOdomTransformationBasedOnLidarCoordiMatrixLet.size() >= dequeMaxSize)
        ) {
            bestLoss = onlineSolverLoss;
            bestRLoss = onlineSolverRLoss;
            bestTLoss = onlineSolverTLoss;
            bestOdomLidarToCameraTransformationMatrix = onlineLidarToCameraTransformationMatrix;
        }
        
        geometry_msgs::PolygonStamped lidarFloorPolygon;
        makePlanePolygon(flattenLidarFloorCoeffs, 1,
                         "show", ros::Time(0),
                         lidarFloorPolygon);
        lidarFloorCloudPub.publish(*pLidarFloorCloud);
        lidarFloorpolygonPub.publish(lidarFloorPolygon);
        geometry_msgs::PolygonStamped cameraFloorPolygon;
        makePlanePolygon(flattenCameraFloorCoeffs, 1,
                         "show", ros::Time(0),
                         cameraFloorPolygon);
        cameraFloorCloudPub.publish(*pCameraFloorCloud);
        cameraFloorpolygonPub.publish(cameraFloorPolygon);

        std::cout.precision(7);
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout << "---" << std::endl;
        std::cout << "Deque Size: " << flattenCameraOdomTransformationMatrixLet.size() << std::endl;
        std::cout << "Online Init Loss: " << onlineInitLoss << std::endl;
        std::cout << "Online Solver Loss: " << onlineSolverLoss << "\t|\t" << "Best Solver Loss: " << bestLoss << "\t|\t" << "Offline Solver Loss: " << offlineSolverLoss << "\t\t|\t" << "Comparison Loss: " << comparisonLoss  << std::endl;
        std::cout << "Online Solver R Loss: " << onlineSolverRLoss << "\t|\t" << "Best Solver R Loss: " << bestRLoss << "\t|\t" << "Offline Solver R Loss: " << offlineSolverRLoss << "\t|\t" << "Comparison R Loss: " << comparisonRLoss << std::endl;
        std::cout << "Online Solver t Loss: " << onlineSolverTLoss << "\t|\t" << "Best Solver t Loss: " << bestTLoss << "\t|\t" << "Offline Solver t Loss: " << offlineSolverTLoss << "\t|\t" << "Comparison t Loss: " << comparisonTLoss << std::endl;
        std::cout << "Online LiDAR to Camera Matrix\n" << onlineLidarToCameraTransformationMatrix << std::endl;
        std::cout << "Offline LiDAR to Camera Matrix\n" << offlineLidarToCameraTransformationMatrix << std::endl;
        std::cout << "Best LiDAR to Camera Matrix\n" << bestOdomLidarToCameraTransformationMatrix << std::endl;

        cv::Mat odomLidarToCameraRotationMat = (cv::Mat_<double>(3, 3));
        cv::Mat odomLidarToCameraRotationVec;
        cv::Mat odomLidarToCameraTranslationMat = (cv::Mat_<double>(3, 1));
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                odomLidarToCameraRotationMat.at<double>(row, col) = bestOdomLidarToCameraTransformationMatrix(row, col);
            }
        }
        cv::Rodrigues(odomLidarToCameraRotationMat, odomLidarToCameraRotationVec);
        odomLidarToCameraTranslationMat.at<double>(0, 0) = bestOdomLidarToCameraTransformationMatrix(0, 3);
        odomLidarToCameraTranslationMat.at<double>(1, 0) = bestOdomLidarToCameraTransformationMatrix(1, 3);
        odomLidarToCameraTranslationMat.at<double>(2, 0) = bestOdomLidarToCameraTransformationMatrix(2, 3);

        //fov move to LiDAR frame
        //innerproduct it to x-y plane(delete z value)
        //calculate angle's max and min
        std::vector<cv::Point3d> fovVertexPtAtLidarLet;
        std::vector<double> angleLet;
        double minAngle = std::numeric_limits<double>::max();
        double maxAngle = -std::numeric_limits<double>::max();
        for(int i = 0; i < fovVertexPtLet.size(); i++) {
            cv::Mat fovVertex = (cv::Mat_<double>(3, 1) << fovVertexPtLet[i].x,
                                 fovVertexPtLet[i].y,
                                 fovVertexPtLet[i].z);
            cv::Mat fovVertexPtAtLidar = odomLidarToCameraRotationMat.inv()*fovVertex;
            fovVertexPtAtLidarLet.push_back(cv::Point3d(fovVertexPtAtLidar.at<double>(0, 0), fovVertexPtAtLidar.at<double>(1, 0), fovVertexPtAtLidar.at<double>(2, 0)));
            double angle = (std::atan2(fovVertexPtAtLidar.at<double>(1, 0), fovVertexPtAtLidar.at<double>(0, 0)));
            angleLet.push_back(angle);

            minAngle = std::min(minAngle, angle);
            maxAngle = std::max(maxAngle, angle);
        }

        std::vector<cv::Point3d> lidarPointLetAtLidar;
        std::vector<double> intensityLet;
        for(int i = 0; i < pCloud->size(); i++) {
            double angle = std::atan2(pCloud->points[i].y, pCloud->points[i].x);
            if(angle < minAngle || angle > maxAngle) {
                continue;
            }
            lidarPointLetAtLidar.push_back(cv::Point3d(pCloud->points[i].x, pCloud->points[i].y, pCloud->points[i].z));
            intensityLet.push_back(pCloud->points[i].intensity);
        }
        std::vector<cv::Point2d> lidarPointLetAtCamera;
        cv::projectPoints(lidarPointLetAtLidar, odomLidarToCameraRotationMat, odomLidarToCameraTranslationMat, cameraMat, distCoeffs, lidarPointLetAtCamera);
        cv::Mat vis = img.clone();
        for(int i = 0; i < lidarPointLetAtCamera.size(); i++) {
            double distance = std::sqrt(std::pow(lidarPointLetAtLidar[i].x, 2) + std::pow(lidarPointLetAtLidar[i].y, 2) + std::pow(lidarPointLetAtLidar[i].z, 2));
            double angle = std::atan2(lidarPointLetAtLidar[i].y, lidarPointLetAtLidar[i].x);
            if(angle < minAngle || angle > maxAngle) {
                continue;
            }
            cv::circle(vis, lidarPointLetAtCamera[i], 1, cv::Scalar(0, (255.0/60.0)*distance, 255 - (255.0/60.0)*distance), -1);
            // cv::circle(vis, lidarPointLetAtCamera[i], 1, scalarHSV2BGR(intensityLet[i]*(180.0/255.0), 255, 255), -1);
        }
        cv::imshow("vis", vis);
        int wk = cv::waitKey(1);
        if(wk == 27) {
            exit(0);
        }

        cnt += sequenceGap;
        
        std::chrono::duration<double> processingTime = std::chrono::system_clock::now() - startTick;
        std::cout << "Processing Time: " << processingTime.count() << std::endl << std::endl;
    }
}

void OnlineCalibration::loadPose(std::string poseName, geometry_msgs::PoseStamped& pose) {
    std::ifstream poseTxt(poseName);
    std::string value;
    poseTxt >> value;
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.orientation.w = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.orientation.x = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.orientation.y = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.orientation.z = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.position.x = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.position.y = std::atof(value.c_str());
    poseTxt >> value;
    poseTxt >> value;
    pose.pose.position.z = std::atof(value.c_str());

    //to rotate imu axis
    Eigen::AngleAxisd zAxisAngle = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> imuRotationQ;
    imuRotationQ = zAxisAngle;
    Eigen::Quaternion<double> origianlQ;
    origianlQ.w() = pose.pose.orientation.w;
    origianlQ.x() = pose.pose.orientation.x;
    origianlQ.y() = pose.pose.orientation.y;
    origianlQ.z() = pose.pose.orientation.z;
    Eigen::Quaternion<double> rotatedQ = imuRotationQ*origianlQ;
    pose.pose.orientation.w = rotatedQ.w();
    pose.pose.orientation.x = rotatedQ.x();
    pose.pose.orientation.y = rotatedQ.y();
    pose.pose.orientation.z = rotatedQ.z();

    poseTxt.close();
}

void OnlineCalibration::estimateCameraFloor(cv::Mat img, cv::Mat cameraMatrix, cv::Mat distCoeffs,
                         HarDNet hardnet, BTS bts,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pCameraFloorCloud,
                         std::vector<double>& cameraFloorCoeffs) {
    pCameraFloorCloud->clear();
    cv::Mat hardnetInput = img;
    cv::Mat btsInput;
    cv::resize(img, btsInput, cv::Size(640, 352));

    std::vector<cv::Mat> hardnetChannels(3);
    cv::split(hardnetInput, hardnetChannels);
    torch::Tensor hardnetInputR = torch::from_blob(hardnetChannels[2].ptr(), {hardnetInput.rows, hardnetInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
    torch::Tensor hardnetInputG = torch::from_blob(hardnetChannels[1].ptr(), {hardnetInput.rows, hardnetInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
    torch::Tensor hardnetInputB = torch::from_blob(hardnetChannels[0].ptr(), {hardnetInput.rows, hardnetInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
    torch::Tensor hardnetInputBatchTensor = torch::cat({hardnetInputR, hardnetInputG, hardnetInputB}).view({1, 3, hardnetInput.rows, hardnetInput.cols}).to(device);

    std::vector<cv::Mat> btsChannels(3);
    cv::split(btsInput, btsChannels);
    torch::Tensor btsInputR = torch::from_blob(btsChannels[2].ptr(), {btsInput.rows, btsInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
    torch::Tensor btsInputG = torch::from_blob(btsChannels[1].ptr(), {btsInput.rows, btsInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
    torch::Tensor btsInputB = torch::from_blob(btsChannels[0].ptr(), {btsInput.rows, btsInput.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
    torch::Tensor btsInputBatchTensor = torch::cat({btsInputR, btsInputG, btsInputB}).view({1, 3, btsInput.rows, btsInput.cols}).to(device);

    torch::Tensor segPredBatchTensor = hardnet(hardnetInputBatchTensor);
    torch::Tensor segPredImgTensor = std::get<1>(segPredBatchTensor.max(1, true))[0][0].to(torch::kUInt8).to(torch::kCPU);
    cv::Mat segPredImg = cv::Mat(hardnetInput.rows, hardnetInput.cols, CV_8UC1, segPredImgTensor.data_ptr<uint8_t>());

    torch::Tensor depthPredBatchTensor = std::get<4>(bts(btsInputBatchTensor, torch::Tensor()));
    torch::Tensor depthPredImgTensor = depthPredBatchTensor[0].mul(100.0).to(torch::kInt16).to(torch::kCPU);
    cv::Mat depthPredImg = cv::Mat(btsInput.rows, btsInput.cols, CV_16UC1, (uint16_t*)depthPredImgTensor.data_ptr<int16_t>());
    cv::resize(depthPredImg, depthPredImg, cv::Size(640, 360));

    cv::Mat undistorted;
    if(lensModelIdx == 3) { //FIXME
        cv::omnidir::undistortImage(img, undistorted, cameraMatrix, distCoeffs, xi, cv::omnidir::RECTIFY_PERSPECTIVE, cameraMatrix, img.size());
    }
    else {
        cv::undistort(img, undistorted, cameraMatrix, distCoeffs);
    }

    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            int segValue = segPredImg.at<uchar>(j, i);
            if(segValue != 1) {
                continue;
            }
            double depth = (double)(depthPredImg.at<uint16_t>(j, i))/100.0;
            double xOverZ = (i - cx)/fx;
            double yOverZ = (j - cy)/fy;
            double z = depth/std::sqrt(1.0 + std::pow(xOverZ, 2) + std::pow(yOverZ, 2));
            double x = xOverZ*z;
            double y = yOverZ*z;
            if(z < 6.0) {
                continue;
            }
            int r = undistorted.at<cv::Vec3b>(j, i)[2];
            int g = undistorted.at<cv::Vec3b>(j, i)[1];
            int b = undistorted.at<cv::Vec3b>(j, i)[0];
            pcl::PointXYZRGB point;
            point.x = x;
            point.y = y;
            point.z = z;
            point.r = r;
            point.g = g;
            point.b = b;
            pCameraFloorCloud->push_back(std::move(point));
        }
    }

    pcl::SACSegmentation<pcl::PointXYZRGB> sacSeg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr pModelCoefficients(new pcl::ModelCoefficients());
    sacSeg.setOptimizeCoefficients(true);
    sacSeg.setModelType(pcl::SACMODEL_PLANE);
    sacSeg.setMethodType(pcl::SAC_RANSAC);
    sacSeg.setMaxIterations(100);
    sacSeg.setDistanceThreshold(0.025);
    sacSeg.setInputCloud(pCameraFloorCloud);
    sacSeg.segment(*inliers, *pModelCoefficients);

    cameraFloorCoeffs.push_back(pModelCoefficients->values[0]);
    cameraFloorCoeffs.push_back(pModelCoefficients->values[1]);
    cameraFloorCoeffs.push_back(pModelCoefficients->values[2]);
    cameraFloorCoeffs.push_back(pModelCoefficients->values[3]);
}

void OnlineCalibration::estimateLidarFloor(pcl::PointCloud<pcl::PointXYZI>::Ptr pCloud, std::shared_ptr<patchwork::PatchWorkpp> pPatchworkpp,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr& pLidarFloorCloud,
                        std::vector<double>& lidarFloorCoeffs) {
    pLidarFloorCloud->clear();
    Eigen::MatrixXf eigenCloud;
    eigenCloud.resize(pCloud->size(), 4);
    for(int i = 0; i < pCloud->size(); i++) {
        eigenCloud.row(i) << (float)pCloud->points[i].x, (float)pCloud->points[i].y, (float)pCloud->points[i].z, (float)pCloud->points[i].intensity;
    }
    pPatchworkpp->estimateGround(eigenCloud);

    Eigen::MatrixXf eigenFloorCloud = pPatchworkpp->getGround();
    for(int i = 0; i < eigenFloorCloud.rows(); i++) {
        pcl::PointXYZ point;
        point.x = eigenFloorCloud.row(i)(0);
        point.y = eigenFloorCloud.row(i)(1);
        point.z = eigenFloorCloud.row(i)(2);
        pLidarFloorCloud->push_back(point);
    }
    pLidarFloorCloud->header = pCloud->header;

    pcl::SACSegmentation<pcl::PointXYZ> sacSeg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::ModelCoefficients::Ptr pModelCoefficients(new pcl::ModelCoefficients());
    sacSeg.setOptimizeCoefficients(true);
    sacSeg.setModelType(pcl::SACMODEL_PLANE);
    sacSeg.setMethodType(pcl::SAC_RANSAC);
    sacSeg.setMaxIterations(100);
    sacSeg.setDistanceThreshold(0.025);
    sacSeg.setInputCloud(pLidarFloorCloud);
    sacSeg.segment(*inliers, *pModelCoefficients);

    lidarFloorCoeffs.push_back(pModelCoefficients->values[0]);
    lidarFloorCoeffs.push_back(pModelCoefficients->values[1]);
    lidarFloorCoeffs.push_back(pModelCoefficients->values[2]);
    lidarFloorCoeffs.push_back(pModelCoefficients->values[3]);
}

void OnlineCalibration::makePlanePolygon(std::vector<double> planeCoeffs, int drawOption,
                                         std::string frameId, ros::Time stamp,
                                         geometry_msgs::PolygonStamped &polygon) {
    // ax + by + cz + d = 0
    double a = planeCoeffs[0];
    double b = planeCoeffs[1];
    double c = planeCoeffs[2];
    double d = planeCoeffs[3];
    std::vector<cv::Point2d> planeVertexLet;
    planeVertexLet.push_back(cv::Point2d(-10.0, -10.0));
    planeVertexLet.push_back(cv::Point2d(10.0, -10.0));
    planeVertexLet.push_back(cv::Point2d(10.0, 10.0));
    planeVertexLet.push_back(cv::Point2d(-10.0, 10.0));
    polygon.header.frame_id = frameId;
    polygon.header.stamp = stamp;
    for(int i = 0; i < planeVertexLet.size(); i++) {
        double x, y, z;
        if(drawOption == 1) { // draw polygon on x-y plane
            x = planeVertexLet[i].x;
            y = planeVertexLet[i].y;
            z = (-a*x - b*y - d)/c;
        }
        else if(drawOption == 2) { // draw polygon on y-z plane
            y = planeVertexLet[i].x;
            z = planeVertexLet[i].y;
            x = (-b*y - c*z - d)/a;
        }
        else if(drawOption == 3) { // draw polygon on z-x plane
            z = planeVertexLet[i].x;
            x = planeVertexLet[i].y;
            y = (-a*x - c*z - d)/b;
        }
        
        geometry_msgs::Point32 point;
        point.x = x;
        point.y = y;
        point.z = z;
        polygon.polygon.points.push_back(point);
    }
}

void OnlineCalibration::calcRotationVector3d(const cv::Point3d fromV, const cv::Point3d toV, cv::Mat& rotationMatrix) {
    cv::Point3d cross = fromV.cross(toV);
    double ddot = fromV.ddot(toV);
    double normFromV = cv::norm(fromV);
    double normCross = cv::norm(cross);
    rotationMatrix = cv::Mat::zeros(3, 3, CV_64F);
    if(cross != cv::Point3d(0.0, 0.0, 0.0)) {
        cv::Mat skewSymmetricCrossMatrix = (cv::Mat_<double>(3, 3) << 0.0, -cross.z, cross.y,
                                            cross.z, 0.0, -cross.x,
                                            -cross.y, cross.x, 0.0);
        rotationMatrix = (cv::Mat::eye(3, 3, CV_64F) + skewSymmetricCrossMatrix + (skewSymmetricCrossMatrix*skewSymmetricCrossMatrix)*(1 - ddot)/std::pow(normCross, 2))/std::pow(normFromV, 2);
    }
}

casadi::SX OnlineCalibration::axisAngleToMatrix(casadi::SX rotvec) {
    casadi::SX roll = rotvec(0);
    casadi::SX pitch = rotvec(1);
    casadi::SX yaw = rotvec(2);
    casadi::SX rollMatrix = casadi::SX::eye(3);
    casadi::SX pitchMatrix = casadi::SX::eye(3);
    casadi::SX yawMatrix = casadi::SX::eye(3);
    rollMatrix(1, 1) = cos(roll);
    rollMatrix(1, 2) = -sin(roll);
    rollMatrix(2, 1) = sin(roll);
    rollMatrix(2, 2) = cos(roll);
    pitchMatrix(0, 0) = cos(pitch);
    pitchMatrix(0, 2) = sin(pitch);
    pitchMatrix(2, 0) = -sin(pitch);
    pitchMatrix(2, 2) = cos(pitch);
    yawMatrix(0, 0) = cos(yaw);
    yawMatrix(0, 1) = -sin(yaw);
    yawMatrix(1, 0) = sin(yaw);
    yawMatrix(1, 1) = cos(yaw);
    casadi::SX rotationMatrix = mtimes(yawMatrix, mtimes(pitchMatrix, rollMatrix));

    return rotationMatrix;
}

Eigen::Matrix3f OnlineCalibration::axisAngleToMatrix(Eigen::Vector3f rotationVector) {
    double roll = rotationVector(0);
    double pitch = rotationVector(1);
    double yaw = rotationVector(2);
    Eigen::Matrix3f rollMatrix;
    Eigen::Matrix3f pitchMatrix;
    Eigen::Matrix3f yawMatrix;
    rollMatrix << 1.0, 0.0, 0.0,
        0.0, std::cos(roll), -std::sin(roll),
        0.0, std::sin(roll), std::cos(roll);
    pitchMatrix << std::cos(pitch), 0.0, std::sin(pitch),
        0.0, 1.0, 0.0,
        -std::sin(pitch), 0.0, std::cos(pitch);
    yawMatrix << std::cos(yaw), -std::sin(yaw), 0.0,
        std::sin(yaw), std::cos(yaw), 0.0,
        0.0, 0.0, 1.0;
    Eigen::Matrix3f rotationMatrix = (yawMatrix*pitchMatrix*rollMatrix);

    return rotationMatrix;
}

casadi::SX OnlineCalibration::asTransitionMatrix(Eigen::Matrix3f r, Eigen::Matrix<float, 3, 1> t, bool inv) {
    casadi::SX transformationMatrix = casadi::SX::eye(4);
    Eigen::Matrix3f transformedR;
    Eigen::Matrix<float, 3, 1> transformedT;
    if(inv) {
        transformedR = r.transpose();
        transformedT = -r.transpose()*t;
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                transformationMatrix(row, col) = transformedR(row, col);
            }
            transformationMatrix(row, 3) = transformedT(row, 0);
        }
    }
    else {
        transformedR = r;
        transformedT = t;
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                transformationMatrix(row, col) = transformedR(row, col);
            }
            transformationMatrix(row, 3) = transformedT(row, 0);
        }
    }

    return transformationMatrix;
}

casadi::SX OnlineCalibration::onlineCalibrateCost(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                                                  casadi::SX rotvec, casadi::SX t) {
    casadi::SX R = axisAngleToMatrix(rotvec);
    casadi::SX T = vertcat(horzcat(R, t), horzcat(casadi::SX::zeros(1, 3), casadi::SX::ones(1, 1)));
    casadi::SX rCost = casadi::SX::zeros(1, 1);
    casadi::SX tCost = casadi::SX::zeros(1, 1);
    for(int i = 0; i < cameraTransformationMatrixLet.size(); i++) {
        //mtimes means @
        casadi::SX err = mtimes(asTransitionMatrix(cameraTransformationMatrixLet[i].block<3, 3>(0, 0), cameraTransformationMatrixLet[i].block<3, 1>(0, 3)), T) -
                         mtimes(T, asTransitionMatrix(lidarTransformationMatrixLet[i].block<3, 3>(0, 0), lidarTransformationMatrixLet[i].block<3, 1>(0, 3)));
        rCost += sqrt(sumsqr(err(casadi::Slice(0, 3), casadi::Slice(0, 3)))/9.0);
        tCost += sqrt(sumsqr(err(casadi::Slice(0, 3), 3))/3.0);
    }
    rCost /= cameraTransformationMatrixLet.size();
    tCost /= cameraTransformationMatrixLet.size();
    casadi::SX cost = 1.0*rCost + 1.0*tCost;

    return cost;
}

Eigen::Matrix4f OnlineCalibration::onlineCalibrate(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                                                   Eigen::Matrix<float, 3, 1> offlineTranslationMatrix, double &loss) {
    casadi::SX rotvec = casadi::SX::sym("rotvec", 3, 1);
    casadi::SX t = casadi::SX::sym("t", 3, 1);
    casadi::SX constraint = vertcat(rotvec(0), rotvec(1), rotvec(2), t(0), t(1), t(2));
    std::map<std::string, casadi::SX> nlp;
    nlp["x"] = vertcat(rotvec, t);
    nlp["f"] = onlineCalibrateCost(cameraTransformationMatrixLet, lidarTransformationMatrixLet, rotvec, t);
    nlp["g"] = constraint;

    float offlineX = offlineTranslationMatrix(0, 0);
    float offlineY = offlineTranslationMatrix(1, 0);
    float offlineZ = offlineTranslationMatrix(2, 0);

    casadi::Dict opts;
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    // opts["ipopt.tol"] = 1e-10;
    opts["ipopt.hessian_approximation"] = "limited-memory";
    casadi::Function solver = nlpsol("solver", "ipopt", nlp, opts);

    casadi::DMVector x0 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    casadi::DMDict arg = {{"x0", x0},
                          {"lbg", casadi::DM({0.0, 0.0, -M_PI, offlineX - 0.05, offlineY - 0.05, offlineZ - 0.05})},
                          {"ubg", casadi::DM({0.0, 0.0, M_PI, offlineX + 0.05, offlineY + 0.05, offlineZ + 0.05})}};
    casadi::DMDict result = solver(arg);
    loss = (double)result["f"];

    Eigen::Vector3f rotationVector((double)(result["x"](0)), (double)(result["x"](1)), (double)(result["x"](2)));
    Eigen::Matrix4f transformationMatrix = Eigen::Matrix4f::Identity();
    transformationMatrix.block<3, 3>(0, 0) = axisAngleToMatrix(rotationVector);;
    transformationMatrix(0, 3) = (double)(result["x"](3));
    transformationMatrix(1, 3) = (double)(result["x"](4));
    transformationMatrix(2, 3) = (double)(result["x"](5));

    return transformationMatrix;
}

void OnlineCalibration::calcSolverLoss(std::deque<Eigen::Matrix4f> cameraTransformationMatrixLet, std::deque<Eigen::Matrix4f> lidarTransformationMatrixLet,
                                       Eigen::Matrix4f transformationMatrix, double &loss, double &rLoss, double &tLoss) {
    double rCost = 0.0;
    double tCost = 0.0;
    for(int i = 0; i < cameraTransformationMatrixLet.size(); i++) {
        double rCostTmp = 0.0;
        double tCostTmp = 0.0;
        Eigen::Matrix4f err = cameraTransformationMatrixLet[i]*transformationMatrix - transformationMatrix*lidarTransformationMatrixLet[i];
        for(int row = 0; row < 3; row++) {
            for(int col = 0; col < 3; col++) {
                rCostTmp += std::pow(err(row, col), 2);
            }
            tCostTmp += std::pow(err(row, 3), 2);
        }
        rCostTmp /= 9.0;
        rCostTmp = std::sqrt(rCostTmp);
        tCostTmp /= 3.0;
        tCostTmp = std::sqrt(tCostTmp);
        rCost += rCostTmp;
        tCost += tCostTmp;
    }
    rCost /= cameraTransformationMatrixLet.size();
    tCost /= cameraTransformationMatrixLet.size();
    
    rLoss = rCost;
    tLoss = tCost;
    loss = 1.0*rCost + 1.0*tCost;
}

void OnlineCalibration::calcComparisonLoss(Eigen::Matrix4f transformationMatrix, Eigen::Matrix4f referenceTransformationMatrix, double& loss, double& rLoss, double& tLoss) {
    rLoss = 0.0;
    tLoss = 0.0;

    Eigen::Matrix3f comparedRotationMatrix = transformationMatrix.block<3, 3>(0, 0);
    Eigen::Matrix<float, 3, 1> comparedTranslationMatrix = transformationMatrix.block<3, 1>(0, 3);
    Eigen::Matrix3f referenceRotationMatrix = referenceTransformationMatrix.block<3, 3>(0, 0);
    Eigen::Matrix<float, 3, 1> referenceTranslationMatrix = referenceTransformationMatrix.block<3, 1>(0, 3);

    Eigen::Matrix3f relativeRotationMatrix = comparedRotationMatrix.inverse()*referenceRotationMatrix;
    Eigen::Matrix<float, 3, 1> relativeTranslationMatrix = referenceTranslationMatrix - comparedTranslationMatrix;

    for(int row = 0; row < 3; row++) {
        for(int col = 0; col < 3; col++) {
            rLoss += std::sqrt(std::pow(relativeRotationMatrix(row, col), 2));
        }
        tLoss += std::sqrt(std::pow(relativeTranslationMatrix(row, 0), 2));
    }
    rLoss /= 9;
    tLoss /= 3;
    loss = 1.0*rLoss + 1.0*tLoss;
}

cv::Scalar OnlineCalibration::scalarHSV2BGR(uchar h, uchar s, uchar v) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
    cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}