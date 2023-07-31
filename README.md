# Camera-LiDAR_Online_Calibration

This repository utilizes a lot of 3rdparty libraries(C++ Only), they are as follows.  
(All libraries are modified for structure as what I configure, and you don't have to download libraries)  
#### Own Libraries
[BTS_cpp](https://github.com/PigBroA/BTS_cpp)  
[HarDNet_cpp](https://github.com/PigBroA/HarDNet_cpp)  
calibration_dk  
#### Others Libraries(Thanks a lot for all)
[SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT)  
[glob](https://github.com/p-ranav/glob)  
[patchwork-plusplus](https://github.com/url-kaist/patchwork-plusplus)  
[ndt_omp](https://github.com/koide3/ndt_omp)  

## Environment
 * Ubuntu 20.04.3 LTS
 * ROS noetic 1.15.14
 * GCC 10.3.0
 * CMake 3.22.2
 * LibTorch 1.9.1
 * CUDA 11.1
 * OpenCV 4.2.0
 * Eigen 3.3.7
 * PCL 1.10.1
 * Yaml-cpp 0.6.2
 * CasADi 3.6.1(with IPOPT, please follow [link](https://github.com/zehuilu/Tutorial-on-CasADi-with-CPP/issues/2#issuecomment-1518988014) for installation)

## Download Large Files
| Sample DB | BTS Libtorch .pt File | HarDNet Libtorch .pt File | SuperPoint TensorRT .engine File | SuperPoint .onnx File | SuperGlue TensorRT .engine File | SuperGlue .onnx File |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/EZ3XaZ393nJJmI7WITic5xwB6bIfqD2ghS2A5be7KBoQBQ?e=TC3Ohp) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/ES0GPFV8I8pHnr8LmZd_I3ABNgdrchMxoSgWl248G39EtA?e=eqlknF) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/EQjpX7EyELZLsphyej7jbUYBI3rRHNbbkP65s5hLL8BTuw?e=pXwxeA) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/EWZT_efATpZIuv4zKpcX8c0B8uIoeBMZ1cdj8_tQm-x3FA?e=XeN52B) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/ESfj4wqqV5VCpNI-kw9YzAUBzyLwPHPIDeCM_isO9LjVGw?e=FVlQ1s) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/Ee4PDVASK2dDtG_E0qZ2IV8B_X2vASLWCAe1q7UBlXHYMQ?e=XyzM9J) | [Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/EYAAt5tgwuhNvIIiQ5fokOoB6gNk8561mlBu-hMv9BfwnQ?e=PBTPrJ) |
| [Tree](./ros/src/online_calibration_with_stored_data/sample_db) | [Tree](./ros/src/online_calibration_with_stored_data/model/bts) | [Tree](./ros/src/online_calibration_with_stored_data/model/hardnet) | [Tree](./ros/src/online_calibration_with_stored_data/model/superglue) | [Tree](./ros/src/online_calibration_with_stored_data/model/superglue) | [Tree](./ros/src/online_calibration_with_stored_data/model/superglue) | [Tree](./ros/src/online_calibration_with_stored_data/model/superglue) |

## Package Folder Tree
Camera-LiDAR_Online_Calibration
 * [ros](./ros)
     * [src](./ros/src)
         * [online_calibration_with_stored_data](./ros/src/online_calibration_with_stored_data)
             * [rviz](./ros/src/online_calibration_with_stored_data/rviz)
             * [src](./ros/src/online_calibration_with_stored_data/src)
             * [3rdparty](./ros/src/online_calibration_with_stored_data/3rdparty)
               * [tensorrtbuffer](./ros/src/online_calibration_with_stored_data/3rdparty/tensorrtbuffer)
                 * [include](./ros/src/online_calibration_with_stored_data/3rdparty/tensorrtbuffer/include)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/tensorrtbuffer/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/tensorrtbuffer/lib)
               * [bts](./ros/src/online_calibration_with_stored_data/3rdparty/bts)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/bts/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/bts/lib)
               * [calibration_dk](./ros/src/online_calibration_with_stored_data/3rdparty/calibration_dk)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/calibration_dk/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/calibration_dk/lib)
               * [fullyHardNet](./ros/src/online_calibration_with_stored_data/3rdparty/fullyHardNet)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/fullyHardNet/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/fullyHardNet/lib)
               * [glob](./ros/src/online_calibration_with_stored_data/3rdparty/glob)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/glob/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/glob/lib)
               * [patchworkpp](./ros/src/online_calibration_with_stored_data/3rdparty/patchworkpp)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/patchworkpp/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/patchworkpp/lib)
               * [pclomp](./ros/src/online_calibration_with_stored_data/3rdparty/pclomp)
                 * [src](./ros/src/online_calibration_with_stored_data/3rdparty/pclomp/src)
                 * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/pclomp/lib)
               * [superglue](./ros/src/online_calibration_with_stored_data/3rdparty/superglue)
               * [src](./ros/src/online_calibration_with_stored_data/3rdparty/superglue/src)
               * [lib](./ros/src/online_calibration_with_stored_data/3rdparty/superglue/lib)
             * [pre_cali](./ros/src/online_calibration_with_stored_data/pre_cali)
             * [model](./ros/src/online_calibration_with_stored_data/model)
               * [hardnet](./ros/src/online_calibration_with_stored_data/model/hardnet)
               * [superglue](./ros/src/online_calibration_with_stored_data/model/superglue)
               * [bts](./ros/src/online_calibration_with_stored_data/model/bts)
             * [sample_db](./ros/src/online_calibration_with_stored_data/sample_db)

## Contact
If you have any question, please contact me
 * Dongkyu Lee, dlehdrb3909@chungbuk.ac.kr or dlehdrb3909@gmail.com
