#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

namespace cv {
    namespace omnidir {
        void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, InputArray xi);
    }
}