#include "util.h"

void cv::omnidir::distortPoints(InputArray undistorted, OutputArray distorted,
                                InputArray K, InputArray D, InputArray xi) {
    CV_Assert(undistorted.type() == CV_64FC2 || undistorted.type() == CV_32FC2);
    CV_Assert((D.depth() == CV_64F || D.depth() == CV_32F) && D.total() == 4);
    CV_Assert(K.size() == Size(3, 3) && (K.depth() == CV_64F || K.depth() == CV_32F));
    CV_Assert(xi.total() == 1 && (xi.depth() == CV_64F || xi.depth() == CV_32F));

    distorted.create(undistorted.size(), undistorted.type());

    cv::Vec2d f, c;
    double s = 0.0;
    if (K.depth() == CV_32F) {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0,0), camMat(1,1));
        c = Vec2f(camMat(0,2), camMat(1,2));
        s = (double)camMat(0,1);
    }
    else if (K.depth() == CV_64F) {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0,0), camMat(1,1));
        c = Vec2d(camMat(0,2), camMat(1,2));
        s = camMat(0,1);
    }

    Vec4d kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>() : (Vec4d)*D.getMat().ptr<Vec4d>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);

    double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();

    const cv::Vec2d *srcd = undistorted.getMat().ptr<cv::Vec2d>();
    const cv::Vec2f *srcf = undistorted.getMat().ptr<cv::Vec2f>();

    cv::Vec2d *dstd = distorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f *dstf = distorted.getMat().ptr<cv::Vec2f>();

    int n = (int)undistorted.total();
    for(int i = 0; i < n; i++) {
        Vec3d Xpu;
        Xpu[0] = undistorted.depth() == CV_32F ? (double)srcf[i][0] : srcd[i][0];
        Xpu[1] = undistorted.depth() == CV_32F ? (double)srcf[i][1] : srcd[i][1];
        Xpu[2] = 1.0;

        // convert to unit sphere
        Vec3d Xs = Xpu/cv::norm(Xpu);

        // convert to normalized image plane
        Vec2d xu = Vec2d(Xs[0]/(Xs[2] + _xi), Xs[1]/(Xs[2] + _xi));

        // add distortion
        Vec2d xd;
        double r2 = xu[0]*xu[0] + xu[1]*xu[1];
        double r4 = r2*r2;

        xd[0] = xu[0]*(1 + k[0]*r2 + k[1]*r4) + 2*p[0]*xu[0]*xu[1] + p[1]*(r2+2*xu[0]*xu[0]);
        xd[1] = xu[1]*(1 + k[0]*r2 + k[1]*r4) + p[0]*(r2+2*xu[1]*xu[1]) + 2*p[1]*xu[0]*xu[1];

        // convert to pixel coordinate
        Vec2d final;
        final[0] = f[0]*xd[0] + s*xd[1] + c[0];
        final[1] = f[1]*xd[1] + c[1];

        if (distorted.depth() == CV_32F) {
            dstf[i] = final;
        }
        else {
            dstd[i] = final;
        }
    }
}