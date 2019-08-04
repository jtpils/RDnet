# RD_Net
Rotation and Density network for sparse point-clouds.created by haixu yan.

## Introduction
***RD_Net*** is a multi-feature analysis method to facilitate simple and efficient classification and segmentation for the significant number of point-clouds geodetic surveysing staff. In this respository,I basically used a similar approach to [PointNet++'s](https://github.com/charlesq34/pointnet2)——Ordering point clouds by use Max-Pooling layer as Symmetric Function——to directly consumes point-clouds.It's really a practical and efficient method,I love that.Note that the biggest feature of RD_Net is the ***Model_A+B.py***.it aggregates different features of point-clouds for training, thus making sparse point-clouds recognition become more efficient.