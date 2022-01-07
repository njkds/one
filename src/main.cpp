#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"
//C++
#include <iostream>
#include <chrono>
#include <string>
//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//Kinect DK
#include <k4a/k4a.hpp>
#include <cmath>
//PCL
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/io.h>
#include <pcl/features/integral_image_normal.h>  //法线估计类头文件
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>             //模型系数头文件
#include <pcl/filters/project_inliers.h>       //投影滤波类头文件

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
typedef struct L {
    int cls;
    double x, y, z, nx, ny, nz, pre_r, real_r;
} Lagori;

//void showPointCloud(int num,
//                    const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer,
//                    int vp,
//                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_filtered,
//                    const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals,
//                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_projected) {
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_filtered_color(source_filtered, 255, 0, 0);
//    viewer->addPointCloud<pcl::PointXYZ>(source_filtered, source_filtered_color,
//                                         "source_filtered_" + std::to_string(num), vp);
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
//                                             "source_filtered_" + std::to_string(num));
//    viewer->addCoordinateSystem(1.0);  // 显示坐标
//    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(source_filtered, cloud_normals, 3, 50,
//                                                             "normals_" + std::to_string(num), vp);
//
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_projected_color(source_projected, 255, 0, 0);
//    viewer->addPointCloud<pcl::PointXYZ>(source_projected, source_projected_color,
//                                         "source_projected_" + std::to_string(num), vp);
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
//                                             "source_projected_" + std::to_string(num));
//    viewer->addCoordinateSystem(1.0);  // 显示坐标
//}

void getLagori(Lagori &lagori,
               const pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
               const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_filtered,
               const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals,
               const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_projected) {
    // 创建滤波器，对每个点分析的临近点的个数设置为50 ，并将标准差的倍数设置为1  这意味着如果一
    // 个点的距离超出了平均距离一个标准差以上，则该点被标记为离群点，并将它移除，存储起来
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;   //创建滤波器对象
    sor.setInputCloud(source);                     //设置待滤波的点云
    sor.setMeanK(100);                              //设置在进行统计时考虑查询点临近点数
    sor.setStddevMulThresh(1);                //设置判断是否为离群点的阀值
    sor.filter(*source_filtered);                    //存储

    // VoxelGrid 滤波
    pcl::PCLPointCloud2::Ptr pcl2(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*source_filtered, *pcl2);
    pcl::VoxelGrid<pcl::PCLPointCloud2> vg;  //创建滤波对象
    vg.setInputCloud(pcl2);            //设置需要过滤的点云给滤波对象
    vg.setLeafSize(10.f, 10.f, 10.f);  //设置滤波时创建的体素体积为1cm的立方体
    vg.filter(*pcl2);           //执行滤波处理，存储输出
    pcl::fromPCLPointCloud2(*pcl2, *source_filtered);
//    pcl::VoxelGrid<pcl::PointXYZ> vg;  //创建滤波对象
//    vg.setInputCloud(source_filtered);            //设置需要过滤的点云给滤波对象
//    vg.setLeafSize(10.f, 10.f, 10.f);  //设置滤波时创建的体素体积为1cm的立方体
//    vg.filter(*source_filtered);           //执行滤波处理，存储输出

    //创建法线估计估计向量
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(source_filtered);
    //创建一个空的KdTree对象，并把它传递给法线估计向量
    //基于给出的输入数据集，KdTree将被建立
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    //使用半径在查询点周围3厘米范围内的所有临近元素
    ne.setRadiusSearch(30.0);
    //计算特征值
    ne.compute(*cloud_normals);
    // cloud_normals->points.size ()应该与input cloud_downsampled->points.size ()有相同的尺寸

    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg; //分割对象
    pcl::ExtractIndices<pcl::PointXYZ> extract;                      //点提取对象
//    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
//    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);

//    // 过滤平面
//    seg.setOptimizeCoefficients(true);
//    seg.setModelType(pcl::SACMODEL_PLANE);
//    seg.setMethodType(pcl::SAC_RANSAC);
//    seg.setDistanceThreshold(0.01);
//    seg.setMaxIterations(500);
//    seg.setInputCloud(source_filtered);
//    seg.segment(*inliers_plane, *coefficients_plane);
//    seg.setInputNormals(cloud_normals);
//    extract.setInputCloud(source_filtered);
//    extract.setIndices(inliers_plane);
//    extract.setNegative(true);
//    extract.filter(*source_filtered);

    // 提取圆柱
    // Create the segmentation object for cylinder segmentation and set all the parameters
    seg.setOptimizeCoefficients(true);        //设置对估计模型优化
    seg.setModelType(pcl::SACMODEL_CYLINDER); //设置分割模型为圆柱形
    seg.setMethodType(pcl::SAC_RANSAC);       //参数估计方法
    seg.setNormalDistanceWeight(0.2);         //设置表面法线权重系数
    seg.setMaxIterations(1000);              //设置迭代的最大次数10000
    seg.setDistanceThreshold(7);           //设置内点到模型的距离允许最大值
    seg.setRadiusLimits(90, 300);              //设置估计出的圆柱模型的半径的范围
    seg.setInputCloud(source_filtered);
    seg.setInputNormals(cloud_normals);
    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients_cylinder);
    extract.setInputCloud(source_filtered);
    extract.setIndices(inliers_cylinder);
    extract.setNegative(false);
    extract.filter(*source_filtered);

    // 投影到轴线
    //定义模型系数对象，并填充对应的数据
    pcl::ModelCoefficients::Ptr coefficients_line(new pcl::ModelCoefficients());
    coefficients_line->values.resize(6);
    coefficients_line->values[0] = coefficients_cylinder->values[0];
    coefficients_line->values[1] = coefficients_cylinder->values[1];
    coefficients_line->values[2] = coefficients_cylinder->values[2];
    coefficients_line->values[3] = coefficients_cylinder->values[3];
    coefficients_line->values[4] = coefficients_cylinder->values[4];
    coefficients_line->values[5] = coefficients_cylinder->values[5];

    // 创建ProjectInliers对象，使用ModelCoefficients作为投影对象的模型参数
    pcl::ProjectInliers<pcl::PointXYZ> proj;     //创建投影滤波对象
    proj.setModelType(pcl::SACMODEL_LINE);      //设置对象对应的投影模型
    proj.setInputCloud(source_filtered);                   //设置输入点云
    proj.setModelCoefficients(coefficients_line);       //设置模型对应的系数
    proj.filter(*source_projected);                 //投影结果存储

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*source_projected, centroid);

    lagori.x = centroid(0);
    lagori.y = centroid(1);
    lagori.z = centroid(2);
    lagori.nx = coefficients_cylinder->values[3];
    lagori.ny = coefficients_cylinder->values[4];
    lagori.nz = coefficients_cylinder->values[5];
    lagori.pre_r = coefficients_cylinder->values[6];

    if (lagori.pre_r > 80 && lagori.pre_r <= 118.75) {
        lagori.cls = 1;
        lagori.real_r = 100;
    } else if (lagori.pre_r > 118.75 && lagori.pre_r <= 156.25) {
        lagori.cls = 2;
        lagori.real_r = 137.5;
    } else if (lagori.pre_r > 156.25 && lagori.pre_r <= 193.75) {
        lagori.cls = 3;
        lagori.real_r = 175;
    } else if (lagori.pre_r > 193.75 && lagori.pre_r <= 231.25) {
        lagori.cls = 4;
        lagori.real_r = 212.5;
    } else if (lagori.pre_r > 231.25 && lagori.pre_r <= 270) {
        lagori.cls = 5;
        lagori.real_r = 250;
    } else {
        lagori.cls = 0;
    }
}

static void create_xy_table(const k4a_calibration_t *calibration,
                            k4a_image_t xy_table) {
    auto *table_data = (k4a_float2_t *) (void *) k4a_image_get_buffer(xy_table);

//    int width = calibration->depth_camera_calibration.resolution_width;
//    int height = calibration->depth_camera_calibration.resolution_height;
    int width = calibration->color_camera_calibration.resolution_width;
    int height = calibration->color_camera_calibration.resolution_height;

    k4a_float2_t p;
    k4a_float3_t ray;
    int valid;

    for (int y = 0, idx = 0; y < height; y++) {
        p.xy.y = (float) y;
        for (int x = 0; x < width; x++, idx++) {
            p.xy.x = (float) x;

            k4a_calibration_2d_to_3d(
                    calibration, &p, 1.f, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &ray, &valid);

            if (valid) {
                table_data[idx].xy.x = ray.xyz.x;
                table_data[idx].xy.y = ray.xyz.y;
//                printf("%f,%f\n",ray.xyz.x,ray.xyz.y);
            } else {
                table_data[idx].xy.x = nanf("");
                table_data[idx].xy.y = nanf("");
            }
        }
    }
}

static void generate_point_cloud(const k4a::image &depth_image,
                                 const k4a_image_t &xy_table,
                                 k4a_image_t &point_cloud,
                                 int *point_count) {
    int width = depth_image.get_width_pixels();
    int height = depth_image.get_height_pixels();
    //int height = k4a_image_get_height_pixels(depth_image);

    auto *depth_data = (uint16_t *) (void *) depth_image.get_buffer();
    auto *xy_table_data = (k4a_float2_t *) (void *) k4a_image_get_buffer(xy_table);
    auto *point_cloud_data = (k4a_float3_t *) (void *) k4a_image_get_buffer(point_cloud);

    *point_count = 0;
    for (int i = 0; i < width * height; i++) {
        if (depth_data[i] != 0 && !isnan(xy_table_data[i].xy.x) && !isnan(xy_table_data[i].xy.y)) {
            point_cloud_data[i].xyz.x = xy_table_data[i].xy.x * (float) depth_data[i];
            point_cloud_data[i].xyz.y = xy_table_data[i].xy.y * (float) depth_data[i];
            point_cloud_data[i].xyz.z = (float) depth_data[i];
            (*point_count)++;
        } else {
            point_cloud_data[i].xyz.x = nanf("");
            point_cloud_data[i].xyz.y = nanf("");
            point_cloud_data[i].xyz.z = nanf("");
        }
    }
}

static void write_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                              const k4a_image_t &point_cloud) {


    int width = k4a_image_get_width_pixels(point_cloud);
    int height = k4a_image_get_height_pixels(point_cloud);

    auto *point_cloud_data = (k4a_float3_t *) (void *) k4a_image_get_buffer(point_cloud);


    for (int i = 0; i < width * height; i++) {
        if (isnan(point_cloud_data[i].xyz.x) || isnan(point_cloud_data[i].xyz.y) || isnan(point_cloud_data[i].xyz.z)) {
            continue;
        }
        pcl::PointXYZ point;
        point.x = point_cloud_data[i].xyz.x;
        point.y = point_cloud_data[i].xyz.y;
        point.z = point_cloud_data[i].xyz.z;
        cloud->points.push_back(point);
    }
}

void getPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   const cv::Mat &cv_depth,
                   k4a::calibration k4aCalibration) {
    k4a_image_t xy_table = nullptr;
    k4a_image_t point_cloud = nullptr;
    k4a_image_t depth = nullptr;
    int point_count = 0;

    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                     k4aCalibration.color_camera_calibration.resolution_width,
                     k4aCalibration.color_camera_calibration.resolution_height,
                     k4aCalibration.color_camera_calibration.resolution_width * (int) sizeof(k4a_float2_t),
                     &xy_table);

    create_xy_table(&k4aCalibration, xy_table);

    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                     k4aCalibration.color_camera_calibration.resolution_width,
                     k4aCalibration.color_camera_calibration.resolution_height,
                     k4aCalibration.color_camera_calibration.resolution_width * (int) sizeof(k4a_float3_t),
                     &point_cloud);

    k4a_image_create_from_buffer(K4A_IMAGE_FORMAT_DEPTH16,
                                 k4aCalibration.color_camera_calibration.resolution_width,
                                 k4aCalibration.color_camera_calibration.resolution_height,
                                 k4aCalibration.color_camera_calibration.resolution_width *
                                 (int) sizeof(uint16_t),
                                 cv_depth.data,
                                 k4aCalibration.color_camera_calibration.resolution_width *
                                 k4aCalibration.color_camera_calibration.resolution_height * 2,
                                 nullptr,
                                 nullptr,
                                 &depth);

    generate_point_cloud(depth, xy_table, point_cloud, &point_count);
    write_point_cloud(cloud, point_cloud);


    k4a_image_release(xy_table);
    k4a_image_release(point_cloud);
    k4a_image_release(depth);
}

void init_kinect(uint32_t &device_count,
                 k4a::device &device,
                 k4a_device_configuration_t &config,
                 k4a::capture &capture) {
    //发现已连接的设备数
    device_count = k4a::device::get_installed_count();
    if (device_count == 0) {
        cout << "Error: no K4A devices found. " << endl;
        return;
    } else {
        std::cout << "Found " << device_count << " connected devices. " << std::endl;
    }

    //打开（默认）设备
    device = k4a::device::open(K4A_DEVICE_DEFAULT);
    std::cout << "Done: open device. " << std::endl;

    //配置并启动设备
    config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
//	config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    config.synchronized_images_only = true; // ensures that depth and color images are both available in the capture
    device.start_cameras(&config);
    std::cout << "Done: start camera." << std::endl;

    //稳定化
    int iAuto = 0;//用来稳定，类似自动曝光
    while (iAuto < 30) {
        if (device.get_capture(&capture))
            iAuto++;
    }
}

void kinect() {
//    TRT::compile(
//            TRT::Mode::FP32,
//            1,
//            "/home/dell/Desktop/yolov5/best.onnx",
//            "my-best.trtmodel"
//    );
//    INFO("Done");

    auto yolo = Yolo::create_infer(
            "my-best.trtmodel",
            Yolo::Type::V5,
            0, 0.25f, 0.5f
    );


    uint32_t device_count;
    k4a::device device;
    k4a_device_configuration_t config;
    k4a::capture capture;
    init_kinect(device_count, device, config, capture);

    k4a::image rgbImage;
    k4a::image depthImage;
    k4a::image transformed_depthImage;

    cv::Mat cv_rgbImage;
    cv::Mat cv_depthImage;

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_projected(new pcl::PointCloud<pcl::PointXYZ>);

    while (true) {
        auto start = std::chrono::system_clock::now();
        if (device.get_capture(&capture, std::chrono::milliseconds(1000))) {
            rgbImage = capture.get_color_image();
            depthImage = capture.get_depth_image();

            //深度图和RGB图配准
            //Get the camera calibration for the entire K4A device, which is used for all transformation functions.
            k4a::calibration k4aCalibration = device.get_calibration(config.depth_mode, config.color_resolution);
            k4a::transformation k4aTransformation = k4a::transformation(k4aCalibration);
            transformed_depthImage = k4aTransformation.depth_image_to_color_camera(depthImage);

            cv_rgbImage = cv::Mat(rgbImage.get_height_pixels(),
                                  rgbImage.get_width_pixels(),
                                  CV_8UC4,
                                  (void *) rgbImage.get_buffer());
            cv::cvtColor(cv_rgbImage, cv_rgbImage, cv::COLOR_BGRA2BGR);

            cv_depthImage = cv::Mat(transformed_depthImage.get_height_pixels(),
                                    transformed_depthImage.get_width_pixels(),
                                    CV_16U,
                                    (void *) transformed_depthImage.get_buffer(),
                                    static_cast<size_t>(transformed_depthImage.get_stride_bytes()));

            auto bboxes = yolo->commit(cv_rgbImage).get();
            int imgHeight = cv_depthImage.rows;
            int imgWidth = cv_depthImage.cols;

            std::vector<Lagori> lagori;

            for (auto &box: bboxes) {
                uint8_t r, g, b;
                std::tie(r, g, b) = iLogger::random_color(box.class_label);

                cv::rectangle(
                        cv_rgbImage,
                        cv::Point((int) box.left, box.top),
                        cv::Point((int) box.right, box.bottom),
                        cv::Scalar(b, g, r),
                        3
                );

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                Lagori l;

                cv::Mat depthCut = cv::Mat::zeros(cv::Size(imgWidth, imgHeight), CV_16U);
                cv::Rect select = cv::Rect((int) box.left,
                                           (int) box.top,
                                           (int) (box.right - box.left),
                                           (int) (box.bottom - box.top));



                cv_depthImage(select).copyTo(depthCut(select));
                getPointCloud(cloud, depthCut, k4aCalibration);
                if (cloud->size() > 3000 && cloud->is_dense) {
                    getLagori(l, cloud, source_filtered, cloud_normals, source_projected);
                    if (l.cls != 0) lagori.push_back(l);
                    cout << "点数： " << cloud->size() << endl;
                }

                cloud.reset();
            }

            cv::imshow("img", cv_rgbImage);
            cv::waitKey(1);

            for (auto &i: lagori) {
                cout << "Lagori 编号：" << i.cls << endl;
                cout << "中心坐标：(" << i.x << ", " << i.y << ", " << i.z << ") " << endl;
                cout << "轴线方向：(" << i.nx << ", " << i.ny << ", " << i.nz << ") " << endl;
                cout << "观测半径：" << i.pre_r << endl;
                cout << "实际半径：" << i.real_r << endl;
            }

            k4aTransformation.destroy();
            cv_rgbImage.release();
            cv_depthImage.release();
            capture.reset();
        } else {
            std::cout << "false: K4A_WAIT_RESULT_TIMEOUT." << std::endl;
        }
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cout << double(duration.count()) * std::chrono::microseconds::period::num /
                std::chrono::microseconds::period::den
             << "s" << endl;
    }

    // 释放，关闭设备
    rgbImage.reset();
    depthImage.reset();
    capture.reset();
    device.close();
}

int main() {

    kinect();

    return 0;
}

#pragma clang diagnostic pop