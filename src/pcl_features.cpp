/**
 *  @file pcl_features.cpp
 *  @authors Miguel Saavedra - Gustavo Salazar
 *  @brief Global feature extractor
 *  @version 0.1
 *  @date 25-09-2020
 *
 *  useful links: http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_4:_3D_object_recognition_(descriptors)
 * 
 *  http://www.willowgarage.com/sites/default/files/Rusu10IROS.pdf
 * 
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
// Descriptors
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/esf.h>
#include <pcl/features/grsd.h>

#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <fstream>
#include <experimental/filesystem>
#include <vector> 
#include <boost/algorithm/string.hpp>

namespace fs = std::experimental::filesystem;

int main (int argc, char** argv)
{

  //std::vector<float> hist;
  // pcl::VFHSignature308 hist; 
  pcl::ESFSignature640 hist; 
  // Cloud for storing the object.
	pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
	// Object for storing the normals.
	//pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  // Object for storing the VFH descriptor.
  //pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);
  // Object for storing the ESF descriptor.
	pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);

  float radius_search = 0.03;

  // Object for storing the normals
	//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	// KDtree object to search the normals
	//pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  // VFH estimation object.
	//pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  // ESF estimation object.
	pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;

  // Iterate over all the instance in the folder
  std::string path = "../dataset/pcd_segmentation/";
  std::string token;
  std::vector<std::string> splitter;
  int cont = 0;
  for (const auto & entry : fs::directory_iterator(path))
  {
    //std::cout << entry.path() << std::endl;
    // Save token as a string
    token = entry.path();

    // Read a PCD file from disk.
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(token, *object) != 0)
    {
      return -1;
    }

    // Uncomment to visualize pointcloud with PCL
    //pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    //viewer.showCloud (object);
    //while (!viewer.wasStopped ())
    //{
    //}

    // Estimate the point cloud normals
    //normalEstimation.setInputCloud(object);
	  //normalEstimation.setRadiusSearch(radius_search);
    //normalEstimation.setSearchMethod(kdtree);
	  //normalEstimation.compute(*normals);

    // Set parameters for VFH estimator
    //vfh.setInputCloud(object);
    //vfh.setInputNormals(normals);
    //vfh.setSearchMethod(kdtree);
    // Optionally, we can normalize the bins of the resulting histogram,
    // using the total number of points.
    //vfh.setNormalizeBins(true);
    // Also, we can normalize the SDC with the maximum size found between
    // the centroid and any of the cluster's points.
    //vfh.setNormalizeDistance(false);
    // Compute descriptor
    //vfh.compute(*descriptor);

    // ESF estimation object.
    esf.setInputCloud(object);

    esf.compute(*descriptor);

    hist = descriptor->points[0];

    //std::cout << hist << std::endl;

    // Plotter object. Uncomment this line to see the histogram  
	  //pcl::visualization::PCLHistogramVisualizer viewer_h;
    // We need to set the size of the descriptor beforehand.
	  //viewer_h.addFeatureHistogram(*descriptor, 308);
    //viewer_h.spin();

    // Split the input screen and obtain the instance's name
    boost::split(splitter, token, [](char c){return c == '/' || c == '.';});

    // Writting the vector
    // Object to save descriptor vectors
    std::ofstream outfile;
    outfile.open("../dataset/point_features/" + splitter[5] + ".txt", std::ios_base::app);
    outfile << hist; 
    outfile.close();
    cont ++;
  }

  std::cout << "The total number of instances proccessed were: " << cont << std::endl;

  // This method also saves the vector as pcd, however, open3d is not able to openit
  // pcl::io::savePCDFileASCII ("../segmentation/car_vfh.pcd", *descriptor);

  return 0;

}

