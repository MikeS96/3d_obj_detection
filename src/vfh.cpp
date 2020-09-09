#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/vfh.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <fstream>

int main (int argc, char** argv)
{

  //std::vector<float> hist;
  pcl::VFHSignature308 hist; 
  // Cloud for storing the object.
	pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  // Object for storing the VFH descriptor.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);

  // Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("../segmentation/car.pcd", *object) != 0)
	{
		return -1;
	}

  float radius_search = 0.03;

  // Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(object);
	normalEstimation.setRadiusSearch(radius_search);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

  // VFH estimation object.
	pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(object);
	vfh.setInputNormals(normals);
	vfh.setSearchMethod(kdtree);
	// Optionally, we can normalize the bins of the resulting histogram,
	// using the total number of points.
	vfh.setNormalizeBins(true);
	// Also, we can normalize the SDC with the maximum size found between
	// the centroid and any of the cluster's points.
	vfh.setNormalizeDistance(false);

	vfh.compute(*descriptor);

  hist = descriptor->points[0];

  std::cout << "The descriptor was successfully computed"  << std::endl;

  // Plotter object.
	pcl::visualization::PCLHistogramVisualizer viewer;
  // We need to set the size of the descriptor beforehand.
	viewer.addFeatureHistogram(*descriptor, 308);

  // Uncomment this line to see the histogram   

  //viewer.spin();

  std::ofstream outfile;

  outfile.open("../segmentation/car_vfh.txt", std::ios_base::app);
  outfile << hist; 

  // This method also saves the vector as pcd, however, open3d is not able to openit
  // pcl::io::savePCDFileASCII ("../segmentation/car_vfh.pcd", *descriptor);

  return 0;

}

