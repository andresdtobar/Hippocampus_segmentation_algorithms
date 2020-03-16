#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define MAXBUFSIZE  ((int) 1e6)

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <unordered_map>
#include <random>

// Dlib includes
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/statistics.h>

// MLpack includes
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/methods/lars/lars.hpp"
#include "mlpack/methods/linear_regression/linear_regression.hpp"
#include "mlpack/methods/linear_svm/linear_svm.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/simple_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "mlpack/core/cv/metrics/mse.hpp"
#include "mlpack/core/cv/metrics/f1.hpp"
#include "mlpack/core/hpt/fixed.hpp"
#include "mlpack/core/hpt/deduce_hp_types.hpp"
#include "mlpack/core/hpt/cv_function.hpp"
#include "mlpack/core/hpt/hpt.hpp"
#include "mlpack/methods/pca/pca.hpp"
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>

// Eigen includes
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <unsupported/Eigen/MatrixFunctions>

// ITK includes
#include "itkImage.h"
#include "itkImageBase.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkNeighborhoodIterator.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkConstantBoundaryCondition.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkNeighborhoodAllocator.h"
#include "itkNormalizeImageFilter.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapToLabelImageFilter.h"
#include "itkMirrorPadImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkStatisticsImageFilter.h"
#include "itkGaussianDistribution.h"
#include "itkJoinImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkFlipImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkScaleTransform.h"
#include "itkIdentityTransform.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkConstSliceIterator.h"
#include "itkSliceIterator.h"
#include "itkOrImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include "itkSobelEdgeDetectionImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"

using namespace std;
using namespace boost::filesystem;
using namespace Eigen;
using Mat = MatrixXf;
using SpMat = SparseMatrix<float>;
using SprMat = SparseMatrix<float,RowMajor>;
using Vec = VectorXf;
using DiagMat = DiagonalMatrix<float,Dynamic>;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;
using namespace mlpack::hpt;
using namespace mlpack::regression;
using namespace mlpack::svm;
using namespace mlpack::pca;


typedef itk::Image< float, 3 > ImageType;
typedef itk::Image<unsigned char, 3>  CharImageType;
typedef itk::Image< float, 2 > SliceType;
typedef itk::Image< unsigned char, 2 > CharSliceType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileReader< CharImageType > LabelReaderType;
typedef itk::ImageFileWriter< ImageType  > WriterType;
typedef itk::ImageFileWriter< CharImageType  > WriterTypeChar;
typedef itk::ImageFileReader< SliceType > SliceReaderType;
typedef itk::ExtractImageFilter< ImageType, SliceType > SliceFilterType;
typedef itk::ExtractImageFilter< CharImageType, CharSliceType > CharSliceFilterType;
typedef itk::ExtractImageFilter< ImageType, ImageType > FilterType;
typedef itk::MinimumMaximumImageCalculator <SliceType> SliceCalculatorFilterType;
typedef itk::MinimumMaximumImageCalculator <CharSliceType> CharSliceCalculatorFilterType;
typedef itk::MinimumMaximumImageCalculator <ImageType>ImageCalculatorFilterType;
typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
typedef itk::ImageRegionIterator< ImageType> IteratorType;
typedef itk::ImageRegionConstIterator< ImageType > ConstIteratorType;
typedef itk::ImageRegionIteratorWithIndex< ImageType > IteratorTypeIndex;
typedef itk::ImageRegionIteratorWithIndex< CharImageType > IteratorTypeIndexChar;
typedef itk::ImageRegionConstIteratorWithIndex< ImageType> ConstIteratorTypeIndex;
typedef itk::ConstantBoundaryCondition<ImageType> BoundaryConditionType;
typedef itk::ConstNeighborhoodIterator< ImageType,BoundaryConditionType > ConstNeighborhoodIteratorType;
typedef itk::NeighborhoodIterator< ImageType,BoundaryConditionType > NeighborhoodIteratorType;
typedef itk::NeighborhoodIterator< ImageType,BoundaryConditionType > NoConstNeighborhoodIteratorType;
typedef itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< ImageType > FaceCalculatorType;
typedef itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>
              StructuringElementType;
typedef itk::BinaryMorphologicalClosingImageFilter <ImageType, ImageType, StructuringElementType>
          BinaryMorphologicalClosingImageFilterType;
typedef itk::BinaryErodeImageFilter <ImageType, ImageType, StructuringElementType>
    BinaryErodeImageFilterType;
typedef itk::BinaryDilateImageFilter <ImageType, ImageType, StructuringElementType>
BinaryDilateImageFilterType;
typedef itk::NormalizeImageFilter< ImageType, ImageType > NormalizeFilterType;
typedef itk::MirrorPadImageFilter <ImageType, ImageType>MirrorPadImageFilterType;
typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
// typedef itk::Statistics::GaussianDistribution NormalPDF;
typedef itk::ScaleTransform<double, 3> TransformType;
typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
typedef itk::BSplineInterpolateImageFunction<ImageType, double, double> BSplineInterpolatorType;
typedef itk::DiscreteGaussianImageFilter< ImageType, ImageType >  SmoothFilterType;
typedef ImageType::PixelType Pixel3D;
typedef ImageType::IndexType Index3D;
typedef ImageType::SizeType Size3D;
typedef NeighborhoodIteratorType::NeighborhoodType::OffsetType Offset3D;
typedef itk::GradientMagnitudeImageFilter<  ImageType, ImageType >  gradientFilterType;
typedef itk::ImageSliceConstIteratorWithIndex < ImageType > SliceIteratorType;
typedef itk::Neighborhood<float,3> NeighborhoodType;
typedef itk::OrImageFilter <CharImageType> OrImageFilterType;
typedef itk::CastImageFilter< ImageType, CharImageType > Float2CharFilterType;
typedef itk::ConnectedComponentImageFilter <CharImageType, CharImageType >
  ConnectedComponentImageFilterType;
typedef itk::LabelShapeKeepNObjectsImageFilter< CharImageType >
  LabelShapeKeepNObjectsImageFilterType;
typedef std::pair<int, double> argsort_pair;

using JoinFilterType = itk::JoinImageFilter< ImageType, ImageType >;
using VectorImageType = JoinFilterType::OutputImageType;
using HistogramFilterType = itk::Statistics::ImageToHistogramFilter<VectorImageType>;
using HistogramSizeType = HistogramFilterType::HistogramSizeType;
using HistogramMeasurementVectorType = HistogramFilterType::HistogramMeasurementVectorType;
using HistogramType = HistogramFilterType::HistogramType;
using SobelFilterType = itk::SobelEdgeDetectionImageFilter<ImageType, ImageType>;
using BiasCorrectionFilterType = itk::N4BiasFieldCorrectionImageFilter<ImageType,CharImageType,ImageType>;

template<class T> int GaussLabelAvg(T, uint, uint, uint);
void CreateImage(ImageType::Pointer&,const ImageType::RegionType&,int);
void GetMaxMinOfSlice(const CharImageType:: Pointer&,CharSliceCalculatorFilterType::Pointer,
int,int, float&, float&);
void ExtractLargestObject(CharImageType::Pointer& img, uint);
void GetSlice(const ImageType::Pointer&,int,int,SliceType::Pointer&);
void GetCharSlice(const CharImageType::Pointer&,int,int,CharSliceType::Pointer&);
void InitializeImage(const ImageType::Pointer&, ImageType::Pointer&);
Vec MixPatchValsV2(const std::vector<Index3D>&, const std::vector
  <NeighborhoodIteratorType::NeighborhoodType>&, const std::vector<Index3D>&,
   const Mat&);
void RescaleImage(ImageType::Pointer& ,int,int);
void Save3DImage(const ImageType::Pointer&, std::string);
void Save3DImageLabel(const CharImageType::Pointer&, string);
void GetImgROI(const CharImageType::Pointer&,Index3D&,Size3D&);
template <class T> void SaveMatrix(std::string, const std::vector<std::vector<T>>&);
template <class T> void SaveVector(std::string filename,const std::vector<T>&);
void ComputeDifference(Index3D,Index3D,Index3D&);
void ComputeMI(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
              const ImageType::RegionType&, std::vector<float>&);
void GetROIs(const std::vector<ImageType::Pointer>&, ImageType::RegionType&, string);
template <class T> void ReadVector(std::string, std::vector<T>&);
void ReadMatrix(std::string, std::vector<std::vector<float>>&);
void WMV(const std::vector<ImageType::Pointer>&,const Vec&,std::vector<int>,
         float,ImageType::Pointer&);
void LWMV(std::vector<ImageType::Pointer>,std::vector<ImageType::Pointer>,
 std::vector<float>,float, ImageType::Pointer&);
void BinaryClosing(ImageType::Pointer&, int);
// void GaussianKernel(const std::vector<float>&, const std::vector<float>&,
//                     std::vector <std::valarray<float>>&);
template <class T> float computePatchSimilarity(const T&, const T&, uint);
template <class T> void Sum2Vecs(const T&, const T&, int, std::vector<float>&);
void ImagePyramid(const ImageType::Pointer&,uint,std::vector<ImageType::Pointer>&,float=0.0);
std::vector <float> GetMaxMinOfImage(const ImageType:: Pointer&);
std::vector<Index3D> GetPositionVec(int);
std::vector<Index3D> GetOffsetVec(int);
template <class T> std::vector<float> Mode(T,int);
template <class T> std::vector <T> Max(std::vector<T>);
template <class T> std::vector <T> Min(std::vector<T>);
std::vector<float> Linspace(float i, float f, int N);
template <class T> float Mean(const T&,int);
inline float NormalPDF(float, float, float);

float ComputePatchSSM(const ConstNeighborhoodIteratorType::NeighborhoodType&,
                const ConstNeighborhoodIteratorType::NeighborhoodType&,int);

float ComputeDice(const ImageType::Pointer&,const ImageType::Pointer&,
                  const std::vector<int>&);

float ComputeJaccard(const ImageType::Pointer&,const ImageType::Pointer&,
                  const std::vector<int>&);

float ComputeRecall(const ImageType::Pointer&,const ImageType::Pointer&,
                  const std::vector<int>&);

float ComputePrecision(const ImageType::Pointer&,const ImageType::Pointer&,
                  const std::vector<int>&);

bool CompareIndex(Index3D, Index3D );
template <class T>
void Unique(const ImageType::Pointer&,const ImageType::RegionType&,
           std::vector <T>&,std::vector<int>&);
Index3D SumIndex(const Index3D&, const Index3D&);
template <class T> T SumVector(const std::vector<T>&);
template <class T> float DotProd(const T&,const T&, uint);
template <class T> int Find (const std::vector<T>&, T);
template <class T> void SortByIndex(std::vector<T>&, std::vector<int>);
template <class T> T Median(const std::vector<T>&);
template <class T>
void HistVec(const T&,uint, std::vector<float>&, std::vector<int>&);
template <class T> bool isZero (T);
template <class T> float Sum(const T, uint);

void MultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&,uint,float=.5);

void MultiScaleSVMNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&,uint=10,float=0.01, float=.1);

void BayesianMultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&,uint);

void CCAMultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void CCAKnnNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void KnnNonLocalPatchSegmentation(ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&,uint,float=.5);

void MultithreadBalanceMultiScaleKnnNonLocalPatchSegmentation(const ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void BalanceMultiScaleSvmNonLocalPatchSegmentation(const ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void BalanceMultiScaleRFNonLocalPatchSegmentation(const ImageType::Pointer&,
  const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void PythonWarpClassification(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void MultiScalePatchMatrixBuilding(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

void NonLocalPatchSegmentation(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
    const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
    const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
    std::vector<float>&, string, const std::vector<int>&);

void LocalPatchSegmentation(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
    const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
    const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
    std::vector<float>&, string, const std::vector<int>&);

void MPNonLocalPatchSegmentation(const ImageType::Pointer&, const std::vector<ImageType::Pointer>&,
  const std::vector<ImageType::Pointer>&, ImageType::RegionType, uint, uint,float,
  const Vec&,string,string, const ImageType::Pointer&, ImageType::Pointer&,
  std::vector<float>&, string, const std::vector<int>&);

float IntPow(float, float);
void SaveEigenMatrix(std::string, const Mat&);
void SaveEigenVector(std::string, const VectorXi&);
Mat ReadEigenMatrix(string filename);
bool comp (float,float);
std::vector<int> BalanceClassSamples(Mat&,VectorXi&,const Mat&, int, int, string,
  size_t=30,uint=0);
template <class T> float GetVariance(const T&, int, float=0);
template <class T> float GetMean(const T&, int);
void RemoveRow(Mat&, uint);
void RemoveColumn(Mat&, uint);
std::vector<float> MatEuDist(const Mat&);
void GetGaussianKernelMatrix(const Mat&, Mat&,string="median");
void GetLabelKernelMatrix(const Vec&, Mat&);
Vec ExtractPatchFeatures(const NeighborhoodIteratorType::NeighborhoodType&,
  uint,uint);
Vec ExtractPatchFeaturesV2(const NeighborhoodIteratorType::NeighborhoodType&,
  const NeighborhoodIteratorType::NeighborhoodType&,uint,uint);
void GetSamplingScaleVec(Size3D, uint,std::vector<std::vector<Index3D>>&);
template <class T> float MutualInformation(const T&, const T&, int);
std::vector<float> ComputeMaxVarSigmaV2 (const std::vector<float>&);
bool argsort_comp(const argsort_pair&, const argsort_pair&);
VectorXi argsort(Vec&, string="ascend");
Mat k_means(const Mat&,std::vector<std::vector<Eigen::ArrayXf>>&,
  std::vector<std::vector<uint>>&, uint16_t, size_t);
void LoadPrecomputedSegmentation(const std::vector<ImageType::Pointer>&,
  ImageType::RegionType, string, string, float, ImageType::Pointer&,
  const vector<int>&, const ImageType::Pointer&, std::vector<float>&,float=.5);
template<class T> std::vector<int> MostRepeatedVal(const T&, int,const std::vector<int>&);
template<class T> float NormCosineSim(const T&, const T&, uint);
template<class T> float TanimotoSim(const T&, const T&, uint);
void ComputeHistogram(const std::vector<float>&, const Vec&, std::vector<uint>&);
Mat ArmaToEigen(const arma::mat&);
arma::mat EigenToArma(const Eigen::MatrixXd&);
Mat SignalConv(const Mat&, const Mat&);
template <class T>
float SquareEuclideanDistance(T, T,uint);
float Similarity(float un, float vn, float up, float vp);
template<class T> float InvDiffSim(const T&, const T&, uint,string="euclidean");
void MedianImage(ImageType::Pointer&, uint);
void BiasCorrection(ImageType::Pointer&, const CharImageType::Pointer& mask);

#endif // MRIFUNCTIONS_H
