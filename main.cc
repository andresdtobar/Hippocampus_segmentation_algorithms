#include "functions/functions.h"

bool comp1(string, string);
template <class T> int Find(const std::vector<T>& vec, T val) {
  for (int x = 0; x != vec.size(); x++) if (vec[x] == val) return x;
}

int main( int argc, char * argv [] )
{
  if( argc < 4 ) {
    std::cerr << "Missing command line arguments" << std::endl;
    std::cerr << "Usage :" << argv[0] << " [root_path database] [test_database] " <<
    "[ranking method] [optimal_number_of_atlases](default 9) [k nearest neighbours](default 2)" <<
    "[patch radius](default 1) [search radius](default 2)" << std::endl;
    return EXIT_FAILURE;
  }
  string root_path = argv[1];
  string test_database = argv[2];
  string ranking_method = argv[3];
  path sp(root_path);
  Vec tmpVec;

  // Paramenter settings
  int optnum_of_trimgs;
  if (argc > 4) {optnum_of_trimgs = atoi(argv[4]);}
  else {optnum_of_trimgs = 9;}
  float knn;
  if (argc > 5) {knn = atof(argv[5]);}
  else {knn = 11;}
  uint patch_radius;
  if (argc > 6) {patch_radius = atof(argv[6]);}
  else {patch_radius = 1;}
  uint neigh_radius;
  if (argc > 7) {neigh_radius = atof(argv[7]);}
  else {neigh_radius = 2;}

  float agrth = 1;

  // Sort and allocate the databases image filenames
  std::vector<string> sv;
  for (auto x : directory_iterator(sp))
    sv.push_back(x.path().string());
  std::sort(sv.begin(), sv.end(),comp1);

  //Select the images which belong to the selected database
  std::vector<string> trv, tsv;
  for (auto x : sv) {
    if (x.find(test_database) != string::npos) tsv.push_back(x);
    if (x.find(test_database) != string::npos) trv.push_back(x);
  }

  // Some initializations
  ImageType::Pointer test_img = ImageType::New();
  ImageType::Pointer manual_seg = ImageType::New();
  vector<string> fsplit_str,msplit_str;
  std::vector<float> dice_index_vec, jac_index_vec, recall_index_vec, pre_index_vec;
  std::vector<std::vector<float>> dice_index_mat;
  std::vector<ImageType::Pointer> training_labels;
  std::vector<ImageType::Pointer> training_images;
  std::vector<ImageType::Pointer> training_labels_ranked;
  std::vector<ImageType::Pointer> training_images_ranked;

  std::vector<float> acc_vec;
  std::vector<float> thres_vec(tsv.size());

  // Setting the optimal decision threshold by each image

  if (test_database == "SATA")
    thres_vec = {0.4147, 0.418, 0.419, 0.4177, 0.4138, 0.25, 0.329, 0.4078, 0.4074,
      0.416, 0.412, 0.4947, 0.4197, 0.336, 0.416, 0.3347, 0.414, 0.334, 0.4158,
      0.462, 0.33, 0.4169, 0.33, 0.4938, 0.3356, 0.495, 0.3327, 0.4157, 0.4146,
      0.408, 0.33, 0.33, 0.2746, 0.33, 0.49};


  if (test_database == "LONI")
    thres_vec = {0.41, 0.46, 0.468, 0.53, 0.47, 0.41, 0.469, 0.356, 0.41, 0.47,
      0.35, 0.468, 0.41, 0.3548, 0.4118, 0.468, 0.53, 0.47, 0.415, 0.47, 0.47,
      0.469, 0.298, 0.41, 0.41, 0.4137, 0.43, 0.5197, 0.525, 0.35, 0.41, 0.35,
      0.41, 0.29, 0.35, 0.4, 0.41, 0.3, 0.35, 0.3579};

  if (test_database == "ADNI")
    thres_vec = {0.429, 0.428, 0.43, 0.43, 0.287, 0.42, 0.288, 0.428, 0.4267,
      0.2879, 0.4256, 0.434, 0.4296, 0.43, 0.55, 0.2797, 0.42, 0.433, 0.42, 0.423,
      0.562, 0.429, 0.42, 0.416, 0.425, 0.428, 0.275, 0.565, 0.4266, 0.429, 0.43,
      0.297, 0.432, 0.438, 0.43, 0.437, 0.426, 0.428, 0.43, 0.426, 0.427, 0.425,
      0.423, 0.429, 0.286, 0.29, 0.42, 0.4258, 0.4259, 0.43, 0.428, 0.429, 0.426,
      0.429, 0.423, 0.428, 0.298, 0.429, 0.42, 0.428, 0.2868, 0.425, 0.424, 0.39,
      0.426, 0.43, 0.4276, 0.425, 0.429, 0.426, 0.427, 0.425, 0.43, 0.29, 0.41,
      0.4236, 0.4235, 0.424, 0.434, 0.432, 0.43, 0.29, 0.414, 0.4248, 0.4247, 0.567,
      0.428, 0.5758, 0.425, 0.433, 0.426, 0.436, 0.414, 0.416, 0.42, 0.426, 0.428,
      0.4286, 0.43};

  // Iterate over the test image set
  for (auto x : tsv) {

    std::cout << x << '\n';
    vector<string> fsplit_str,split_str;

    // Set the output dir and filename
    stringstream output_dir;
    output_dir << x << "/mri/estimated_seg";

    // Verify if the output dir is already created, otherwise create it
    if (!exists(output_dir.str())) {
      stringstream mkoutput_dir;
      mkoutput_dir << "mkdir " << output_dir.str();
      system(mkoutput_dir.str().c_str());
    }

    // Set the output filename
    stringstream output_filename;
    output_filename << output_dir.str() << "/brainhc_est_seg.nii.gz";

    // Read and allocate the manual segmentation image
    stringstream tslb_image_name;
    tslb_image_name << x << "/mri/manual_seg/brainhc_seg.nii.gz";
    ReaderType::Pointer readerl = ReaderType::New();
    readerl->SetFileName(tslb_image_name.str());readerl->Update();
    manual_seg = readerl->GetOutput();

    // Find which labels have the manual segmentation image
    std::vector<int> lbls;std::vector<int> hist;
    Unique(manual_seg,manual_seg->GetLargestPossibleRegion(),lbls,hist);
    sort(lbls.begin(),lbls.end());

    // Initialize the output image (the segmentation result)
    ImageType::Pointer test_lbl = ImageType::New();
    InitializeImage(manual_seg,test_lbl);
    CreateImage(test_lbl,test_lbl->GetRequestedRegion(),0.0);

    // Verify whether the estimated segmentation is already done
    // if (exists(output_filename.str())) {
    //   ReaderType::Pointer oreader = ReaderType::New();
    //   oreader->SetFileName(output_filename.str());oreader->Update();
    //   test_lbl = oreader->GetOutput();
    //   float dice = ComputeDice(manual_seg, test_lbl, lbls);
    //   dice_index_vec.push_back(dice);
    //   std::cout << "dice index: " << dice << '\n';
    //   continue;
    // }

    // Read and allocate the test intensity image
    stringstream ts_image_name;
    ts_image_name << x << "/mri/brain.nii.gz";
    ReaderType::Pointer readeri = ReaderType::New();
    readeri->SetFileName(ts_image_name.str());readeri->Update();
    test_img = readeri->GetOutput();
    RescaleImage(test_img,0,255);
    // Following line is optional dependind on the amount of noise still present in the image
    // MedianImage(test_img,1);

    // Set the warped images directory
    stringstream reg_images_path;
    reg_images_path << x << "/mri/images_reg_hc/";
    stringstream reg_imagesROI_path;
    reg_imagesROI_path << x << "/mri/imagesROI_reg/";

    // Verify if the warped images dir is already created, otherwise create it
    if (!exists(reg_images_path.str())) {
      stringstream mkreg_images_path;
      mkreg_images_path << "mkdir " << reg_images_path.str();
      system(mkreg_images_path.str().c_str());
    }

    // Set the mapping directory
    stringstream mapping_path;
    mapping_path << x << "/mri/affine_mapping/";

    // Extract the index of the current test image
    split_str.clear(); fsplit_str.clear();
    boost::split(fsplit_str,x,boost::is_any_of("/"));
    boost::split(split_str,fsplit_str.back(),boost::is_any_of("_"));
    string current_sub = split_str.back();
    uint subjidx = atoi(current_sub.c_str());

    // Read warped training atlas
    training_labels.clear(); training_images.clear();
    for (auto y : trv) {
      if (y == x) continue;
      stringstream tmpstr, ifilename, lfilename;
      msplit_str.clear();
      boost::split(msplit_str,y,boost::is_any_of("/"));
      tmpstr << fsplit_str.back() << "_" << msplit_str.back();

      // Read and allocate the warped training intensity images
      ifilename << reg_imagesROI_path.str() << tmpstr.str() << ".nii.gz";
      ReaderType::Pointer ireader = ReaderType::New();
      ireader->SetFileName(ifilename.str());
      ireader->Update();
      training_images.push_back(ireader->GetOutput());

      // Read and allocate the warped training label images
      lfilename << reg_imagesROI_path.str() << tmpstr.str() << "_seg.nii.gz";
      ReaderType::Pointer lreader = ReaderType::New();
      lreader->SetFileName(lfilename.str());
      lreader->Update();
      training_labels.push_back(lreader->GetOutput());
    }

    // Compute similarity metric between the test intensity image and the
    // warped training atlases if it was not computed previusly

    std::vector<float> W_vec;
    stringstream W_filename;

    if (ranking_method == "NMIh")
      W_filename << x << "/NMIh_" << test_database << ".txt";
    if (ranking_method == "NMI")
      W_filename << x << "/NMI_" << test_database << ".txt";

    if (!exists(W_filename.str())) {

      // Compute ROI in such images
      ImageType::RegionType ROI_regionf;
      GetROIs(training_labels,ROI_regionf,x);
      if (ranking_method == "NMIh") {
        W_vec.clear();
        ComputeMI(test_img, training_labels, ROI_regionf,W_vec);
      }
      if (ranking_method == "NMI") {
        W_vec.clear();
        ComputeMI(test_img, training_labels, ROI_regionf,W_vec);
      }
      SaveVector(W_filename.str(),W_vec);
    }
    else ReadVector(W_filename.str(),W_vec);
    std::cout << "training set size: " << W_vec.size() << '\n';

    std::cout << "Similarity vector loaded!" << '\n';

    // Sort the similarities vector and save the indices
    Vec Wvec = Map<Vec>(W_vec.data(),W_vec.size());
    VectorXi indVec = argsort(Wvec,"descend");
    tmpVec = Wvec.head(optnum_of_trimgs);
    Wvec = tmpVec;

    // Store the optimum number of most similar atlases
    training_images_ranked.clear();
    training_labels_ranked.clear();
    for (int i = 0; i != optnum_of_trimgs; i++) {
      training_images_ranked.push_back(training_images[indVec(i)]);
      training_labels_ranked.push_back(training_labels[indVec(i)]);
    }

    std::cout << "optimal training set size: " << optnum_of_trimgs << '\n';

    // Rescale the intensity values inside each images
    uint it = 0;
    for ( auto & img : training_images_ranked) {
      RescaleImage(img,0,255);
      // MedianImage(img,1);
      training_images_ranked[it] = img;
      it++;
    }

    // Get ROIs for all atlases
    ImageType::RegionType ROI_region_rank;
    GetROIs(training_labels_ranked,ROI_region_rank,x);
    std::cout << "ROI region computed!" << '\n';


    float threshold = thres_vec[subjidx-1];
    MultiScaleKnnNonLocalPatchSegmentation(test_img, training_images_ranked,training_labels_ranked,
      ROI_region_rank,patch_radius,neigh_radius,1,Wvec, test_database,current_sub,
      manual_seg,test_lbl,acc_vec, ranking_method, lbls, knn, threshold);

    // Uncoment to compute the segmentation based on precalculated label files

    // LoadPrecomputedSegmentation(training_labels_ranked, ROI_region_rank,
    //   test_database, current_sub, 1, test_lbl,lbls,manual_seg,acc_vec,threshold);

    //Save the resulting segmentation
    MedianImage(test_lbl,1);
    Save3DImage(test_lbl,output_filename.str());

    // Validation stage
    // float dice = Validation(output_filename.str(),manual_seg, lbls);
    float dice = ComputeDice(manual_seg, test_lbl, lbls);
    float jac = ComputeJaccard(manual_seg, test_lbl, lbls);
    float recall = ComputeRecall(manual_seg, test_lbl, lbls);
    float pre = ComputePrecision(manual_seg, test_lbl, lbls);
    std::cout << "dice index: " << dice << '\n';

    dice_index_vec.push_back(dice);
    jac_index_vec.push_back(jac);
    recall_index_vec.push_back(recall);
    pre_index_vec.push_back(pre);
  }
  stringstream dice_filename;
  dice_filename << test_database << "_" << ranking_method << "_" << optnum_of_trimgs <<
   "_" << knn << "_" << patch_radius << "_" << neigh_radius << "_dice.txt";
  SaveVector(dice_filename.str(),dice_index_vec);

  stringstream jac_filename;
  jac_filename << test_database << "_" << ranking_method << "_" << optnum_of_trimgs <<
   "_" << knn << "_" << patch_radius << "_" << neigh_radius << "_jaccard.txt";
  SaveVector(jac_filename.str(),jac_index_vec);

  stringstream recall_filename;
  recall_filename << test_database << "_" << ranking_method << "_" << optnum_of_trimgs <<
   "_" << knn << "_" << patch_radius << "_" << neigh_radius << "_recall.txt";
  SaveVector(recall_filename.str(),recall_index_vec);

  stringstream pre_filename;
  pre_filename << test_database << "_" << ranking_method << "_" << optnum_of_trimgs <<
   "_" << knn << "_" << patch_radius << "_" << neigh_radius << "_precision.txt";
  SaveVector(pre_filename.str(),pre_index_vec);

  stringstream acc_filename;
  acc_filename << test_database << "_" << ranking_method << "_" << optnum_of_trimgs <<
   "_" << knn << "_" << patch_radius << "_" << neigh_radius << "_accuracy.txt";
  SaveVector(acc_filename.str(),acc_vec);

  return 0;
}

bool comp1(string a, string b) {

  /*****************************************************************************
  This functions compare the number contained at the end of the name of
   certain files in order to sort them descendently.
  *****************************************************************************/

  vector<string> fsplit_str,msplit_str,split_str;
  boost::split(split_str,a,boost::is_any_of("/"));
  boost::split(fsplit_str,split_str.back(),boost::is_any_of("_"));
  split_str.clear();
  boost::split(split_str,b,boost::is_any_of("/"));
  boost::split(msplit_str,split_str.back(),boost::is_any_of("_"));

  return atoi(fsplit_str.back().c_str()) < atoi(msplit_str.back().c_str());
}

template <class T>
void Unique(const ImageType::Pointer& image,const ImageType::RegionType& image_region,
 std::vector <T>& values, std::vector<int>& hist)
/*------------------------------------------------------------------------------
Find how many values are diferents withing a image.
--------------------------------------------------------------------------------
Input parameters:
  image: The image in which we want to know the unique values.
Output parameters:
  values: The set of unique values od the image.
------------------------------------------------------------------------------*/
{
  ConstIteratorTypeIndex imageIt(image,image_region);
  int temp;
  values.clear();hist.clear();
  values.push_back(imageIt.Get());
  hist.push_back(1);
  while (!imageIt.IsAtEnd()){
    temp = 0;
    for (uint k = 0; k != values.size(); k++) if (imageIt.Get() != values[k]) temp++;
    else hist[k] += 1;
    if (temp == values.size()) {values.push_back(imageIt.Get()); hist.push_back(1);}
    ++imageIt;
  }
}

template <class T>
void SaveVector(std::string filename,const std::vector<T>& vec) {

  std::ofstream output_file(filename);
  uint n = vec.size();
  for (int k = 0; k != n; k++) if (k != n-1) output_file << vec[k] << " ";else output_file << vec[k];
  output_file << "\n";

  output_file.close();
}

template <class T>
void ReadVector(std::string filename, std::vector<T>& vec) {
  std::ifstream file(filename);
  float val;
  while (file >> val) vec.push_back(val);
  file.close();
}
