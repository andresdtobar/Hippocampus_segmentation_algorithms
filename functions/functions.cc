#include "functions.h"

bool comp (float i,float j) { return (i>j);} // To sort a vector descendently

template<class T> float NormCosineSim(const T& array1, const T& array2, uint numel) {
  float num = 0.0;
  for (uint i = 0; i != numel; i++) num += array1[i]*array2[i];
  float norm1 = 0.0;
  for (uint i = 0; i != numel; i++) norm1 += array1[i]*array1[i];
  float norm2 = 0.0;
  for (uint i = 0; i != numel; i++) norm2 += array2[i]*array2[i];

  float sim = .5*(num/(float)sqrt(norm1*norm2)+1);

  return sim;
}

template<class T> float TanimotoSim(const T& array1, const T& array2, uint numel) {
  float num = 0.0;
  for (uint i = 0; i != numel; i++) num += array1[i]*array2[i];
  float norm1 = 0.0;
  for (uint i = 0; i != numel; i++) norm1 += array1[i]*array1[i];
  float norm2 = 0.0;
  for (uint i = 0; i != numel; i++) norm2 += array2[i]*array2[i];

  float sim = num/(float)(norm1 + norm2 - num);

  return sim;
}

template<class T> float InvDiffSim(const T& array1, const T& array2, uint numel,
  string dist_type) {

  float diff, sim;
  if (dist_type == "manhattan") {
    diff = 0.0;
    for (uint i = 0; i != numel; i++) diff += abs(array1[i] - array2[i]);
  }
  if (dist_type == "euclidean") {
    diff = 0.0;
    for (uint i = 0; i != numel; i++) diff += pow(array1[i] - array2[i],2);
    diff = sqrt(diff);
  }
  sim = 1.0/(1.0 + diff);

  return sim;
}

void LocalPatchSegmentation(const ImageType::Pointer& timg, const std::vector<ImageType::Pointer>&
  itrain_set,const std::vector<ImageType::Pointer>& ltrain_set, ImageType::RegionType ROI_reg,
  uint patch_radius,uint neigh_radius, float agr_th, const Vec& Wvec, string database,
  string current_sub,const ImageType::Pointer& manual_seg,ImageType::Pointer& test_lbl,
  std::vector<float>& acc_vec, string ranking_method, const std::vector<int>& labels) {

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, aMat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Compute prior agreement
  // patch = neighIt.GetNeighborhood();
  // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  // Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  Xt.resize(uncertain_inds.size(),patch_vol);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        sort_uncertain_inds[it] = index;
        patch = neighIt.GetNeighborhood();
        for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        // Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    ssmth = 0;
    // Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    Xn.resize(numOfTrainSamp, patch_vol);
    Ln.resize(numOfTrainSamp);
    n_samples = {0,0};
    it = 0;

    // Construc the feature matrix with the most similar patches
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.SetLocation(index);
      lneighIt.SetLocation(index);
      patch1 = neighIt1.GetNeighborhood();
      lpatch = lneighIt.GetNeighborhood();
      tr_label = lpatch.GetCenterValue();
      if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
      else {n_samples[1]++;Ln(it) = 1;}
      for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
      // featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
      // Xn(it,seqN(0,featVec.size())) = featVec;
      it++;
    }

    if (it == 0) {
      flbls(t) = 0;
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    // dists.minCoeff(&ksig);
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();
    flbls(t) = (float)(ksim.transpose()*Ln.cast<float>())/ksim.sum();
    flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  // stringstream stm_str;
  // stm_str << "estimated_labels_"<< database << "_" << current_sub;
  // SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void NonLocalPatchSegmentation(const ImageType::Pointer& timg, const std::vector<ImageType::Pointer>&
  itrain_set,const std::vector<ImageType::Pointer>& ltrain_set, ImageType::RegionType ROI_reg,
  uint patch_radius,uint neigh_radius, float agr_th, const Vec& Wvec, string database,
  string current_sub,const ImageType::Pointer& manual_seg,ImageType::Pointer& test_lbl,
  std::vector<float>& acc_vec, string ranking_method, const std::vector<int>& labels) {

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, aMat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Compute prior agreement
  // neighIt.SetLocation(uncertain_inds[0]);
  // patch = neighIt.GetNeighborhood();
  // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  // Xt.resize(uncertain_inds.size(),featVec.size());
  Xt.resize(uncertain_inds.size(),patch_vol);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        sort_uncertain_inds[it] = index;
        patch = neighIt.GetNeighborhood();
        for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        // Xt(it,all) = featVec;
        it++;
      }
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl, Knn;
  std::vector<int> n_samples(2);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, gW, sW;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    // Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size());
    Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    // Construc the feature matrix with the most similar patches
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        lpatch = lneighIt.GetNeighborhood();
        tr_label = lpatch.GetCenterValue();
        if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
        else {n_samples[1]++;Ln(it) = 1;}
        for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
        // featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
        // Xn(it,all) = featVec;
        it++;
      }
    }


    if (it == 0) {
      flbls(t) = 0;
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();
    flbls(t) = (float)(ksim.transpose()* Ln.cast<float>())/ksim.sum();

    if (isnan(flbls(t))) {
      if (!flag) {flag = true;goto feature_matrix_building;}
      else flbls(t) = 0;
    }
    flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  // stringstream stm_str;
  // stm_str << "estimated_labels_"<< database << "_" << current_sub;
  // SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void MPNonLocalPatchSegmentation(const ImageType::Pointer& timg, const std::vector<ImageType::Pointer>&
  itrain_set,const std::vector<ImageType::Pointer>& ltrain_set, ImageType::RegionType ROI_reg,
  uint patch_radius,uint neigh_radius, float agr_th, const Vec& Wvec, string database,
  string current_sub,const ImageType::Pointer& manual_seg,ImageType::Pointer& test_lbl,
  std::vector<float>& acc_vec, string ranking_method, const std::vector<int>& labels) {

  float ssmth,sig, ksig, sumw, acc, est_th, tmp_est;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  std::vector<int> agr, tr_lbls;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec;
  if (neigh_radius > 0) offsetvec = GetOffsetVec(neigh_radius);
  else {
    index = {0,0,0};
    offsetvec.push_back(index);
  }

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Compute prior agreement
  // patch = neighIt.GetNeighborhood();
  // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  // Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  Xt.resize(uncertain_inds.size(),patch_vol);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        sort_uncertain_inds[it] = index;
        patch = neighIt.GetNeighborhood();
        for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        // Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, knnEstVec(5);
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  std::vector<ConstNeighborhoodIteratorType::NeighborhoodType> est_patch_label_vec(np);
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    // Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Lnp.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    // Construc the feature matrix with the most similar patches
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();patchSelectVec.clear();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        lpatch = lneighIt.GetNeighborhood();
        tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = -1;}
          else {n_samples[1]++;Ln(it) = 1;}
          for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          for (uint i = 0; i != patch_vol; i++) Lnp(it,i) = lpatch[i];
          // featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
          // Xn(it,seqN(0,featVec.size())) = featVec;
          it++;
      }
    }

    if (it == 0) {
      flbls(t) = -1;
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = -1;
      }
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    Xn.conservativeResize(it, NoChange);
    Lnp.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();

    //Multi-point labeling
    // multiLabelVec = (Lnp.array()*ksim.replicate(1,patch_vol).array()).colwise().sum()/ksim.sum();

    multiLabelVec = Vec::Zero(patch_vol);
    for (size_t p = 0; p != Ln.size(); p++)
      multiLabelVec.array() += (Lnp.row(p).array()*ksim(p));
    multiLabelVec.array() /= ksim.sum();

    for (auto &l : multiLabelVec) if (isnan(l)) l = 0;
    ConstNeighborhoodIteratorType::NeighborhoodType estlpatch;
    estlpatch.SetRadius(patch_radius);
    for (i = 0; i != patch_vol; i++) estlpatch[i] = multiLabelVec(i);
    est_patch_label_vec[t] = estlpatch;
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    t++;
  }

  // Compute the input kernel between test samples
  Mat testKer = Mat::Zero(Xt.rows(),Xt.rows());
  GetGaussianKernelMatrix(Xt,testKer,"median");
  // Get the patch offset vec
  std::vector<Index3D> patch_offsetvec = GetOffsetVec(patch_radius);
  flbls = MixPatchValsV2(sort_uncertain_inds,est_patch_label_vec,patch_offsetvec,testKer);
  for (uint ind = 0; ind != np; ind++) flbls(ind) < 0.5 ? flbls_th(ind) = 0 : flbls_th(ind) = 1;
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  // stringstream stm_str;
  // stm_str << "estimated_labels_"<< database << "_" << current_sub;
  // SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }

  std::cout << "Segmentation done!" << '\n';
}

void KnnNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels, uint Knn, float th) {

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, aMat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Compute prior agreement
  // neighIt.SetLocation(uncertain_inds[0]);
  // patch = neighIt.GetNeighborhood();
  // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  // Xt.resize(uncertain_inds.size(),featVec.size());
  Xt.resize(uncertain_inds.size(),patch_vol);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        sort_uncertain_inds[it] = index;
        patch = neighIt.GetNeighborhood();
        for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        // featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        // Xt(it,all) = featVec;
        it++;
      }
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, knnEstVec(5), gW, sW;
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    // Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size());
    Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    // Construc the feature matrix with the most similar patches
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();patchSelectVec.clear();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        if (TanimotoSim(patch,patch1,patch_vol) < 0.95) continue;
        lneighIt.SetLocation(index1);
        lpatch = lneighIt.GetNeighborhood();
        tr_label = lpatch.GetCenterValue();
        lpatch = lneighIt.GetNeighborhood();
        tr_label = lpatch.GetCenterValue();
        if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
        else {n_samples[1]++;Ln(it) = 1;}
        for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
        // featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
        // Xn(it,all) = featVec;
        it++;
      }
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    /********************** Anti silly error ocurrencies **********************/

    if (it == 0) {
      flbls(t) = 0;
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    /**************************** Balance classes *****************************/
    msamp = Min(n_samples);
    if (msamp[0] <= .05*(Ln.size() - msamp[0])) {
      msamp[1] == 0? flbls(t) = 1:flbls(t) = 0;
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // Remove possible nan rows
    std::vector<int> nan_rows;
    for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    k = 0;
    for (int nr : nan_rows) {
      nr -= k;
      RemoveRow(Xn,nr);
      Ln.tail(nr) = Ln.tail(nr+1);
      Ln.conservativeResize(Ln.size()-1);
      k++;
    }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /*************************** Compute similarities *************************/
    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();
    knn_indices = argsort(ksim,"descend");

    /********************** Knn-label estimation ******************************/
    if (Ln.size() < Knn) flbls(t) = (float)(ksim.transpose()*
      Ln(knn_indices).cast<float>())/ksim.sum();
    else flbls(t) = (float)(ksim.head(Knn).transpose()*
      Ln(knn_indices.head(Knn)).cast<float>())/ksim.head(Knn).sum();

    /**************************************************************************/

    if (isnan(flbls(t))) {
      if (!flag) {flag = true;goto feature_matrix_building;}
      else flbls(t) = 0;
    }

    flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void MultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels, uint Knn, float th) {

  /******************* Variable and object initializations ********************/

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, k1, k2, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch, grad_patch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  grad_patch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, Smat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;
  /****************************************************************************/

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Get scaled images from test image
  std::vector<ImageType::Pointer> tsimg_piramid;
  ImagePyramid(timg,Ns,tsimg_piramid);

  //Set the regions per each scale
  std::vector<ImageType::RegionType> scaledROIs(Ns-1);
  Size3D img_size = ROI_reg.GetSize();
  Index3D img_start = ROI_reg.GetIndex();
  for (int s = 2; s <= Ns; s++) {
    for (int it = 0; it != 3; it++) img_start[it] = round(img_start[it]/s);
    for (int it = 0; it != 3; it++) img_size[it] = round(img_size[it]/s);
    ImageType::RegionType scaledROI;
    scaledROI.SetSize(img_size);
    scaledROI.SetIndex(img_start);
    scaledROIs[s-2] = scaledROI;
  }

  /********************* Compute test image patch features ********************/

  neighIt.SetLocation(uncertain_inds[0]);
  patch = neighIt.GetNeighborhood();
  featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  // Xt.resize(uncertain_inds.size(),patch_vol*Ns);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();//gradNeighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        patch = neighIt.GetNeighborhood();
        sort_uncertain_inds[it] = index;
        // for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }

  // Add feaures of scaled images
  std::vector<std::vector<Index3D>> samplingvec;
  GetSamplingScaleVec(input_size, Ns,samplingvec);

  for (sc = 2; sc <= Ns; sc++) {
    neighIt = ConstNeighborhoodIteratorType(patchradius,tsimg_piramid[sc-1],scaledROIs[sc-2]);
    it = 0;
    for (Index3D idx : sort_uncertain_inds) {
      count = idx[0] + idx[1]*input_size[0] + idx[2]*input_size[0]*input_size[1];
      cindex = samplingvec[sc-2][count];
      neighIt.SetLocation(cindex);
      patch = neighIt.GetNeighborhood();
      // for (uint i = 0; i != patch_vol; i++) Xt(it,(sc-1)*patch_vol + i) = patch[i];
      featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
      Xt(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
      it++;
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);
  /****************************************************************************/

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // stringstream tlfilaname;
  // tlfilaname << "testLabels_" << current_sub << ".txt";
  // SaveEigenMatrix(tlfilaname.str(),true_lbls);

  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  //Build the image piramid for the training images
  std::vector<std::vector<ImageType::Pointer>> trimg_pyramids;
  std::vector<ImageType::Pointer> trimg_piramid;
  for (auto& trimg : itrain_set) {
    trimg_piramid.clear();
    ImagePyramid(trimg,Ns,trimg_piramid);
    trimg_pyramids.push_back(trimg_piramid);
  }

  /************************ label uncertain voxels ****************************/
  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    // Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol*Ns);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    /******* Construc the feature matrix with the most similar patches ********/
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    patchSelectVec.clear();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        if (TanimotoSim(patch,patch1,patch_vol) >= 0.95) {
          lpatch = lneighIt.GetNeighborhood();
          tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
          else {n_samples[1]++;Ln(it) = 1;}
          // for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
          Xn(it,seqN(0,featVec.size())) = featVec;
          patchSelectVec.push_back(true);
          it++;
        } else patchSelectVec.push_back(false);
      }
    }

    // Add scale patch features
    count = index[0] + index[1]*input_size[0] + index[2]*input_size[0]*input_size[1];
    for (sc = 2; sc <= Ns; sc++) {
      cindex = samplingvec[sc-2][count];
      it = 0;i = 0;
      for (k = 0; k != numOfTrainSamp; k++) {
        neighIt2 = ConstNeighborhoodIteratorType(patchradius,trimg_pyramids[k][sc-1],scaledROIs[sc-2]);
        neighIt2.GoToBegin();
        for (auto v : offsetvec) {
          if (patchSelectVec[i]) {
            index1 = SumIndex(v,cindex);
            neighIt2.SetLocation(index1);
            patch2 = neighIt2.GetNeighborhood();
            featVec = ExtractPatchFeatures(patch2, patch_radius, patch_vol);
            Xn(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
            // for (uint i = 0; i != patch_vol; i++) Xn(it,(sc-1)*patch_vol + i) = patch2[i];
            it++;
          }
          i++;
        }
      }
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    /********************** Anti silly error ocurrencies **********************/

    if (it == 0) {
      flbls(t) = 0;
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    // Assign the label of mayority class if the amount of minority class is less than 5%
    msamp = Min(n_samples);
    if (msamp[0] <= .05*(Ln.size() - msamp[0])) {
      msamp[1] == 0? flbls(t) = 1:flbls(t) = 0;
      flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    /**************************** Balance classes *****************************/
    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // Remove possible nan rows
    std::vector<int> nan_rows;
    for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    k = 0;
    for (int nr : nan_rows) {
      nr -= k;
      RemoveRow(Xn,nr);
      Ln.tail(nr) = Ln.tail(nr+1);
      Ln.conservativeResize(Ln.size()-1);
      k++;
    }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /*************************** Compute similarities *************************/
    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();

    /********************** Knn-label estimation ******************************/
    knn_indices = argsort(ksim,"descend");
    if (Ln.size() < Knn) flbls(t) = (float)(ksim.transpose()*
      Ln(knn_indices).cast<float>())/ksim.sum();
    else flbls(t) = (float)(ksim.head(Knn).transpose()*
      Ln(knn_indices.head(Knn)).cast<float>())/ksim.head(Knn).sum();

    if (isnan(flbls(t))) {
      if (!flag) {flag = true;goto feature_matrix_building;}
      else flbls(t) = 0;
    }

    flbls(t) > th ? flbls_th(t) = 1 : flbls_th(t) = 0;
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;

    /**************************************************************************/
  }

  // Compute the overall accuracy
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void BayesianMultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels, uint Knn) {

  /******************* Variable and object initializations ********************/

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, k1, k2, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch, grad_patch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  grad_patch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec,u_vec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, Smat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  std::vector<double> u(2), var(2), prior(2), cond(2), post(2), class_sim(2);
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, gW, sW, knnEstVec(5);
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;
  /****************************************************************************/

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Get scaled images from test image
  std::vector<ImageType::Pointer> tsimg_piramid;
  ImagePyramid(timg,Ns,tsimg_piramid);

  //Set the regions per each scale
  std::vector<ImageType::RegionType> scaledROIs(Ns-1);
  Size3D img_size = ROI_reg.GetSize();
  Index3D img_start = ROI_reg.GetIndex();
  for (int s = 2; s <= Ns; s++) {
    for (int it = 0; it != 3; it++) img_start[it] = round(img_start[it]/s);
    for (int it = 0; it != 3; it++) img_size[it] = round(img_size[it]/s);
    ImageType::RegionType scaledROI;
    scaledROI.SetSize(img_size);
    scaledROI.SetIndex(img_start);
    scaledROIs[s-2] = scaledROI;
  }

  /********************* Compute test image patch features ********************/

  neighIt.SetLocation(uncertain_inds[0]);
  patch = neighIt.GetNeighborhood();
  featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  // Xt.resize(uncertain_inds.size(),patch_vol*Ns);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        patch = neighIt.GetNeighborhood();
        sort_uncertain_inds[it] = index;
        featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }

  // Add feaures of scaled images
  std::vector<std::vector<Index3D>> samplingvec;
  GetSamplingScaleVec(input_size, Ns,samplingvec);

  for (sc = 2; sc <= Ns; sc++) {
    neighIt = ConstNeighborhoodIteratorType(patchradius,tsimg_piramid[sc-1],scaledROIs[sc-2]);
    it = 0;
    for (Index3D idx : sort_uncertain_inds) {
      count = idx[0] + idx[1]*input_size[0] + idx[2]*input_size[0]*input_size[1];
      cindex = samplingvec[sc-2][count];
      neighIt.SetLocation(cindex);
      patch = neighIt.GetNeighborhood();
      // for (uint i = 0; i != patch_vol; i++) Xt(it,(sc-1)*patch_vol + i) = patch[i];
      featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
      Xt(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
      it++;
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);
  /****************************************************************************/

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  //Build the image piramid for the training images
  std::vector<std::vector<ImageType::Pointer>> trimg_pyramids;
  std::vector<ImageType::Pointer> trimg_piramid;
  for (auto& trimg : itrain_set) {
    trimg_piramid.clear();
    ImagePyramid(trimg,Ns,trimg_piramid);
    trimg_pyramids.push_back(trimg_piramid);
  }

  /************************ label uncertain voxels ****************************/
  BDCSVD<Mat> svd;
  SelfAdjointEigenSolver<Mat> eig;

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    // Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol*Ns);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    u_vec.resize(numOfTrainSamp*offsetvec.size());
    it = 0;

    /******* Construc the feature matrix with the most similar patches ********/
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    patchSelectVec.clear();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        // if (NormCosineSim(patch,patch1,patch_vol) >= 0.99) {
        if (TanimotoSim(patch,patch1,patch_vol) >= 0.95) {
        // if (InvDiffSim(patch,patch1,patch_vol,"manhattan") >= 0.3) {
          lpatch = lneighIt.GetNeighborhood();
          tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
          else {n_samples[1]++;Ln(it) = 1;}
          u_vec(it) = patch1.GetCenterValue();
          // for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
          Xn(it,seqN(0,featVec.size())) = featVec;
          patchSelectVec.push_back(true);
          it++;
        } else patchSelectVec.push_back(false);
      }
    }

    // Add scale patch features
    count = index[0] + index[1]*input_size[0] + index[2]*input_size[0]*input_size[1];
    for (sc = 2; sc <= Ns; sc++) {
      cindex = samplingvec[sc-2][count];
      it = 0;i = 0;
      for (k = 0; k != numOfTrainSamp; k++) {
        neighIt2 = ConstNeighborhoodIteratorType(patchradius,trimg_pyramids[k][sc-1],scaledROIs[sc-2]);
        neighIt2.GoToBegin();
        for (auto v : offsetvec) {
          if (patchSelectVec[i]) {
            index1 = SumIndex(v,cindex);
            neighIt2.SetLocation(index1);
            patch2 = neighIt2.GetNeighborhood();
            featVec = ExtractPatchFeatures(patch2, patch_radius, patch_vol);
            Xn(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
            // for (uint i = 0; i != patch_vol; i++) Xn(it,(sc-1)*patch_vol + i) = patch2[i];
            it++;
          }
          i++;
        }
      }
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);
    u_vec.conservativeResize(it);

    /********************** Anti silly error ocurrencies **********************/

    if (it == 0) {
      flbls(t) = 0;
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << 100*acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << 100*acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    /*********************** Balance classes (Optional) ***********************/
    msamp = Min(n_samples);
    if (msamp[0] <= 10) {
      msamp[1] == 0? flbls(t) = 1:flbls(t) = 0;
      flbls(t) < 0.5 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << 100*acc << '\n';
      t++;
      continue;
    }

    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // // Remove possible nan rows
    // std::vector<int> nan_rows;
    // for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    // k = 0;
    // for (int nr : nan_rows) {
    //   nr -= k;
    //   RemoveRow(Xn,nr);
    //   Ln.tail(nr) = Ln.tail(nr+1);
    //   Ln.conservativeResize(Ln.size()-1);
    //   k++;
    // }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /*************************** Compute similarities *************************/

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();
    knn_indices = argsort(ksim,"descend");
    /************************* Bayesian MAP estimation ************************/

    // Compute class means
    u = {0,0};
    class_sim = {0,0};
    VectorXi Lns = Ln(knn_indices);
    Vec u_vecs = u_vec(knn_indices);
    for (size_t i = 0; i < Knn; i++) {
      if (Lns(i) == 0) {
        u[0] += ksim(i)*u_vecs(i);
        class_sim[0] += ksim(i);
      }
      else {
        u[1] += ksim(i)*u_vecs(i);
        class_sim[1] += ksim(i);
      }
    }

    neighIt.SetLocation(index);
    double x = (double)neighIt.GetCenterValue()[0];
    if (class_sim[0] == 0) {
      prior[0] = 0;
      cond[0] = 0;
    } else {
      u[0] /= class_sim[0];
      var[0] = 0;
      for (size_t i = 0; i < Knn; i++)
        if (Lns(i) == 0) var[0] += ksim(i)*pow(u_vecs(i) - u[0],2)/class_sim[0];
      cond[0] = NormalPDF(x,u[0],var[0]);
      prior[0] = class_sim[0]/ksim.head(Knn).sum();
    }

    if (class_sim[1] == 0) {
      prior[1] = 0;
      cond[1] = 0;
    }
    else {
      u[1] /= class_sim[1];
      var[1] = 0;
      for (size_t i = 0; i < Knn; i++)
        if (Lns(i) == 1) var[1] += ksim(i)*pow(u_vecs(i) - u[1],2)/class_sim[1];
      cond[1] = NormalPDF(x,u[1],var[1]);
      prior[1] = class_sim[1]/ksim.head(Knn).sum();
    }

    // Compute posterior probability
    post[0] = prior[0]*cond[0];
    post[1] = prior[1]*cond[1];

    flbls_th(t) = Max(post)[1];
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }

  // Compute the overall accuracy
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

inline float NormalPDF(float x, float mu, float var) {
  float prob;
  if (var == 0) prob = 0;
  else prob = exp(-(x-mu)*(x-mu)/2/var);///(float)sqrt(2*var);
  return prob;
}

void CCAMultiScaleKnnNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels) {

  /******************* Variable and object initializations ********************/

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch, grad_patch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  grad_patch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, aMat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl, Knn;
  std::vector<int> n_samples(2);
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, gW, sW, knnEstVec(5);
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;
  /****************************************************************************/

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Get scaled images from test image
  std::vector<ImageType::Pointer> tsimg_piramid;
  ImagePyramid(timg,Ns,tsimg_piramid);

  //Set the regions per each scale
  std::vector<ImageType::RegionType> scaledROIs(Ns-1);
  Size3D img_size = ROI_reg.GetSize();
  Index3D img_start = ROI_reg.GetIndex();
  for (int s = 2; s <= Ns; s++) {
    for (int it = 0; it != 3; it++) img_start[it] = round(img_start[it]/s);
    for (int it = 0; it != 3; it++) img_size[it] = round(img_size[it]/s);
    ImageType::RegionType scaledROI;
    scaledROI.SetSize(img_size);
    scaledROI.SetIndex(img_start);
    scaledROIs[s-2] = scaledROI;
  }

  /********************* Compute test image patch features ********************/

  neighIt.SetLocation(uncertain_inds[0]);
  patch = neighIt.GetNeighborhood();
  featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  // Xt.resize(uncertain_inds.size(),patch_vol*Ns);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();//gradNeighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        // gradNeighIt.SetLocation(index);
        patch = neighIt.GetNeighborhood();
        // grad_patch = gradNeighIt.GetNeighborhood();
        sort_uncertain_inds[it] = index;
        // for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }

  // Add feaures of scaled images
  std::vector<std::vector<Index3D>> samplingvec;
  GetSamplingScaleVec(input_size, Ns,samplingvec);

  for (sc = 2; sc <= Ns; sc++) {
    neighIt = ConstNeighborhoodIteratorType(patchradius,tsimg_piramid[sc-1],scaledROIs[sc-2]);
    it = 0;
    for (Index3D idx : sort_uncertain_inds) {
      count = idx[0] + idx[1]*input_size[0] + idx[2]*input_size[0]*input_size[1];
      cindex = samplingvec[sc-2][count];
      neighIt.SetLocation(cindex);
      patch = neighIt.GetNeighborhood();
      // for (uint i = 0; i != patch_vol; i++) Xt(it,(sc-1)*patch_vol + i) = patch[i];
      featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
      Xt(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
      it++;
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);
  /****************************************************************************/

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  //Build the image piramid for the training images
  std::vector<std::vector<ImageType::Pointer>> trimg_pyramids;
  std::vector<ImageType::Pointer> trimg_piramid;
  for (auto& trimg : itrain_set) {
    trimg_piramid.clear();
    ImagePyramid(trimg,Ns,trimg_piramid);
    trimg_pyramids.push_back(trimg_piramid);
  }

  //Build the image piramid for the training label images
  std::vector<std::vector<ImageType::Pointer>> ltrimg_pyramids;
  std::vector<ImageType::Pointer> ltrimg_piramid;
  for (auto& ltrimg : ltrain_set) {
    ltrimg_piramid.clear();
    ImagePyramid(ltrimg,Ns,ltrimg_piramid);
    ltrimg_pyramids.push_back(ltrimg_piramid);
  }

  /************************ label uncertain voxels ****************************/

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    Lnp.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    // Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol*Ns);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    /******* Construc the feature matrix with the most similar patches ********/
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    patchSelectVec.clear();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        // if (NormCosineSim(patch,patch1,patch_vol) >= 0.98) {
        if (TanimotoSim(patch,patch1,patch_vol) >= 0.95) {
        // if (InvDiffSim(patch,patch1,patch_vol,"manhattan") >= 0.3) {
          lpatch = lneighIt.GetNeighborhood();
          tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = -1;}
          else {n_samples[1]++;Ln(it) = 1;}
          // for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
          Xn(it,seqN(0,featVec.size())) = featVec;
          featVec = ExtractPatchFeatures(lpatch, patch_radius, patch_vol);
          Lnp(it,seqN(0,featVec.size())) = featVec;
          patchSelectVec.push_back(true);
          it++;
        } else patchSelectVec.push_back(false);
      }
    }

    // Add scale patch features
    count = index[0] + index[1]*input_size[0] + index[2]*input_size[0]*input_size[1];
    for (sc = 2; sc <= Ns; sc++) {
      cindex = samplingvec[sc-2][count];
      it = 0;i = 0;
      for (k = 0; k != numOfTrainSamp; k++) {
        neighIt2 = ConstNeighborhoodIteratorType(patchradius,trimg_pyramids[k][sc-1],scaledROIs[sc-2]);
        lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrimg_pyramids[k][sc-1],scaledROIs[sc-2]);
        neighIt2.GoToBegin();lneighIt.GoToBegin();
        for (auto v : offsetvec) {
          if (patchSelectVec[i]) {
            index1 = SumIndex(v,cindex);
            neighIt2.SetLocation(index1);
            patch2 = neighIt2.GetNeighborhood();
            lneighIt.SetLocation(index1);
            lpatch = lneighIt.GetNeighborhood();
            featVec = ExtractPatchFeatures(patch2, patch_radius, patch_vol);
            Xn(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
            featVec = ExtractPatchFeatures(lpatch, patch_radius, patch_vol);
            Lnp(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
            // for (uint i = 0; i != patch_vol; i++) Xn(it,(sc-1)*patch_vol + i) = patch2[i];
            it++;
          }
          i++;
        }
      }
    }

    Xn.conservativeResize(it, NoChange);
    Lnp.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    /********************** Anti silly error ocurrencies **********************/

    if (it == 0) {
      flbls(t) = -1;
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = -1;
      }
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    /**************************** Balance classes *****************************/
    msamp = Min(n_samples); if (msamp[1] == 0) msamp[1] = -1;
    if (msamp[0] <= 10) {
      msamp[1] == -1? flbls(t) = 1:flbls(t) = -1;
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // Remove possible nan rows
    std::vector<int> nan_rows;
    for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    k = 0;
    for (int nr : nan_rows) {
      nr -= k;
      RemoveRow(Xn,nr);
      RemoveRow(Lnp,nr);
      Ln.tail(nr) = Ln.tail(nr+1);
      Ln.conservativeResize(Ln.size()-1);
      k++;
    }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    mu = Lnp.colwise().mean();
    s = Lnp.colwise().maxCoeff() - Lnp.colwise().minCoeff();
    Lnp = (Lnp - mu.replicate(Lnp.rows(),1)).array()/s.replicate(Lnp.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(Lnp,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /******************************* Compute CCA ******************************/
    dlib::matrix<double> alpha, beta;
    dlib::matrix<double> Xmat(Xn.rows(),Xn.cols());
    dlib::matrix<double> Lmat(Xn.rows(),Xn.cols());
    for (uint i = 0; i != Xn.rows(); i++)
      for (uint j = 0; j != Xn.cols(); j++) Xmat(i,j) = Xn(i,j);

    for (uint i = 0; i != Xn.rows(); i++)
      for (uint j = 0; j != Xn.cols(); j++) Lmat(i,j) = Lnp(i,j);

    dlib::matrix<double,0,1> correlations = dlib::cca(Xmat,Lmat,alpha,beta,10);
    Mat Wmat(alpha.nr(),alpha.nc());
    for (uint i = 0; i != alpha.nr(); i++)
      for (uint j = 0; j != alpha.nc(); j++) Wmat(i,j) = alpha(i,j);

    // Mapping data
    Xn *= Wmat;
    xt *= Wmat;

    /*************************** Compute similarities *************************/

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();

    /********************** Knn-label estimation ******************************/
    knn_indices = argsort(ksim,"descend");

    if (Ln.size() < knnVec(0)) flbls(t) = (float)(ksim.transpose()*
      Ln(knn_indices).cast<float>())/ksim.sum();
    else {
      it = 0;
      for (auto knn : knnVec) {
        if (Ln.size() >= knn) {
          flbls(t) = (float)(ksim.head(knn).transpose()*
            Ln(knn_indices.head(knn)).cast<float>())/ksim.head(knn).sum();
          if (isnan(flbls(t))) {
            if (!flag) {flag = true;goto feature_matrix_building;}
            else flbls(t) = -1;
          }
          knnEstVec(it) = abs(flbls(t));
          it++;
        } else {knnEstVec(it) = 0;it++;}
      }
      knnEstVec.maxCoeff(&r);
      Knn = knnVec(r);
      flbls(t) = (float)(ksim.head(Knn).transpose()*
        Ln(knn_indices.head(Knn)).cast<float>())/ksim.head(Knn).sum();
    }
    /**************************************************************************/

    if (isnan(flbls(t))) {
      if (!flag) {flag = true;goto feature_matrix_building;}
      else flbls(t) = -1;
    }

    flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
    acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }

  // Compute the overall accuracy
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void CCAKnnNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels) {

  /******************* Variable and object initializations ********************/

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 1;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch, grad_patch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  grad_patch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist,lambda, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, aMat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl, Knn;
  std::vector<int> n_samples(2);
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, gW, sW, knnEstVec(5);
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;
  /****************************************************************************/

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  /********************* Compute test image patch features ********************/

  Xt.resize(uncertain_inds.size(),patch_vol);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        patch = neighIt.GetNeighborhood();
        sort_uncertain_inds[it] = index;
        for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        it++;
      }
    }
  }

  // SaveEigenMatrix("test_FMat.txt", Xt);
  /****************************************************************************/

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // SaveEigenMatrix("true_labels.txt",true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  /************************ label uncertain voxels ****************************/

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    // Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Lnp.resize(numOfTrainSamp*offsetvec.size(), patch_vol);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    /******* Construc the feature matrix with the most similar patches ********/
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    patchSelectVec.clear();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        if (NormCosineSim(patch,patch1,patch_vol) >= 0.98) {
          lpatch = lneighIt.GetNeighborhood();
          tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = -1;}
          else {n_samples[1]++;Ln(it) = 1;}
          for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          for (uint i = 0; i != patch_vol; i++) Lnp(it,i) = lpatch[i];
          patchSelectVec.push_back(true);
          it++;
        } else patchSelectVec.push_back(false);
      }
    }

    Xn.conservativeResize(it, NoChange);
    Lnp.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    /************************* Anti silly errores *****************************/

    if (it == 0) {
      flbls(t) = -1;
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = -1;
      }
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    /**************************** Balance classes *****************************/
    msamp = Min(n_samples); if (msamp[1] == 0) msamp[1] = -1;
    if (msamp[0] <= 10) {
      msamp[1] == -1? flbls(t) = 1:flbls(t) = -1;
      flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
      acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // Remove possible nan rows
    std::vector<int> nan_rows;
    for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    k = 0;
    for (int nr : nan_rows) {
      nr -= k;
      RemoveRow(Xn,nr);
      RemoveRow(Lnp,nr);
      Ln.tail(nr) = Ln.tail(nr+1);
      Ln.conservativeResize(Ln.size()-1);
      k++;
    }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(Lnp,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /******************************* Compute CCA ******************************/
    dlib::matrix<double> alpha, beta;
    dlib::matrix<double> Xmat(Xn.rows(),Xn.cols());
    dlib::matrix<double> Lmat(Xn.rows(),Xn.cols());
    for (uint i = 0; i != Xn.rows(); i++)
      for (uint j = 0; j != Xn.cols(); j++) Xmat(i,j) = Xn(i,j);

    for (uint i = 0; i != Xn.rows(); i++)
      for (uint j = 0; j != Xn.cols(); j++) Lmat(i,j) = Lnp(i,j);

    dlib::matrix<double,0,1> correlations = dlib::cca(Xmat,Lmat,alpha,beta,10);
    Mat Wmat(alpha.nr(),alpha.nc());
    for (uint i = 0; i != alpha.nr(); i++)
      for (uint j = 0; j != alpha.nc(); j++) Wmat(i,j) = alpha(i,j);

    // Mapping data
    Xn *= Wmat;
    xt *= Wmat;

    /*************************** Compute similarities *************************/

    dists = Vec((xt.replicate(Xn.rows(),1) - Xn).rowwise().norm());
    dist_vec.assign(dists.array().begin(), dists.array().end());
    ksig = Median(dist_vec);
    // dists.minCoeff(&ksig);
    ksim = (-dists.array().square()/2/ksig/ksig).exp();

    /********************** Knn-label estimation ******************************/
    knn_indices = argsort(ksim,"descend");

    if (Ln.size() < knnVec(0)) flbls(t) = (float)(ksim.transpose()*
      Ln(knn_indices).cast<float>())/ksim.sum();
    else {
      it = 0;
      for (auto knn : knnVec) {
        if (Ln.size() >= knn) {
          flbls(t) = (float)(ksim.head(knn).transpose()*
            Ln(knn_indices.head(knn)).cast<float>())/ksim.head(knn).sum();
          if (isnan(flbls(t))) {
            if (!flag) {flag = true;goto feature_matrix_building;}
            else flbls(t) = -1;
          }
          knnEstVec(it) = abs(flbls(t));
          it++;
        } else {knnEstVec(it) = 0;it++;}
      }
      knnEstVec.maxCoeff(&r);
      Knn = knnVec(r);
      flbls(t) = (float)(ksim.head(Knn).transpose()*
        Ln(knn_indices.head(Knn)).cast<float>())/ksim.head(Knn).sum();
    }
    /**************************************************************************/

    if (isnan(flbls(t))) {
      if (!flag) {flag = true;goto feature_matrix_building;}
      else flbls(t) = -1;
    }

    flbls(t) < 0 ? flbls_th(t) = 0 : flbls_th(t) = 1;
    acc = (true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }

  // Compute the overall accuracy
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

void MultiScaleSVMNonLocalPatchSegmentation(ImageType::Pointer& timg,
  const std::vector<ImageType::Pointer>& itrain_set,const std::vector<ImageType::Pointer>&
  ltrain_set,ImageType::RegionType ROI_reg,uint patch_radius,uint neigh_radius,float agr_th,
  const Vec& Wvec, string database,string current_sub,const ImageType::Pointer&
  manual_seg,ImageType::Pointer& test_lbl,std::vector<float>& acc_vec,
  string ranking_method, const std::vector<int>& labels, uint support_vecs,
  float lambda, float gamma) {

  /******************* Variable and object initializations ********************/

  float ssm, ssmth, max_ssm,sig, ksig, sumw, acc, est_th;
  const uint Ns = 3;

  Size3D ROI_size = ROI_reg.GetSize();
  Size3D input_size = timg->GetLargestPossibleRegion().GetSize();
  std::cout << ROI_size << '\n';
  uint k = 0, k1, k2, ind;

  const uint numOfTrainSamp = itrain_set.size();
  std::vector<bool> patchSelectVec;
  std::vector<std::vector<bool>> patchSelectMat(numOfTrainSamp);
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(patch_radius);
  ConstNeighborhoodIteratorType::NeighborhoodType patch, patch1, patch2, lpatch, grad_patch;
  patch.SetRadius(patch_radius); patch1.SetRadius(patch_radius);
  patch2.SetRadius(patch_radius); lpatch.SetRadius(patch_radius);
  grad_patch.SetRadius(patch_radius);
  std::vector<int> tr_lbls, agr;
  std::vector<float> norm_vec;
  std::vector<Index3D> uncertain_inds;
  uint patch_vol = pow(2*patch_radius + 1,3);
  uint neigh_vol = pow(2*neigh_radius + 1,3);
  Mat Xt,Kn,Lnp,Klnp,Kw,Kb,Xn,Xm,Sm,xt,Xtm,Xtg,Wmat,Wmat1,mu,s,Xtmr,Xmr,ksimMat;
  uint it, it1, sc, count,i;
  Index3D index, index1, cindex;
  std::vector<float> dist_vec,sigma_copy;
  std::vector<int> wloc,wind,bloc,bind, balance_idx;
  VectorXi wlocv,windv,blocv,bindv,eig_inds;
  std::vector<float> tmp;
  Vec wdist, bdist, featVec, tmpVec, ksim,dists,ksimVec,multiLabelVec;
  Mat Lw, Lb, Sw, Sb, Psi,Phi,Lmat, Ar, Air, tmpMat, Cn, Cr, Vmat, Zmat, Hmat, Smat;
  VectorXi wlocv1, blocv1, Ln,knn_indices;

  std::vector<Index3D> offsetvec = GetOffsetVec(neigh_radius);

  ConstNeighborhoodIteratorType neighIt(patchradius,timg,ROI_reg);
  IteratorTypeIndex outIt(test_lbl,ROI_reg);

  int t = 0,bg = 0, hip = 0, tmp_val, r, tr_label,true_lbl;
  std::vector<int> n_samples(2);
  std::vector<int> indices, indvec;
  Index3D temp_index;
  std::vector<Index3D> indices_vec;
  bool flag,flag1 = false;
  Vec x,y,dlambda, label_est, gW, sW, knnEstVec(5);
  VectorXi knnVec(5);
  knnVec << 3,5,7,9,11;
  ConstNeighborhoodIteratorType neighIt1, neighIt2, lneighIt;
  /****************************************************************************/

  // Identify the uncertain voxels
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (auto&& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Get scaled images from test image
  std::vector<ImageType::Pointer> tsimg_piramid;
  ImagePyramid(timg,Ns,tsimg_piramid);

  //Set the regions per each scale
  std::vector<ImageType::RegionType> scaledROIs(Ns-1);
  Size3D img_size = ROI_reg.GetSize();
  Index3D img_start = ROI_reg.GetIndex();
  for (int s = 2; s <= Ns; s++) {
    for (int it = 0; it != 3; it++) img_start[it] = round(img_start[it]/s);
    for (int it = 0; it != 3; it++) img_size[it] = round(img_size[it]/s);
    ImageType::RegionType scaledROI;
    scaledROI.SetSize(img_size);
    scaledROI.SetIndex(img_start);
    scaledROIs[s-2] = scaledROI;
  }

  /********************* Compute test image patch features ********************/

  neighIt.SetLocation(uncertain_inds[0]);
  patch = neighIt.GetNeighborhood();
  featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
  Xt.resize(uncertain_inds.size(),featVec.size()*Ns);
  // Xt.resize(uncertain_inds.size(),patch_vol*Ns);
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  it = 0;
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    neighIt.GoToBegin();//gradNeighIt.GoToBegin();
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (auto&& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,labels);
      if (agr[1] == i) {
        neighIt.SetLocation(index);
        // gradNeighIt.SetLocation(index);
        patch = neighIt.GetNeighborhood();
        // grad_patch = gradNeighIt.GetNeighborhood();
        sort_uncertain_inds[it] = index;
        // for (uint i = 0; i != patch_vol; i++) Xt(it,i) = patch[i];
        featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
        Xt(it,seqN(0,featVec.size())) = featVec;
        it++;
      }
    }
  }

  // Add feaures of scaled images
  std::vector<std::vector<Index3D>> samplingvec;
  GetSamplingScaleVec(input_size, Ns,samplingvec);

  for (sc = 2; sc <= Ns; sc++) {
    neighIt = ConstNeighborhoodIteratorType(patchradius,tsimg_piramid[sc-1],scaledROIs[sc-2]);
    it = 0;
    for (Index3D idx : sort_uncertain_inds) {
      count = idx[0] + idx[1]*input_size[0] + idx[2]*input_size[0]*input_size[1];
      cindex = samplingvec[sc-2][count];
      neighIt.SetLocation(cindex);
      patch = neighIt.GetNeighborhood();
      // for (uint i = 0; i != patch_vol; i++) Xt(it,(sc-1)*patch_vol + i) = patch[i];
      featVec = ExtractPatchFeatures(patch, patch_radius, patch_vol);
      Xt(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
      it++;
    }
  }
  // SaveEigenMatrix("test_FMat.txt", Xt);
  /****************************************************************************/

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI_reg);
  gtIt.GoToBegin();it = 0;
  Vec true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }
  // stringstream tlfilaname;
  // tlfilaname << "testLabels_" << current_sub << ".txt";
  // SaveEigenMatrix(tlfilaname.str(),true_lbls);

  // std::vector<Vec> lest_set(numOfTrainSamp);
  const uint np = uncertain_inds.size();
  std::cout << "Number of uncertain voxels: " << np << '\n';

  //Build the image piramid for the training images
  std::vector<std::vector<ImageType::Pointer>> trimg_pyramids;
  std::vector<ImageType::Pointer> trimg_piramid;
  for (auto& trimg : itrain_set) {
    trimg_piramid.clear();
    ImagePyramid(trimg,Ns,trimg_piramid);
    trimg_pyramids.push_back(trimg_piramid);
  }

  /************************ Initialize SVM model ****************************/
  typedef dlib::matrix<float,249,1> sample_type;
  typedef dlib::radial_basis_kernel<sample_type> kernel_type;

  dlib::svm_pegasos<kernel_type> trainer;
  trainer.set_lambda(lambda);
  trainer.set_kernel(kernel_type(gamma));
  trainer.set_max_num_sv(support_vecs);

  int label;
  float pred;
  sample_type sample;

  /************************ label uncertain voxels ****************************/
  BDCSVD<Mat> svd;
  SelfAdjointEigenSolver<Mat> eig;

  std::vector<int> msamp;
  Vec flbls = Vec::Zero(np);
  Vec flbls_th = Vec::Zero(np);
  neighIt = ConstNeighborhoodIteratorType(patchradius,timg,ROI_reg);
  neighIt.GoToBegin();

  for (Index3D index : sort_uncertain_inds) {
    flag = false;
    feature_matrix_building:
    Xn.resize(numOfTrainSamp*offsetvec.size(), featVec.size()*Ns);
    // Xn.resize(numOfTrainSamp*offsetvec.size(), patch_vol*Ns);
    Ln.resize(numOfTrainSamp*offsetvec.size());
    n_samples = {0,0};
    it = 0;

    /******* Construc the feature matrix with the most similar patches ********/
    neighIt.SetLocation(index);
    patch = neighIt.GetNeighborhood();
    patchSelectVec.clear();
    for (k = 0; k != numOfTrainSamp; k++) {
      neighIt1 = ConstNeighborhoodIteratorType(patchradius,itrain_set[k],ROI_reg);
      lneighIt = ConstNeighborhoodIteratorType(patchradius,ltrain_set[k],ROI_reg);
      neighIt1.GoToBegin();lneighIt.GoToBegin();
      for (auto v : offsetvec) {
        index1 = SumIndex(v,index);
        neighIt1.SetLocation(index1);
        lneighIt.SetLocation(index1);
        patch1 = neighIt1.GetNeighborhood();
        // if (NormCosineSim(patch,patch1,patch_vol) >= 0.99) {
        if (TanimotoSim(patch,patch1,patch_vol) >= 0.95) {
        // if (InvDiffSim(patch,patch1,patch_vol,"manhattan") >= 0.3) {
          lpatch = lneighIt.GetNeighborhood();
          tr_label = lpatch.GetCenterValue();
          if (tr_label == 0) {n_samples[0]++;Ln(it) = 0;}
          else {n_samples[1]++;Ln(it) = 1;}
          // for (uint i = 0; i != patch_vol; i++) Xn(it,i) = patch1[i];
          featVec = ExtractPatchFeatures(patch1, patch_radius, patch_vol);
          Xn(it,seqN(0,featVec.size())) = featVec;
          patchSelectVec.push_back(true);
          it++;
        } else patchSelectVec.push_back(false);
      }
    }

    // Add scale patch features
    count = index[0] + index[1]*input_size[0] + index[2]*input_size[0]*input_size[1];
    for (sc = 2; sc <= Ns; sc++) {
      cindex = samplingvec[sc-2][count];
      it = 0;i = 0;
      for (k = 0; k != numOfTrainSamp; k++) {
        neighIt2 = ConstNeighborhoodIteratorType(patchradius,trimg_pyramids[k][sc-1],scaledROIs[sc-2]);
        neighIt2.GoToBegin();
        for (auto v : offsetvec) {
          if (patchSelectVec[i]) {
            index1 = SumIndex(v,cindex);
            neighIt2.SetLocation(index1);
            patch2 = neighIt2.GetNeighborhood();
            featVec = ExtractPatchFeatures(patch2, patch_radius, patch_vol);
            Xn(it,seqN((sc-1)*featVec.size(),featVec.size())) = featVec;
            // for (uint i = 0; i != patch_vol; i++) Xn(it,(sc-1)*patch_vol + i) = patch2[i];
            it++;
          }
          i++;
        }
      }
    }

    Xn.conservativeResize(it, NoChange);
    Ln.conservativeResize(it);

    /********************** Anti silly error ocurrencies **********************/

    if (it == 0) {
      flbls(t) = 0;
      flbls(t) == 1 ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    if (it == 1) {
      flbls(t) = Ln(0);
      if (isnan(flbls(t))) {
        if (!flag) {flag = true;goto feature_matrix_building;}
        else flbls(t) = 0;
      }
      flbls(t) == 1 ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }
    // std::cout << "Feature matrix builded successfully!" << '\n';

    /**************************** Balance classes *****************************/
    msamp = Min(n_samples);
    if (msamp[0] <= .05*(Ln.size() - msamp[0])) {
      msamp[1] == 0? flbls(t) = 1:flbls(t) = 0;
      flbls(t) == 1 ? flbls_th(t) = 1 : flbls_th(t) = 0;
      acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
        .cast<int>().sum()/(float)(t+1);
      std::cout << "computed sample " << t+1 << " from " << np << '\n';
      std::cout << "partial acc: " << acc << '\n';
      t++;
      continue;
    }

    // BalanceClassSamples(Xn, Ln, Xt.row(t), msamp[0],msamp[1],"wang2018");

    // Remove possible nan rows
    std::vector<int> nan_rows;
    for (k = 0; k != Xn.rows(); k++) if(isnan(Xn(k,1))) nan_rows.push_back(k);
    k = 0;
    for (int nr : nan_rows) {
      nr -= k;
      RemoveRow(Xn,nr);
      Ln.tail(nr) = Ln.tail(nr+1);
      Ln.conservativeResize(Ln.size()-1);
      k++;
    }

    /**************************** Normalice data ******************************/
    mu = Xn.colwise().mean();
    s = Xn.colwise().maxCoeff() - Xn.colwise().minCoeff();
    Xn = (Xn - mu.replicate(Xn.rows(),1)).array()/s.replicate(Xn.rows(),1).array();

    xt = Xt.row(t);
    xt = (xt - mu).array()/s.array();

    // Remove possible nan cols
    std::vector<int> nan_cols;
    for (k = 0; k != Xt.cols(); k++) if(isnan(xt(0,k))) nan_cols.push_back(k);

    k = 0;
    for (int nc : nan_cols) {
      nc -= k;
      RemoveColumn(Xn,nc);
      RemoveColumn(xt,nc);
      k++;
    }

    /********************** SVM-label estimation ******************************/
    for (size_t i = 0; i != Xn.rows(); i++) {
      for (size_t j = 0; j != Xn.cols(); j++) sample(j) = Xn(i,j);
      // let the svm_pegasos learn about this sample
      Ln(i) == 0? label = -1: label = 1;
      trainer.train(sample,label);
    }
    for (size_t j = 0; j != Xn.cols(); j++) sample(j) = xt(j);
    pred = trainer(sample);
    pred < 0 ? flbls(t) = 0 : flbls(t) = 1;

    /**************************************************************************/
    flbls(t) == 1 ? flbls_th(t) = 1 : flbls_th(t) = 0;
    acc = 100*(true_lbls.head(t+1).array() == flbls_th.head(t+1).array())
      .cast<int>().sum()/(float)(t+1);
    std::cout << "computed sample " << t+1 << " from " << np << '\n';
    std::cout << "partial acc: " << acc << '\n';
    t++;
  }

  // Compute the overall accuracy
  acc = (true_lbls.array() == flbls_th.array()).cast<int>().sum()/(float)(t+1);
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  stringstream stm_str;
  stm_str << "estimated_labels_"<< database << "_" << current_sub;
  SaveEigenMatrix(stm_str.str(),flbls);

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI_reg);
  outnIt.GoToBegin();it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(flbls_th(it));
    it++;
  }
  std::cout << "Segmentation done!" << '\n';
}

Vec ExtractPatchFeatures(const ConstNeighborhoodIteratorType::NeighborhoodType& patch,
  uint patch_radius, uint patch_vol) {

  std::vector<float> feature_vec;
  uint i, k;
  std::vector<float> patch_vec;
  for (i = 0; i != patch_vol; i++) patch_vec.push_back(patch[i]);
  // feature_vec.insert(feature_vec.end(),patch_vec.begin(),patch_vec.end());

  // FODs
  int cidx = patch.GetCenterNeighborhoodIndex();
  Offset3D coffset = patch.GetOffset(cidx);
  Offset3D offsetidx,uoffset;
  float h1,h2;

  for (k = 0; k != patch_vol; k++) {
    if (k == cidx) continue;
    offsetidx = patch.GetOffset(k);
    uoffset[0] = offsetidx[0] + coffset[0];
    uoffset[1] = offsetidx[1] + coffset[1];
    uoffset[2] = offsetidx[2] + coffset[2];
    h1 = patch[uoffset];
    uoffset[0] = -offsetidx[0] + coffset[0];
    uoffset[1] = -offsetidx[1] + coffset[1];
    uoffset[2] = -offsetidx[2] + coffset[2];
    h2 = patch[uoffset];

    feature_vec.push_back(h1 - h2);
  }

  // SODs
  for (k = 0; k != patch_vol; k++) {
    if (k == cidx) continue;
    offsetidx = patch.GetOffset(k);
    uoffset[0] = offsetidx[0] + coffset[0];
    uoffset[1] = offsetidx[1] + coffset[1];
    uoffset[2] = offsetidx[2] + coffset[2];
    h1 = patch[uoffset];
    uoffset[0] = -offsetidx[0] + coffset[0];
    uoffset[1] = -offsetidx[1] + coffset[1];
    uoffset[2] = -offsetidx[2] + coffset[2];
    h2 = patch[uoffset];

    feature_vec.push_back(h1 + h2 - 2*patch[coffset]);
  }

  //Laplacian Filters
  for (k = 0; k != patch_vol; k++) {
    if (k == cidx) continue;
    feature_vec.push_back(patch[k] - patch[cidx]);
  }

  // centralize euclidian distance
  float lf = 0;
  for (k = 0; k != patch_vol; k++) lf += pow(patch[k] - patch[cidx],2);
  feature_vec.push_back(sqrt(lf));

  //Range difference filter
  feature_vec.push_back(Max(patch_vec)[0] - Min(patch_vec)[0]);

  // LBP features
  float lbp_code = 0;
  int lbp_th = 5, p = 0, diff_val;
  for (k = 0; k != patch_vol; k++) {
    if (k == cidx) continue;
      (patch[k] - patch[cidx]) >= lbp_th? diff_val = 1 : diff_val = 0;
      lbp_code += pow(2,p)*diff_val;
      k < cidx ? p++ : p--;
      // feature_vec.push_back(diff_val);
  }
  feature_vec.push_back(lbp_code);

  // Inter slice-patch euclidean distance
  NeighborhoodType* patch1 = new NeighborhoodType(patch);
  uint patch_side = 2*patch_radius+1;
  uint patch_plane = patch_side*patch_side;

  float dist;
  for (uint r = 1; r <= patch_radius; r++) {
    slice slc((patch_radius - r)*patch_plane,patch_plane,1);
    slice slc1((patch_radius + r)*patch_plane,patch_plane,1);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt(patch1, slc);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt1(patch1, slc1);
    dist = 0;
    while (SlicePatchIt != SlicePatchIt.End()) {
      dist += pow(*SlicePatchIt - *SlicePatchIt1,2);
      SlicePatchIt++;SlicePatchIt1++;
    }
    feature_vec.push_back(sqrt(dist));
  }

  // Inter slice-patch kernel Cosine similarity
  float fnorm, norm, norm1;
  for (uint r = 1; r <= patch_radius; r++) {
    slice slc((patch_radius - r)*patch_plane,patch_plane,1);
    slice slc1((patch_radius + r)*patch_plane,patch_plane,1);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt(patch1, slc);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt1(patch1, slc1);
    fnorm = 0;
    while (SlicePatchIt != SlicePatchIt.End()) {
      while (SlicePatchIt1 != SlicePatchIt1.End()) {
        fnorm += *SlicePatchIt * *SlicePatchIt1;
        SlicePatchIt1++;
      }
      SlicePatchIt++;
    }

    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt2(patch1, slc);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt3(patch1, slc);
    norm = 0;
    while (SlicePatchIt2 != SlicePatchIt2.End()) {
      while (SlicePatchIt3 != SlicePatchIt3.End()) {
        norm += *SlicePatchIt2 * *SlicePatchIt3;
        SlicePatchIt3++;
      }
      SlicePatchIt2++;
    }

    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt4(patch1, slc1);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt5(patch1, slc1);
    norm1 = 0;
    while (SlicePatchIt4 != SlicePatchIt4.End()) {
      while (SlicePatchIt5 != SlicePatchIt5.End()) {
        norm1 += *SlicePatchIt4 * *SlicePatchIt5;
        SlicePatchIt5++;
      }
      SlicePatchIt4++;
    }

    feature_vec.push_back(fnorm/sqrt(norm*norm1));
  }
  delete patch1;

  Vec featureVec = Vec::Map(feature_vec.data(), feature_vec.size());
  return featureVec;
}

Vec ExtractPatchFeaturesV2(const ConstNeighborhoodIteratorType::NeighborhoodType& patch,
  const ConstNeighborhoodIteratorType::NeighborhoodType& grad_patch, uint patch_radius,
  uint patch_vol) {

  std::vector<float> feature_vec;
  uint i, k;

  // Store the patch values into a vector
  std::vector<float> patch_vec;
  for (i = 0; i != patch_vol; i++) patch_vec.push_back(patch[i]);

  // Insert the gradient patch values to the feature vector
  for (i = 0; i != patch_vol; i++) feature_vec.push_back(grad_patch[i]);

  // FODs
  int cidx = patch.GetCenterNeighborhoodIndex();
  Offset3D coffset = patch.GetOffset(cidx);
  Offset3D offsetidx,uoffset;
  float h1,h2;

  for (k = 0; k != patch_vol; k++) {
    offsetidx = patch.GetOffset(k);
    uoffset[0] = offsetidx[0] + coffset[0];
    uoffset[1] = offsetidx[1] + coffset[1];
    uoffset[2] = offsetidx[2] + coffset[2];
    h1 = patch[uoffset];
    uoffset[0] = -offsetidx[0] + coffset[0];
    uoffset[1] = -offsetidx[1] + coffset[1];
    uoffset[2] = -offsetidx[2] + coffset[2];
    h2 = patch[uoffset];

    feature_vec.push_back(h1 - h2);
  }

  // SODs
  for (k = 0; k != patch_vol; k++) {
    offsetidx = patch.GetOffset(k);
    uoffset[0] = offsetidx[0] + coffset[0];
    uoffset[1] = offsetidx[1] + coffset[1];
    uoffset[2] = offsetidx[2] + coffset[2];
    h1 = patch[uoffset];
    uoffset[0] = -offsetidx[0] + coffset[0];
    uoffset[1] = -offsetidx[1] + coffset[1];
    uoffset[2] = -offsetidx[2] + coffset[2];
    h2 = patch[uoffset];

    feature_vec.push_back(h1 + h2 - 2*patch[coffset]);
  }

  //Laplacian Filters
  for (k = 0; k != patch_vol; k++) feature_vec.push_back(patch[k] - patch[cidx]);

  // centralize euclidian distance
  float lf = 0;
  for (k = 0; k != patch_vol; k++) lf += pow(patch[k] - patch[cidx],2);
  feature_vec.push_back(sqrt(lf));

  //Range difference filter
  feature_vec.push_back(Max(patch_vec)[0] - Min(patch_vec)[0]);

  // LBP features
  float lbp_code = 0;
  int lbp_th = 5, p = 0, diff_val;
  for (k = 0; k != patch_vol; k++) {
    if (k != cidx) {
      (patch[k] - patch[cidx]) >= lbp_th? diff_val = 1 : diff_val = 0;
      lbp_code += pow(2,p)*diff_val;
      k < cidx ? p++ : p--;
      // feature_vec.push_back(diff_val);
    }
  }
  feature_vec.push_back(lbp_code/patch_vol);

  // Inter slice-patch euclidean distance
  NeighborhoodType* patch1 = new NeighborhoodType(patch);
  uint patch_side = 2*patch_radius+1;
  uint patch_plane = patch_side*patch_side;

  float dist;
  for (uint r = 1; r <= patch_radius; r++) {
    slice slc((patch_radius - r)*patch_plane,patch_plane,1);
    slice slc1((patch_radius + r)*patch_plane,patch_plane,1);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt(patch1, slc);
    itk::SliceIterator<float,NeighborhoodType> SlicePatchIt1(patch1, slc1);
    dist = 0;
    while (SlicePatchIt != SlicePatchIt.End()) {
      dist += pow(*SlicePatchIt - *SlicePatchIt1,2);
      SlicePatchIt++;SlicePatchIt1++;
    }
    feature_vec.push_back(sqrt(dist));
  }
  delete patch1;

  Vec featureVec = Vec::Map(feature_vec.data(), feature_vec.size());
  return featureVec;
}

void ImagePyramid(const ImageType::Pointer& image,uint levels,
  std::vector<ImageType::Pointer>& img_piramid,float scale_factor) {

  //Fill the image piraid vector with the first level image
  img_piramid.push_back(image);

  // Resize
  ImageType::SizeType inputSize = image->GetLargestPossibleRegion().GetSize();
  ImageType::SpacingType inputSpacing = image->GetSpacing();
  ImageType::DirectionType inputDirection = image->GetDirection();
  ImageType::PointType inputOrigin = image->GetOrigin();
  ImageType::SizeType outputSize;
  ImageType::SpacingType outputSpacing;

  SmoothFilterType::Pointer gaussianFilter = SmoothFilterType::New();
  gaussianFilter->SetInput( image );
  gaussianFilter->SetVariance(1);
  gaussianFilter->Update();

  typedef itk::IdentityTransform<double, 3>  TransformType;
  TransformType::Pointer transform =  TransformType::New();
  transform->SetIdentity();

  float scale_factorcp = scale_factor;
  if (scale_factorcp != 0)
    for (uint i = 0; i != 3; i++) outputSize[i] = inputSize[i];

  for (uint k = 2; k <= levels; k++) {

    if (scale_factorcp == 0) {
      scale_factor = k;
      for (uint i = 0; i != 3; i++) outputSize[i] = inputSize[i] / scale_factor;
    } else for (uint i = 0; i != 3; i++) outputSize[i] = outputSize[i] / scale_factor;

    if (outputSize[0] < 3 or outputSize[1] < 3 or outputSize[2] < 3) break;

    // std::cout << "new size: " << outputSize << '\n';

    for (uint i = 0; i != 3; i++) outputSpacing[i] = inputSpacing[i] * scale_factor;

    ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
    resample->SetInput(gaussianFilter->GetOutput());
    resample->SetSize(outputSize);
    resample->SetOutputSpacing(outputSpacing);
    resample->SetOutputDirection(inputDirection);
    resample->SetOutputOrigin( inputOrigin );
    resample->SetTransform(transform);
    resample->UpdateLargestPossibleRegion();

    img_piramid.push_back(resample->GetOutput());

    // Create and setup a Gaussian filter
    SmoothFilterType::Pointer gaussianFilter = SmoothFilterType::New();
    gaussianFilter->SetInput( resample->GetOutput() );
    gaussianFilter->SetVariance(k);
    gaussianFilter->Update();
  }
}

std::vector<Index3D> GetOffsetVec(int PatchRadius) {
  std::vector <Index3D> offsetvec;
  Index3D offidx;
  int ii = -1 - PatchRadius, jj = -PatchRadius, kk = -PatchRadius;
  int tmp1 = pow(2*PatchRadius + 1,2);
  for (int n = 0; n < pow(2*PatchRadius + 1,3); n++) {
    if (n > 0 && n%(2*PatchRadius + 1) == 0) jj++;
    if (ii == PatchRadius) ii = -PatchRadius; else ii++;
    if (n > 0 && n%tmp1 == 0) kk++;
    if (kk == PatchRadius + 1) kk = -PatchRadius;
    if (jj == PatchRadius +1) jj = -PatchRadius;
    offidx = {jj,ii,kk};
    offsetvec.push_back(offidx);
  }
  return offsetvec;
}

void GetSamplingScaleVec(Size3D input_size, uint Ns,std::vector<std::vector<Index3D>>&
  samplingvec) {
  std::vector<std::vector<int>> binvec;
  Index3D cindex;std::vector<std::vector<float>> svec(3);
  for (int s = 2; s <= Ns; s++) {
    svec.clear();binvec.clear();
    for (int i = 0; i != 3 ; i++) {
      std::vector<int> tmp(input_size[i],0);
      binvec.push_back(tmp);
    }
    std::vector<Index3D> sidxvec;
    for (int l = 0; l != 3 ; l++)
      svec[l] = Linspace(0.0,input_size[l]-1.0,(int)input_size[l]/s);

    // Round the values inside the vectors
    for (int i = 0; i != 3 ; i++)
      for (int j = 0; j != svec[i].size(); j++) svec[i][j] = round(svec[i][j]);

    for (int i = 0; i != 3 ; i++) for (auto& j : svec[i]) binvec[i][j] = 1;
    binvec[0][0] = 0;binvec[1][0] = 0;binvec[2][0] = 0;

    Index3D idx = {0,0,0};
    for (int i = 0; i != input_size[2]; i++) {
      idx[0] = 0;idx[1] = 0;idx[2] += binvec[2][i];
      for (int j = 0; j != input_size[1]; j++) {
        idx[0] = 0;idx[1] += binvec[1][j];
        for (int k = 0; k != input_size[0]; k++) {
          idx[0] += binvec[0][k];
          sidxvec.push_back(idx);
        }
      }
    }
    samplingvec.push_back(sidxvec);
  }
}

float ComputePatchSSM(const ConstNeighborhoodIteratorType::NeighborhoodType& target_patch,
                const ConstNeighborhoodIteratorType::NeighborhoodType& neigh_patch,
                int patch_vol)
/*------------------------------------------------------------------------------
This function obtains the similarity between the patch centered at the target
voxel and another patch centered at any voxel (within its neigborhood).
--------------------------------------------------------------------------------
Input parameters:
  target_patch: patch centered at the target voxel in the intensity input image.

Output parameters:
  sim: Structural similarity between input patches
------------------------------------------------------------------------------*/
{
  // Compute the mean of the target patch
  float up = 0;
  for (int k = 0; k != patch_vol; k++) up += (float)target_patch[k]/patch_vol;

  if (isnan(up)) return 0;

  // Compute the variance of the target patch
  float vp = 0;
  for (int k = 0; k != patch_vol; k++) vp += (float)pow((target_patch[k]- up),2)/patch_vol;
  float sp = sqrt(vp);

  // Compute the mean of the neighbor patch
  float un = 0;
  for (int k = 0; k != patch_vol; k++) un += (float)neigh_patch[k]/patch_vol;
  if (isnan(un)) return 0;

  // Compute the variance of the target patch
  float vn = 0;
  for (int k = 0; k != patch_vol; k++) vn += (float)pow((neigh_patch[k]- un),2)/patch_vol;
  float sn = sqrt(vn);

  return Similarity(un,sn,up,sp);
}

template<class T> int GaussLabelAvg(T vec, uint N, uint patch_radius,
  uint patch_vol) {

  std::vector<Index3D> offsetvec = GetOffsetVec(patch_radius);
  uint k;
  Offset3D offset;
  float est_lbl = 0, sw = 0,w, d = .01;
  for ( auto v : offsetvec) {
    w = exp(-(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])/1.5);
    sw += w;
    offset = {v[0],v[1],v[2]};
    est_lbl += w*vec[offset];
  }
  est_lbl /= sw;
  if (est_lbl > .5 + d or est_lbl < .5 - d) est_lbl = round(est_lbl);
  else est_lbl = -1;
  return est_lbl;
}

std::vector<float> ComputeMaxVarSigmaV2 (const std::vector<float>& distvec) {

  std::vector<float> variance,simvec;
  std::vector<std::vector<float>> simmat;
  // std::cout << SumVector(v0,v0.size()) << '\n';
  // std::cout << v0.size() << '\n';
  float sigma_med = Median(distvec);
  float n = distvec.size();
  // std::cout << sigma_med << '\n';
  float sigma_sil = sqrt(GetVariance(distvec,n))*pow(4.0/3/n,1.0/5);
  // std::cout << sigma_sil << '\n';
  float max_sigma = 3*max(sigma_sil,sigma_med);
  uint N = 5000, it = 1;
  float min_sigma = pow(10,-16);
  float increment = (max_sigma - min_sigma)/N;
  float sigma = min_sigma;
  std::vector<float> v;
  while (sigma < max_sigma) {
    simvec.clear();
    for (auto const val : distvec) simvec.push_back(exp(-val*val/sigma/sigma/2));
    variance.push_back(GetVariance(simvec,simvec.size()));
    if (it > 1000) if ((variance[it-1] - variance[it-50]) < 0) break;
    sigma += increment;
    it++;
  }
  std::vector<std::vector<float>> tmp_var;
  tmp_var.push_back(variance);
  // SaveMatrix("variance.txt",tmp_var);
  std::vector<float> sigvec = Linspace(min_sigma,sigma,it);
  float sigma_opt = sigvec[Max(variance)[1]];
  float max_var = Max(variance)[0];
  v = {sigma_opt, max_var};
  return v;
}

template <class T>
std::vector <T> Min(std::vector<T> vec)
/*------------------------------------------------------------------------------
Compute the minimum value of a vector and its position.
--------------------------------------------------------------------------------
Input parameters:
  vec: Set of elements of which we will find the minimum value and its position.
------------------------------------------------------------------------------*/
{
  std::vector <T> min_vec;
  float min = 10000.0; int pos = 0, k = 0;
  for (const auto x : vec){
    if (x < min){
      min = x;
      pos = k;
    }
  k++;
  }
  min_vec.push_back(min);
  min_vec.push_back(pos);
  return min_vec;
}

std::vector<int> BalanceClassSamples(Mat& Xn,VectorXi& Ln, const Mat& xt, int m,
  int lbl, string method, size_t N_it, uint batch_size) {

  // m: The number of samples of the minority class
  //lbl: The monirity class

  int lbl1,k = 0;
  lbl == -1? lbl1 = 1: lbl1 = -1;
  std::vector<int> idx;
  for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl1) idx.push_back(i);

  if (method == "random") {
    std::srand ( unsigned ( std::time(0) ) );
    std::random_shuffle ( idx.begin(), idx.end() );
    std::vector<int> idx1(idx.begin(),idx.begin()+m);
    idx.clear();
    for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl) idx.push_back(i);
    idx.insert(idx.end(),idx1.begin(),idx1.end());
    VectorXi Ln1 = Ln(idx);
    Mat Xn1 = Xn(idx,all);
    Ln = Ln1; Xn = Xn1;
  }
  if (method == "cluster") {

    // find the class centroid
    Mat centroid = Mat::Zero(1,Xn.cols());
    for (auto id : idx) centroid += Xn.row(id)/idx.size();

    // find the m closest elements to the centroid
    Vec distVec = (centroid.replicate(idx.size(),1) - Xn(idx,all)).rowwise().norm();
    std::vector<float> dist_vec(distVec.array().begin(),distVec.array().end());
    float s = ComputeMaxVarSigmaV2(dist_vec)[0];
    Vec simVec = (-distVec.array().square()/2/s/s).exp();

    // Sort the distance vector ascending and choose the m first elements
    VectorXi idxVec = VectorXi::Map(idx.data(),idx.size());
    VectorXi indices = argsort(simVec,"descend");
    VectorXi choseIndices = idxVec(indices);

    idx.clear();
    for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl) idx.push_back(i);
    idx.insert(idx.end(),choseIndices.array().begin(),choseIndices.array().begin()+m);
    VectorXi Ln1 = Ln(idx);
    Mat Xn1 = Xn(idx,all);
    Ln = Ln1; Xn = Xn1;
  }
  if (method == "k_means") {

    Mat Xnc = Xn(idx,all);
    std::vector<std::vector<Eigen::ArrayXf>> kGroups(m);
    std::vector<std::vector<uint>> kIndices(m);
    Mat centroids = k_means(Xnc, kGroups, kIndices, m, N_it);
    kGroups.clear();
    idx.clear();
    for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl) idx.push_back(i);
    Mat Xn1(2*m,Xn.cols());
    Xn1 << Xn(idx,all), centroids;
    Xn = Xn1;
    VectorXi Ln1(2*m);
    Ln1 << VectorXi::Ones(m)*lbl, VectorXi::Ones(m)*lbl1;
    Ln = Ln1;
  }

  if (method == "MBK_means") {
    SaveEigenMatrix("featureMatrix.txt",Xn);
    stringstream command;
    command << "cd ..;python mini_batch_kmeans.py build/featureMatrix.txt " <<
    m << " " << batch_size;
    system(command.str().c_str());
    Mat centroids = ReadEigenMatrix("../centroids.txt");
    idx.clear();
    for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl) idx.push_back(i);
    Mat Xn1(2*m,Xn.cols());
    Xn1 << Xn(idx,all), centroids;
    Xn = Xn1;
    VectorXi Ln1(2*m);
    Ln1 << VectorXi::Ones(m)*lbl, VectorXi::Ones(m)*lbl1;
    Ln = Ln1;
  }
  if (method == "wang2018") {

    // Compute the similarities between training data and target sample
    Vec Dx = (Xn(idx,all) - xt.replicate(idx.size(),1)).rowwise().norm();
    std::vector<float> dist_vec(Dx.array().begin(),Dx.array().end());
    // float s = ComputeMaxVarSigmaV2(dist_vec)[0];
    float s = Median(dist_vec);
    // float s;
    // Dx.minCoeff(&s);
    Vec simVec = (-Dx.array().square()/2/s/s).exp();


    // Select the 80% most similar samples from the majority class
    uint p = ceil(.8*m), q = floor(.2*m);
    VectorXi tmp_indices = argsort(simVec,"descend");
    VectorXi mSimIndices = tmp_indices.head(p);

    // Select m/2 random samples from the majority class
    static std::random_device seed;
    VectorXi lSimIndices = tmp_indices.tail(idx.size() - p);
    std::srand(unsigned(seed()));
    std::random_shuffle (lSimIndices.array().begin(), lSimIndices.array().end());
    std::vector<int> idx1(lSimIndices.array().begin(),lSimIndices.array().begin()+q);

    // Concatenate most similar and random samples
    idx1.insert(idx1.end(),mSimIndices.array().begin(), mSimIndices.array().end());
    VectorXi indVec = Map<VectorXi>(idx.data(),idx.size());
    VectorXi choseIndices = indVec(idx1);

    idx.clear();
    for (int i = 0; i != Ln.size(); i++) if (Ln(i) == lbl) idx.push_back(i);
    idx.insert(idx.end(),choseIndices.array().begin(),choseIndices.array().end());
    VectorXi Ln1 = Ln(idx);
    Mat Xn1 = Xn(idx,all);
    Ln = Ln1; Xn = Xn1;
  }

  return idx;
}

void RemoveRow(Mat& matrix, uint rowToRemove) {

  uint numRows = matrix.rows()-1;
  uint numCols = matrix.cols();

  if( rowToRemove < numRows )
    matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

  matrix.conservativeResize(numRows,numCols);
}

void RemoveColumn(Mat& matrix, uint colToRemove) {

    uint numRows = matrix.rows();
    uint numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void GetGaussianKernelMatrix(const Mat& Xn, Mat& Kn, string sigma_est) {
  std::vector<float> distmat = MatEuDist(Xn);
  float sigma_opt;
  if (sigma_est == "median") sigma_opt = Median(distmat);
  if (sigma_est == "max_var") sigma_opt = ComputeMaxVarSigmaV2(distmat)[0];
  uint it = 0;
  for (int i = 0; i != Xn.rows(); i++)
    for (int j = i+1; j != Xn.rows(); j++) {
      Kn(i,j) = exp(-distmat[it]*distmat[it]/2/sigma_opt/sigma_opt);
      Kn(j,i) = Kn(i,j);
      Kn(i,i) = 1;
      it++;
    }
}

std::vector<float> MatEuDist(const Mat& mat) {

  std::vector<float> dist;
  for (int i = 0; i != mat.rows(); i++)
    for (int j = i+1; j != mat.rows(); j++)
      dist.push_back((mat.row(i) - mat.row(j)).norm());

  return dist;
}

bool argsort_comp(const argsort_pair& left, const argsort_pair& right) {
    return left.second > right.second;
}

VectorXi argsort(Vec &x, string var) {
  VectorXi indices(x.size());
  int i;
  std::vector<argsort_pair> data(x.size());
  for(i = 0; i != x.size(); i++) {
      data[i].first = i;
      data[i].second = x(i);
  }
  if (var == "descend")
    std::sort(data.begin(), data.end(), argsort_comp);
  else std::sort(data.begin(), data.end());
  for(i = 0; i != data.size(); i++) indices(i) = data[i].first;
  for(i = 0; i != data.size(); i++) x(i) = data[i].second;

  return indices;
}

Vec MixPatchValsV2(const std::vector<Index3D>& test_indices, const std::vector
  <ConstNeighborhoodIteratorType::NeighborhoodType>& est_patch_vec, const
  std::vector<Index3D>& offsetvec, const Mat& testKer) {

  uint i = 0,j;
  std::vector<float> estimation_vec;
  std::vector<int> ind_vec;
  Offset3D offset;
  Vec estVec, est_labels(test_indices.size());
  // SaveEigenMatrix("testKernel.txt",testKer);
  for (auto const& x : test_indices) {
    j = 0;estimation_vec.clear();ind_vec.clear();
    for (auto const& y : test_indices) {
      for (auto const& v : offsetvec) {
        if (CompareIndex(SumIndex(y,v),x)) {
          offset = {v[0],v[1],v[2]};
          estimation_vec.push_back(est_patch_vec[j][offset]);
          ind_vec.push_back(j);
          break;
        }
      }
      j++;
    }
    // est_labels(i) = Median(estimation_vec);
    estVec = Vec::Map(estimation_vec.data(),estimation_vec.size());
    // est_labels(i) = estVec.mean();
    est_labels(i) = (float)(estVec.array()*testKer(ind_vec,i).array()).sum()/
      testKer(i,ind_vec).sum();
    i++;
  }
  return est_labels;
}

void SaveEigenMatrix(std::string filename,const Mat& matrix) {

  std::ofstream output_file(filename);
  output_file << matrix  << '\n';
  output_file.close();
}

void SaveEigenVector(std::string filename,const VectorXi& vector) {

  std::ofstream output_file(filename);
  output_file << vector  << '\n';
  output_file.close();
}

template<class T> std::vector<float> Mode(T vec,int N)
/*------------------------------------------------------------------------------
This function computes the mode and how many times it is present in the vector
--------------------------------------------------------------------------------
Input parameters:
  v: input vector
Output parameters:
  mode: Element within the vector which appear most times in the vector
  maxcount: Numer of repetitions of the mode
------------------------------------------------------------------------------*/
{
  float prev = vec[0], mode;
  int maxcount = 0, currcount = 0;
  for (int k = 0; k != N; k++) {
    if (vec[k] == prev) {
      ++currcount;
      if (currcount > maxcount){maxcount = currcount;mode = vec[k];}
    } else currcount = 1;
    prev = vec[k];
  }
  std::vector<float> modecount;
  modecount.push_back(mode);
  modecount.push_back(maxcount);
  return modecount;
}

template<class T> std::vector<int> MostRepeatedVal(const T& vec, int N,
  const std::vector<int>& unique_vals) {
  std::unordered_map<int,int> freq;
  int k,most_rep;
  uint maxoc = 0;
  for (k = 0; k != N; k++) freq[vec[k]]++;
  for (int const &i : unique_vals)
    if (freq[i] > maxoc) {
      maxoc = freq[i];
      most_rep = i;
    }
  std::vector<int> output(2);
  output = {most_rep,maxoc};
  return output;
}

std::vector<float> Linspace(float i, float f, int N) {
  float dt = (f-i)/(N-1.0);
  std::vector<float> vec(N);
  float val = i;
  for (auto& x : vec) {x = val; val += dt;}
  return vec;
}

template <class T> T Median(const std::vector<T>& v) {
  std::vector<T> vec(v);
  sort(vec.begin(),vec.end());
  return vec[round(v.size()/2)];
}

template <class T>
float GetVariance(const T& vec, int N, float mean) {
  if (mean == 0) for (int k = 0; k != N; k++) mean += (float)vec[k]/N;
  // Compute the variance of vec
  float var = 0;
  for (int k = 0; k != N; k++) var += (float)(vec[k] - mean)*(vec[k] - mean)/(N-1);
  return var;
}

template <class T>
float GetMean(const T& vec, int N) {
  float mean = 0;
  for (int k = 0; k != N; k++) mean += (float)vec[k]/N;
  return mean;
}

template <class T>
std::vector <T> Max(std::vector<T> vec)
/*------------------------------------------------------------------------------
Compute the maximun value of a vector and its position.
--------------------------------------------------------------------------------
Input parameters:
  vec: Set of elements of which we will find the maximum value and its position.
------------------------------------------------------------------------------*/
{
  std::vector <T> max_vec;
  float max = 0.0; int pos = 0, k = 0;
  for (const auto x : vec){
    if (x > max){
      max = x;
      pos = k;
    }
    k++;
  }
  max_vec.push_back(max);
  max_vec.push_back(pos);
  return max_vec;
}

Index3D SumIndex(const Index3D& index1, const Index3D& index2)
/*------------------------------------------------------------------------------
Compute de sum of elements of two index and obtain a index of the same size.
--------------------------------------------------------------------------------
Input parameters:
  index1: Firts Index3D type variable
  index2: Second Index3D type variable
------------------------------------------------------------------------------*/
{
  Index3D sum = {index1[0] + index2[0], index1[1] + index2[1], index1[2] + index2[2]};
  return sum;
}

bool CompareIndex(Index3D index1,Index3D index2)
/*------------------------------------------------------------------------------
Compare two index elements and return true if these are equal and false otherwise.
--------------------------------------------------------------------------------
Input parameters:
  index1: Firts Index3D element .
  index2: Second Index3D element.
------------------------------------------------------------------------------*/
{
  if(index1[0] == index2[0] && index1[1] == index2[1] && index1[2] == index2[2])
    return true;
  else return false;
}

Mat k_means(const Mat &data, std::vector<std::vector<Eigen::ArrayXf>>& kGroups,
  std::vector<std::vector<uint>>& kIndices, uint16_t k, size_t n_iter) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.rows() - 1);

  uint d = data.cols();
  uint m = data.rows();
  float th = pow(10,-6), differ;

  std::cout <<"number of clusters: "<< k << '\n';

  // Stardarization
  Mat mu(1,d), s(1,d);
  mu << data.colwise().mean();
  s << data.colwise().maxCoeff() - data.colwise().minCoeff();
  const Mat datac = (data - mu.replicate(m,1)).array()/s.replicate(m,1).array();

  Mat centroids(k, d);
  Mat centroidscp(k, d);
  Mat dists, distances;
  Eigen::ArrayXXf sum;
  Eigen::ArrayXf counts;
  Eigen::ArrayXd::Index argmin;
  for (size_t cluster = 0; cluster != k; cluster++)
    centroids.row(cluster) = datac.row(indices(random_number_generator));

  Mat centroidsr(k*m,d);
  for (size_t cluster = 0; cluster != k; cluster++)
    centroidsr(seqN(cluster*m,m),all) = centroids.row(cluster).replicate(m,1);

  const Mat datar = datac.replicate(k,1);
  size_t iter = 0;
  do{
    dists = (datar - centroidsr).rowwise().norm();
    distances = Map<Mat>(dists.data(), m,k);
    sum = Eigen::ArrayXXf::Zero(k, d);
    counts = Eigen::ArrayXf::Ones(k);
    for (size_t index = 0; index < m; ++index) {
      distances.row(index).minCoeff(&argmin);
      kGroups[argmin].push_back(data.row(index).array());
      kIndices[argmin].push_back(index);
      sum.row(argmin) += data.row(index).array();
      counts(argmin) += 1;
    }
    centroidscp = centroids;
    centroids = sum.colwise() / counts;
    for (size_t cluster = 0; cluster != k; cluster++)
      centroidsr(seqN(cluster*m,m),all) = centroids.row(cluster).replicate(m,1);
    iter++;
    differ = ((centroids - centroidscp).rowwise().norm()).mean();
    std::cout << differ << '\n';
  } while(differ > th or iter < n_iter);

  return centroids;
}

Mat ReadEigenMatrix(string filename) {
  int cols = 0, rows = 0;
  double buff[MAXBUFSIZE];

  // Read numbers from file into buffer.
  std::ifstream infile(filename);
  while (! infile.eof())
      {
      string line;
      getline(infile, line);

      int temp_cols = 0;
      stringstream stream(line);
      while(! stream.eof())
          stream >> buff[cols*rows+temp_cols++];

      if (temp_cols == 0)
          continue;

      if (cols == 0)
          cols = temp_cols;

      rows++;
      }

  infile.close();

  rows--;

  // Populate matrix with numbers.
  Mat result(rows,cols);
  for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
          result(i,j) = buff[ cols*i+j ];

  return result;
}

float Similarity(float un, float vn, float up, float vp)
/*------------------------------------------------------------------------------
This function obtains the structural similarity between two patches.
--------------------------------------------------------------------------------
Input parameters:
  un: Mean of a neighborhood patch.
  vn: Standard desviation of a neighborhood patch.
  up: Mean of the target patch.
  vp: Standad desviation of the target patch.

Output parameters:
  ssm: resulting of ute the structural similarity measure (SSM).
------------------------------------------------------------------------------*/
{
  float eps = pow(10,-5);
  return ((float)(2.0*up*un + eps)/(up*up + un*un + eps))*((float)(2.0*vp*vn + eps)/(vp*vp + vn*vn + eps));
  //return exp((abs(up - un) + abs(vn - vp))/vn/vp);
}

float ComputeDice(const ImageType::Pointer& true_segmentation_image,
  const ImageType::Pointer& estimated_segmentation_image, const std::vector<int>& labels) {

  ImageType::RegionType img_region = true_segmentation_image->GetLargestPossibleRegion();
  IteratorType trueSegIt(true_segmentation_image, img_region);
  ConstIteratorType estimatedSegIt(estimated_segmentation_image, img_region);
  float mean_dice_index = 0,dice_index;
  std::vector<int> lbls(labels.begin()+1,labels.end());
  int num_of_labels = lbls.size();
    for (auto lbl : lbls) {
      float num = 0,true_count = 0, estimated_count = 0;
      trueSegIt.GoToBegin(); estimatedSegIt.GoToBegin();
      while (!trueSegIt.IsAtEnd()) {
        if (trueSegIt.Get() == lbl) true_count++;
        if (estimatedSegIt.Get() == lbl) estimated_count++;
        if (trueSegIt.Get() == estimatedSegIt.Get()) if (trueSegIt.Get() == lbl) num++;
        ++trueSegIt;++estimatedSegIt;
      }
      dice_index = 200*num/(true_count + estimated_count);
      //std::cout << "dice_index_" << lbl << ": " << dice_index << '\n';
      mean_dice_index += dice_index/num_of_labels;
    }
  return mean_dice_index;
}

float ComputeJaccard(const ImageType::Pointer& true_segmentation_image,
  const ImageType::Pointer& estimated_segmentation_image, const std::vector<int>& labels) {

  ImageType::RegionType img_region = true_segmentation_image->GetLargestPossibleRegion();
  IteratorType trueSegIt(true_segmentation_image, img_region);
  ConstIteratorType estimatedSegIt(estimated_segmentation_image, img_region);
  float mean_jaccard_index = 0,jaccard_index;
  std::vector<int> lbls(labels.begin()+1,labels.end());// Does not take into account background
  int num_of_labels = lbls.size();
    for (auto lbl : lbls) {
      float num = 0,true_count = 0, estimated_count = 0;
      trueSegIt.GoToBegin(); estimatedSegIt.GoToBegin();
      while (!trueSegIt.IsAtEnd()) {
        if (trueSegIt.Get() == lbl) true_count++;
        if (estimatedSegIt.Get() == lbl) estimated_count++;
        if (trueSegIt.Get() == estimatedSegIt.Get()) if (trueSegIt.Get() == lbl) num++;
        ++trueSegIt;++estimatedSegIt;
      }
      jaccard_index = 100*num/(true_count + estimated_count -num);
      //std::cout << "dice_index_" << lbl << ": " << dice_index << '\n';
      mean_jaccard_index += jaccard_index/num_of_labels;
    }
  return mean_jaccard_index;
}

float ComputeRecall(const ImageType::Pointer& true_segmentation_image,
  const ImageType::Pointer& estimated_segmentation_image, const std::vector<int>& labels) {

  ImageType::RegionType img_region = true_segmentation_image->GetLargestPossibleRegion();
  IteratorType trueSegIt(true_segmentation_image, img_region);
  ConstIteratorType estimatedSegIt(estimated_segmentation_image, img_region);
  float mean_recall_index = 0,recall_index;
  std::vector<int> lbls(labels.begin()+1,labels.end());// Does not take into account background
  int num_of_labels = lbls.size();
    for (auto lbl : lbls) {
      float num = 0,true_count = 0, estimated_count = 0;
      trueSegIt.GoToBegin(); estimatedSegIt.GoToBegin();
      while (!trueSegIt.IsAtEnd()) {
        if (trueSegIt.Get() == lbl) true_count++;
        if (estimatedSegIt.Get() == lbl) estimated_count++;
        if (trueSegIt.Get() == estimatedSegIt.Get()) if (trueSegIt.Get() == lbl) num++;
        ++trueSegIt;++estimatedSegIt;
      }
      recall_index = 100*num/true_count;
      //std::cout << "dice_index_" << lbl << ": " << dice_index << '\n';
      mean_recall_index += recall_index/num_of_labels;
    }
  return mean_recall_index;
}

float ComputePrecision(const ImageType::Pointer& true_segmentation_image,
  const ImageType::Pointer& estimated_segmentation_image, const std::vector<int>& labels) {

  ImageType::RegionType img_region = true_segmentation_image->GetLargestPossibleRegion();
  IteratorType trueSegIt(true_segmentation_image, img_region);
  ConstIteratorType estimatedSegIt(estimated_segmentation_image, img_region);
  float mean_precision_index = 0,precision_index;
  std::vector<int> lbls(labels.begin()+1,labels.end());// Does not take into account background
  int num_of_labels = lbls.size();
    for (auto lbl : lbls) {
      float num = 0,true_count = 0, estimated_count = 0;
      trueSegIt.GoToBegin(); estimatedSegIt.GoToBegin();
      while (!trueSegIt.IsAtEnd()) {
        if (trueSegIt.Get() == lbl) true_count++;
        if (estimatedSegIt.Get() == lbl) estimated_count++;
        if (trueSegIt.Get() == estimatedSegIt.Get()) if (trueSegIt.Get() == lbl) num++;
        ++trueSegIt;++estimatedSegIt;
      }
      precision_index = 100*num/estimated_count;
      //std::cout << "dice_index_" << lbl << ": " << dice_index << '\n';
      mean_precision_index += precision_index/num_of_labels;
    }
  return mean_precision_index;
}

void InitializeImage(const ImageType::Pointer& inputImg,ImageType::Pointer& image)
/*------------------------------------------------------------------------------
This function creates an image with reference in another image, with the same size,
spacing, direction and origin, this in order to conserve the same coordenate the
input/reference image.
--------------------------------------------------------------------------------
Input parameters: Reference image
        Region of reference image
Output parameters: New image
         Region of the new image
--------------------------------------------------------------------------------
*/
{
  ImageType::RegionType inputRegion = inputImg -> GetLargestPossibleRegion();
  image->SetRegions( inputRegion );
  image->SetSpacing(inputImg->GetSpacing());
  image->SetOrigin(inputImg->GetOrigin());
  image->SetDirection(inputImg->GetDirection());
  image->Allocate();
}

void CreateImage(ImageType::Pointer& image,const ImageType::RegionType& input_region,
                 int val)
/*------------------------------------------------------------------------------
Create an image full of the values especified by the val varible.
--------------------------------------------------------------------------------
Inpunt parameters:
  input_region: Region that will assigned to the new image.
  val: conts values that willbe assigned to every voxel in the image.

Output parameters:
  image: New image with a region especified by input_region and with all voxeles
  equal to val.
------------------------------------------------------------------------------*/
{
  //mutex myMutex;
  //lock_guard <mutex> guard(myMutex);
  image->SetRegions( input_region );
  image->Allocate();

  IteratorType imageIt(image,input_region);
  while(!imageIt.IsAtEnd()){
    imageIt.Set(val);
    ++imageIt;
  }
}

void GetROIs(const std::vector<ImageType::Pointer>& tr_imgs, ImageType::RegionType&
   ROI_region, string subject_path) {

  CharImageType::Pointer union_img = CharImageType::New();
  ImageType::RegionType img_region = tr_imgs[0]->GetLargestPossibleRegion();

  union_img->SetRegions(img_region);
  union_img->SetSpacing(tr_imgs[0]->GetSpacing());
  union_img->SetOrigin(tr_imgs[0]->GetOrigin());
  union_img->SetDirection(tr_imgs[0]->GetDirection());
  union_img->Allocate();

  IteratorTypeIndexChar outIt(union_img,img_region);
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    outIt.Set(0);
    for (auto tr_img : tr_imgs)
      if (tr_img->GetPixel(outIt.GetIndex()) >= 1.0) {
        outIt.Set(1);
        break;
      }
    ++outIt;
  }
  ExtractLargestObject(union_img,2);

  stringstream ROI_filename, ROIBox_filename;
  ROI_filename << subject_path << "/ROI.nii.gz";
  Save3DImageLabel(union_img,ROI_filename.str());
  Index3D ROI_start;Size3D ROI_size;
  GetImgROI(union_img,ROI_start,ROI_size);

  //Set the ROI of input image
  ROI_region.SetIndex(ROI_start);
  ROI_region.SetSize(ROI_size);

  IteratorTypeIndexChar outIt1(union_img,ROI_region);
  outIt1.GoToBegin();
  while (!outIt1.IsAtEnd()) {
    outIt1.Set(1);
    ++outIt1;
  }
  ROIBox_filename << subject_path << "/ROIBox.nii.gz";
  Save3DImageLabel(union_img,ROIBox_filename.str());
}

void ComputeMI(const ImageType::Pointer& test_img, const std::vector<ImageType::Pointer>&
  training_images,const ImageType::RegionType& ROI_region,std::vector<float>& MI_vec) {


  ImageType::Pointer test_img_ROI = ImageType::New();
  Index3D start = {0,0,0};
  ImageType::RegionType desiredRegion(start,ROI_region.GetSize());
  test_img_ROI->SetRegions( desiredRegion );
  test_img_ROI->SetSpacing(test_img->GetSpacing());
  test_img_ROI->SetOrigin(test_img->GetOrigin());
  test_img_ROI->SetDirection(test_img->GetDirection());
  test_img_ROI->Allocate();
  ConstIteratorType inputIt(test_img,ROI_region);
  IteratorType outputIt(test_img_ROI,test_img_ROI->GetRequestedRegion());
  inputIt.GoToBegin();outputIt.GoToBegin();
  while (!inputIt.IsAtEnd()) {outputIt.Set(inputIt.Get());++inputIt;++outputIt;}

  //Save3DImage(test_img_ROI,"regionimg.nii");

  int img_count = 0;
  for (auto&& tr_img : training_images) {

    ImageType::Pointer tr_img_ROI = ImageType::New();
    Index3D start = {0,0,0};
    ImageType::RegionType desiredRegion(start,ROI_region.GetSize());
    tr_img_ROI->SetRegions( desiredRegion );
    tr_img_ROI->SetSpacing(tr_img->GetSpacing());
    tr_img_ROI->SetOrigin(tr_img->GetOrigin());
    tr_img_ROI->SetDirection(tr_img->GetDirection());
    tr_img_ROI->Allocate();
    ConstIteratorType inputIt1(tr_img,ROI_region);
    IteratorType outputIt1(tr_img_ROI,tr_img_ROI->GetRequestedRegion());
    inputIt1.GoToBegin();outputIt1.GoToBegin();
    while (!inputIt1.IsAtEnd()) {outputIt1.Set(inputIt1.Get());++inputIt1;++outputIt1;}

    //Save3DImage(tr_img_ROI,"regiontrimg.nii");
    JoinFilterType::Pointer joinFilter = JoinFilterType::New();
    joinFilter->SetInput1( test_img_ROI );
    joinFilter->SetInput2( tr_img_ROI );

    try
      {
      joinFilter->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
      std::cerr << excp << std::endl;
      std::cout << "training image index: " << img_count << '\n';
      }
      img_count++;

    HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
    histogramFilter->SetInput(  joinFilter->GetOutput()  );
    histogramFilter->SetMarginalScale( 10.0 );

    std::vector<float> v1 = GetMaxMinOfImage(test_img);
    std::vector<float> v2 = GetMaxMinOfImage(tr_img);

    HistogramSizeType size( 2 );
    size[0] = v1[0] - v1[1] + 1;  // number of bins for the first  channel
    size[1] = v2[0] - v2[1] + 1;  // number of bins for the second channel

    histogramFilter->SetHistogramSize( size );

    HistogramMeasurementVectorType binMinimum( 3 );
    HistogramMeasurementVectorType binMaximum( 3 );
    float max_val = std::max(v1[0],v2[0]);
    float min_val = std::min(v1[1],v2[1]);

    binMinimum[0] = min_val - 0.5;
    binMinimum[1] = min_val - 0.5;
    binMinimum[2] = min_val - 0.5;

    binMaximum[0] = max_val + 0.5;
    binMaximum[1] = max_val + 0.5;
    binMaximum[2] = max_val + 0.5;

    histogramFilter->SetHistogramBinMinimum( binMinimum );
    histogramFilter->SetHistogramBinMaximum( binMaximum );
    histogramFilter->Update();

    const HistogramType * histogram = histogramFilter->GetOutput();
    HistogramType::ConstIterator itr = histogram->Begin();
    HistogramType::ConstIterator end = histogram->End();
    const double Sum = histogram->GetTotalFrequency();

    double JointEntropy = 0.0;
    while( itr != end )
      {
      const double count = itr.GetFrequency();
      if( count > 0.0 )
        {
        const double probability = count / Sum;
        JointEntropy +=
          - probability * std::log( probability ) / std::log( 2.0 );
        }
      ++itr;
      }

    size[0] = v1[0] - v1[1] + 1;  // number of bins for the first  channel
    size[1] =   1;  // number of bins for the second channel
    histogramFilter->SetHistogramSize( size );
    histogramFilter->Update();

    itr = histogram->Begin();
    end = histogram->End();
    double Entropy1 = 0.0;
    while( itr != end )
      {
      const double count = itr.GetFrequency();
      if( count > 0.0 )
        {
        const double probability = count / Sum;
        Entropy1 += - probability * std::log( probability ) / std::log( 2.0 );
        }
      ++itr;
      }

    size[0] =   1;  // number of bins for the first channel
    size[1] = v2[0] - v2[1] + 1;  // number of bins for the second channel
    histogramFilter->SetHistogramSize( size );
    histogramFilter->Update();

    itr = histogram->Begin();
    end = histogram->End();
    double Entropy2 = 0.0;
    while( itr != end )
      {
      const double count = itr.GetFrequency();
      if( count > 0.0 )
        {
        const double probability = count / Sum;
        Entropy2 += - probability * std::log( probability ) / std::log( 2.0 );
        }
      ++itr;
      }

    double NormalizedMutualInformation = 2.0 * (1 - JointEntropy / ( Entropy1 + Entropy2 ));
    std::cout << "NMI: " << NormalizedMutualInformation << '\n';
    MI_vec.push_back(NormalizedMutualInformation);
  }
}

void BinaryClosing(ImageType::Pointer& image, int radius) {
  ImageType::Pointer img = ImageType::New();
  InitializeImage(image,img);
  IteratorType imgIt(img,image->GetRequestedRegion());
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(radius);
  int patch_vol = pow(2*radius + 1,3);
  ConstNeighborhoodIteratorType patchIt(patchradius,image,image->GetRequestedRegion());
  imgIt.GoToBegin();patchIt.GoToBegin();
  while(!imgIt.IsAtEnd()) {
    imgIt.Set(Mode(patchIt.GetNeighborhood(),patch_vol)[0]);
    ++imgIt;++patchIt;
  }
  image = img;
}

template <class T>
void SaveMatrix(std::string filename,const std::vector<std::vector<T>>& matrix) {

  std::ofstream output_file(filename);
  int n;
  for (auto x : matrix){
    n = x.size();
    for (int k = 0; k != n; k++) if (k != n-1) output_file << x[k] << " ";else output_file << x[k];
    output_file << "\n";
  }
  output_file.close();
}

template <class T>
void SaveVector(std::string filename,const std::vector<T>& vec) {

  std::ofstream output_file(filename);
  uint n = vec.size();
  for (int k = 0; k != n; k++) if (k != n-1) output_file << vec[k] << " ";else output_file << vec[k];
  output_file << "\n";

  output_file.close();
}

void ReadMatrix(std::string filename, std::vector<std::vector<float>>& mat) {

  std::ifstream file(filename);
  string line;
  std::vector<float> vec;
  std::vector<std::string> split_str;

  while (getline(file,line)) {
    boost::split(split_str,line,boost::is_any_of(" "));
    vec.clear();
    for (string value : split_str) vec.push_back(atof(value.c_str()));
    mat.push_back(vec);
  }
  file.close();
}

void ExtractLargestObject(CharImageType::Pointer& img, uint desiredObjects) {
  ConnectedComponentImageFilterType::Pointer connected = ConnectedComponentImageFilterType::New ();
  connected->SetInput(img);
  connected->Update();
  if (connected->GetObjectCount() > desiredObjects) {
    LabelShapeKeepNObjectsImageFilterType::Pointer
      labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
    labelShapeKeepNObjectsImageFilter->SetInput( connected -> GetOutput() );
    labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
    labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 2 );
    labelShapeKeepNObjectsImageFilter->SetAttribute(
      LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
    labelShapeKeepNObjectsImageFilter->Update();
    img = labelShapeKeepNObjectsImageFilter->GetOutput();

    // Convert to binary image
    ImageType::RegionType img_region = img->GetLargestPossibleRegion();
    IteratorTypeIndexChar outIt(img,img_region);
    outIt.GoToBegin();
    while (!outIt.IsAtEnd()) {
        if (outIt.Get() > 1.0) outIt.Set(1);
      ++outIt;
    }
  }
}

void Save3DImageLabel(const CharImageType::Pointer& image, string filename)
/*------------------------------------------------------------------------------
This function saves a 3D image in a format specified by the filename, in
general .nii.
-------------------maximumValue-------------------------------------------------
Input parameters:
  image: 3D image to be saved
  filename: name of the image with its respective format
--------------------------------------------------------------------------------
*/
{
  WriterTypeChar::Pointer writer = WriterTypeChar::New();
  writer->SetInput(image);
  writer->SetFileName(filename);
  writer->Update();
}

void GetImgROI(const CharImageType::Pointer& image,Index3D& start,Size3D& size)
/*******************************************************************************
In order to use this function and it works properly the input images must be
clean(without noise)or at least with a homogeneous black background.
*******************************************************************************/
{
  CharImageType::RegionType img_region = image->GetLargestPossibleRegion();
  // Find the size of the input image
  CharImageType::SizeType img_size = img_region.GetSize();

  CharSliceCalculatorFilterType::Pointer SliceCalculatorFilter
      = CharSliceCalculatorFilterType::New ();

  int tol = 5;
	// Create points
	Index3D p0, p1;
  int SliceIdx, dif, count;
  float maximumValue, minimumValue;
  for(int axis = 0; axis != 3; axis++){
    SliceIdx = 0;
    count = 0;
    float tStart = clock();
		while(SliceIdx != img_size[axis]){
      // Finding the maximun and minimun value of the slice
			GetMaxMinOfSlice(image, SliceCalculatorFilter, SliceIdx, axis, maximumValue,
        minimumValue);
      if (maximumValue > 0)  count++;
      if (count >= tol) {p0[axis] = SliceIdx - tol - 1;break;}
      SliceIdx++;
		}
    // std::cout << "t1: " << (clock() - tStart)/CLOCKS_PER_SEC << '\n';
    SliceIdx = img_size[axis]-tol;
    count = 0;
    float tStart1 = clock();
    while(SliceIdx != tol){
      // Finding the maximun and minimun value of the slice
			GetMaxMinOfSlice(image, SliceCalculatorFilter, SliceIdx,
        axis, maximumValue,minimumValue);
      if (maximumValue > 0)  count++;
      if (count >= tol) {p1[axis] = SliceIdx + tol + 1;break;}
      SliceIdx--;
		}
    //std::cout << "t2: " << (clock() - tStart1)/CLOCKS_PER_SEC << '\n';
	}
	start = p0;
  size[0] = p1[0] -p0[0]; size[1] = p1[1] -p0[1]; size[2] = p1[2] -p0[2];
}

vector <float> GetMaxMinOfImage(const ImageType::Pointer& input)
/*------------------------------------------------------------------------------
This function computes the maximum and minimum value of a slice of a volumetric
image.
--------------------------------------------------------------------------------
Input parameters:
  Input: Image to extract the slice
  Axis: The axis of the image in which we are going to extract the slice
  Slice: The number of the slice to extract

Output parameters:
  maximumValue: The maximum value of the slice extracted
  minimumValue: The minimum value of the slice extracted
--------------------------------------------------------------------------------
*/
{
  ImageCalculatorFilterType::Pointer imageCalculatorFilter
        = ImageCalculatorFilterType::New ();
  imageCalculatorFilter->SetImage(input);
  imageCalculatorFilter->Compute();
  float maximumValue = imageCalculatorFilter->GetMaximum();
  float minimumValue = imageCalculatorFilter->GetMinimum();
  vector<float> maxmin;
  maxmin.push_back(maximumValue);
  maxmin.push_back(minimumValue);
  return maxmin;
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

void GetMaxMinOfSlice(const CharImageType:: Pointer& input,CharSliceCalculatorFilterType::Pointer
                      SliceCalculatorFilter, int slicenum, int axis, float&
                      maximumValue, float& minimumValue){

  CharSliceType::Pointer slice = CharSliceType::New();
  GetCharSlice(input, slicenum, axis, slice);
  SliceCalculatorFilter->SetImage(slice);
  SliceCalculatorFilter->Compute();
  maximumValue = SliceCalculatorFilter->GetMaximum();
  minimumValue = SliceCalculatorFilter->GetMinimum();
}

void RescaleImage(ImageType::Pointer& ImInput,int valmin,int valmax){

    RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    rescaleFilter->SetInput(ImInput);
    rescaleFilter->SetOutputMinimum(valmin);
    rescaleFilter->SetOutputMaximum(valmax);
    rescaleFilter->Update();
    ImInput = rescaleFilter->GetOutput();
}

void GetCharSlice(const CharImageType::Pointer& input,int sliceNumber,int axis,
  CharSliceType::Pointer& output)
{
  CharSliceFilterType::Pointer filter = CharSliceFilterType::New();
  CharImageType::RegionType inputRegion = input->GetLargestPossibleRegion();
  CharImageType::IndexType start = {0,0,0};
  CharImageType::SizeType size = inputRegion.GetSize();
  size[axis] = 0; start[axis] = sliceNumber;
  CharImageType::RegionType desiredRegion;
  desiredRegion.SetSize( size );
  desiredRegion.SetIndex( start );
  filter->SetExtractionRegion( desiredRegion );
  filter->SetInput( input );
  #if ITK_VERSION_MAJOR >= 4
  filter->SetDirectionCollapseToIdentity(); // This is required.
  #endif
  filter->Update();
  output = filter->GetOutput();
}

template <class T>
void ReadVector(std::string filename, std::vector<T>& vec) {
  std::ifstream file(filename);
  float val;
  while (file >> val) vec.push_back(val);
  file.close();
}

template <class T>
float SquareEuclideanDistance(T point1, T point0,uint N)
/*------------------------------------------------------------------------------
This function as its name says ute the square euclidean distance between
two vectors or points.
--------------------------------------------------------------------------------
Input parameters:
  point1: firts vector or point
  point0: second vector or point
Output parameters:
  dist: distance between point1 and point0
------------------------------------------------------------------------------*/
{
  float dist = 0.0;
  for(uint k = 0; k != N; k++) dist += pow(point1[k] - point0[k],2);
  return dist;
}

template <class T>
void Sum2Vecs(const T& v1, const T&v2, int N, std::vector<float>& s) {
  for (int k = 0; k != N; k++) s[k] = v1[k] + v2[k];
}

void Save3DImage(const ImageType::Pointer& image, string filename)
/*------------------------------------------------------------------------------
This function saves a 3D image in a format specified by the filename, in
general .nii.
-------------------maximumValue-------------------------------------------------
Input parameters:
  image: 3D image to be saved
  filename: name of the image with its respective format
--------------------------------------------------------------------------------
*/
{
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(image);
  writer->SetFileName(filename);
  writer->Update();
}

void LoadPrecomputedSegmentation(const std::vector<ImageType::Pointer>& ltrain_set,
  ImageType::RegionType ROI, string database, string current_sub, float agr_th,
  ImageType::Pointer& test_lbl, const std::vector<int>& classes,
  const ImageType::Pointer& manual_seg, std::vector<float>& acc_vec,float th) {

  const uint numOfTrainSamp = ltrain_set.size();

  uint it, i;
  stringstream filename;
  filename << "estimated_labels_" << database << "_" << current_sub;
  std::vector<int> agr, tr_lbls;
  std::vector<float> test_labels;
  std::vector<Index3D> uncertain_inds;
  ReadVector(filename.str(),test_labels);
  // if (atoi(current_sub.c_str()) != 1)
  //   for (float &lbl : test_labels) lbl = (lbl + 1)/2;
  std::vector<uint> labels;
  for (float lbl : test_labels) lbl < th? labels.push_back(0): labels.push_back(1);
  VectorXi est_labels = VectorXi::Map((int*)labels.data(),labels.size());

  IteratorTypeIndex outIt(test_lbl,ROI);
  outIt.GoToBegin();
  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (const auto& tr_limg : ltrain_set)
      tr_lbls.push_back(tr_limg -> GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,classes);
    if (agr[1] == numOfTrainSamp) outIt.Set(agr[0]);
    else uncertain_inds.push_back(outIt.GetIndex());
    ++outIt;
  }

  // Compute prior agreement
  it = 0;
  std::vector<Index3D> sort_uncertain_inds(uncertain_inds.size());
  for (i = numOfTrainSamp-1; i >= round(numOfTrainSamp/2.0); i--) {
    for (Index3D index : uncertain_inds) {
      tr_lbls.clear();
      for (const auto& tr_limg : ltrain_set) tr_lbls.push_back(tr_limg -> GetPixel(index));
      agr = MostRepeatedVal(tr_lbls,numOfTrainSamp,classes);
      if (agr[1] == i) {
        sort_uncertain_inds[it] = index;
        it++;
      }
    }
  }

  // Save true labels for uncertain indeces
  ConstNeighborhoodIteratorType::RadiusType patchradius; patchradius.Fill(1);
  ConstNeighborhoodIteratorType gtIt(patchradius,manual_seg,ROI);
  gtIt.GoToBegin();it = 0;
  VectorXi true_lbls(uncertain_inds.size());
  for (Index3D index : sort_uncertain_inds) {
    gtIt.SetLocation(index);
    true_lbls(it) = gtIt.GetCenterValue()[0];
    it++;
  }

  float acc = (true_lbls.array() == est_labels.array()).cast<int>().sum()/(float)(est_labels.size());
  acc_vec.push_back(acc);
  std::cout << "final acc: " << acc << '\n';

  // Save labels for uncertain indices
  NeighborhoodIteratorType outnIt(patchradius,test_lbl,ROI);
  outnIt.GoToBegin();
  it = 0;
  for (Index3D index : sort_uncertain_inds) {
    outnIt.SetLocation(index);
    outnIt.SetCenterPixel(labels[it]);
    it++;
  }

  std::cout << "Segmentation done!" << '\n';
}

void WMV(const std::vector<ImageType::Pointer>& training_label_maps,const Vec& Wvec,
  std::vector<int> lbls,float agr_th, ImageType::Pointer& test_lbl) {
  IteratorType outIt(test_lbl,test_lbl->GetRequestedRegion());
  outIt.GoToBegin();
  std::vector<int> tr_lbls;
  int num_lbls = lbls.size();
  std::vector<int> agr;
  unsigned const int kNumOfTrainingSamples = training_label_maps.size();
  std::vector<float> weighted_match(num_lbls);

  while (!outIt.IsAtEnd()) {
    tr_lbls.clear();
    for (const auto& training_label_map : training_label_maps)
      tr_lbls.push_back(training_label_map->GetPixel(outIt.GetIndex()));
    agr = MostRepeatedVal(tr_lbls,kNumOfTrainingSamples,lbls);
    if (agr[1] >=  round(agr_th*kNumOfTrainingSamples)) outIt.Set(agr[0]);
    else {
      weighted_match = {0,0};
      for (auto lbl : lbls) {
        int k = 0;
        for (auto tr_lbl : tr_lbls) {
          if (tr_lbl == lbl) weighted_match[lbl] += Wvec(k);
          // if (tr_lbl == lbl) weighted_match[lbl] += 1;
          k++;
        }
      }
      std::vector<float> max;
      max = Max(weighted_match);
      outIt.Set(max[1]);
    }
    ++outIt;
  }
}

arma::mat EigenToArma(const Eigen::MatrixXd& eigen_mat) {
  arma::mat arma_mat(eigen_mat.rows(),eigen_mat.cols());
  for (size_t i = 0; i != eigen_mat.rows(); i++)
    for (size_t j = 0; j != eigen_mat.cols(); j++) arma_mat(i,j) = eigen_mat(i,j);

  return arma_mat;
}

Mat ArmaToEigen(const arma::mat& arma_mat) {
  Mat eigen_mat(arma_mat.n_rows,arma_mat.n_cols);
  for (size_t i = 0; i != arma_mat.n_rows; i++)
    for (size_t j = 0; j != arma_mat.n_cols; j++) eigen_mat(i,j) = arma_mat(i,j);

  return eigen_mat;
}

void MedianImage(ImageType::Pointer& img, uint radius) {
  using MedianFilterType = itk::MedianImageFilter<ImageType, ImageType>;
  MedianFilterType::Pointer filter = MedianFilterType::New();
  filter->SetRadius(radius);
  filter->SetInput(img);
  filter->Update();
  img = filter->GetOutput();
}

void BiasCorrection(ImageType::Pointer& img, const CharImageType::Pointer& mask) {
  BiasCorrectionFilterType::Pointer filter = BiasCorrectionFilterType::New();
  filter->SetInput1(img);
  filter->SetInput2(mask);
  filter->SetConvergenceThreshold(0.00001);
  // filter->SetMaximumNumberOfIterations(30);
  filter->SetNumberOfHistogramBins(100);
  filter->Update();
  img = filter->GetOutput();
}
