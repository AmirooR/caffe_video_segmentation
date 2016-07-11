#include "caffe/util/tracker_math.hpp"
#include "caffe/util/math_functions.hpp"
#include <iomanip>
#include <fstream>
#include "caffe/syncedmem.hpp"

namespace caffe {
  

template <typename Dtype>
void tracker_printMat(std::ostream& buffer, const Dtype* mat, int col, int count) {
  buffer << std::fixed << std::setprecision(5);
  for (int i = 0; i < count; ++i) {
    if(i != 0 && i % col == 0) {
      buffer << ';' << std::endl;
    }
    buffer << std::setw(10) << mat[i] << ' ';
  }
  buffer << std::endl;
}


template void tracker_printMat<double>(std::ostream& buffer, const double* mat, int col, int count);
template void tracker_printMat<float>(std::ostream& buffer, const float* mat, int col, int count);
template void tracker_printMat<int>(std::ostream& buffer, const int* mat, int col, int count);


template <typename Dtype>
void tracker_saveMat(string filename, const Dtype* mat, int col, int count) {
  std::ofstream outfile;
  outfile.open(filename.c_str(), std::ios::out | std::ios::binary);
  outfile.write((char*)&col, sizeof(int));
  outfile.write((char*)&count, sizeof(int));
  outfile.write((char*)mat, count*sizeof(Dtype));
  outfile.close();
}


template void tracker_saveMat<float>(string filename, const float* mat, int col, int count);
template void tracker_saveMat<double>(string filename, const double* mat, int col, int count);
template void tracker_saveMat<int>(string filename, const int* mat, int col, int count);
            
}  // namespace caffe
