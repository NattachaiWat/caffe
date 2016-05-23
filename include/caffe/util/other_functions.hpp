#ifndef CAFFE_UTIL_OTHER_FUNCTIONS_H_
#define CAFFE_UTIL_OTHER_FUNCTIONS_H_

#include <iostream>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

namespace caffe {

// add the cell for online pair hard negative mining
typedef struct Pair
{
    float dist;
    int first;
    int second;
    int flag;
} PairDist;
bool pair_ascend(const PairDist& a, const PairDist& b);
bool pair_descend(const PairDist& a, const PairDist& b);

int pair_ascend_qsort(const void * a, const void * b);
int pair_descend_qsort(const void * a, const void * b);



} // end namespance caffe

#endif // end the other functions
