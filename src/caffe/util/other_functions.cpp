#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/other_functions.hpp"


namespace caffe {

bool pair_ascend(const PairDist& a, const PairDist& b)
{
    return a.dist < b.dist;
}
bool pair_descend(const PairDist& a, const PairDist& b)
{
    return a.dist > b.dist;
}


} // end the namespace caffe

