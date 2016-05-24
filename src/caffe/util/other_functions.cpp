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

int pair_ascend_qsort(const void * a, const void * b)
{
    PairDist aa = *((PairDist*)a);
    PairDist bb = *((PairDist*)b);
    if (aa.dist > bb.dist)
    {
        return 1;
    }
    if (aa.dist == bb.dist)
    {
        return 0;
    }
    return -1;
}
int pair_descend_qsort(const void * a, const void * b)
{
    PairDist aa = *((PairDist*)a);
    PairDist bb = *((PairDist*)b);
    if (aa.dist < bb.dist)
    {
        return 1;
    }
    if (aa.dist == bb.dist)
    {
        return 0;
    }
    return -1;
}


} // end the namespace caffe

