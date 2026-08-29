#include "bits/stdc++.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

using namespace std;

int main(int argc, char *argv[]){
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("infer_cpp");
    FLAGS_log_dir = "./log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";
    return RUN_ALL_TESTS();
}