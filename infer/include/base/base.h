#ifndef INFER_INCLUDE_BASE_H
#define INFER_INCLUDE_BASE_H
#include "glog/logging.h"
#include <cstring>
#include <string>

#define ANA_MEM_TIME 1
#ifdef ANA_MEM_TIME
#include <iostream>
#include <chrono>
#include <algorithm>
#include <chrono>
#include <stack>
#include <map>

#include <iomanip>
#include <nvml.h> // -lnvidia-ml
using hsClock = std::chrono::high_resolution_clock;
using microSec = std::chrono::microseconds;
#endif


#define UNUSED(expr) \
    do {               \
      (void)(expr);    \
    } while (0)

#define CEIL(a,b) ((a+b-1)/b)

namespace model {
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kForwardOutput = 15,
  kForwardOutputCPU = 16,

  kSinCache = 17,
  kCosCache = 18,
};
}

namespace base{
enum class DeviceType: uint8_t{
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

enum class MemcpyKind{
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDACUDA = 3,
};

enum class TokenizerType {
  kEncodeUnknown = -1,
  kEncodeSpe = 0,
  kEncodeBpe = 1,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
  kModelTypeQwen3 = 2,
};

inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
      return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
      return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
      return sizeof(int32_t);
    } else {
      return 0;
    }
}

class NoCopyable {
protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};


class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};


namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)
Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

// std::ostream& operator<<(std::ostream& os, const Status& x);// TODO


#ifdef ANA_MEM_TIME

//!
//! \class nvtxTimeTable
//!
//! \brief time trace usage:
//!
//!    nvtxTimeTable nvTT;
//!    nvTT.time_push("abcd");
//!    nvTT.time_pop("abcd");
//!    nvTT.print_avg_stats();
class nvtxTimeTable{
public:
    std::stack<std::pair<std::string, hsClock::time_point>> timeStack;
    std::map<std::string, microSec> timeTable;
    std::vector<std::string> tagOrder;
    nvtxTimeTable() = default; // 默认构造函数
    
    void time_push(const std::string& tag){
      if(timeTable.find(tag) == timeTable.end()){
        timeTable[tag] = microSec(0);
        tagOrder.push_back(tag);
      }
      timeStack.push({tag, hsClock::now()});
    };

    void time_pop(const std::string& tag){
        if(timeStack.empty()){
            printf("[nvTT] empty error timeTable.\n");
            return; 
        }
        microSec ts = std::chrono::
            duration_cast<microSec>(hsClock::now()- timeStack.top().second);
        timeTable[timeStack.top().first] += ts;
        timeStack.pop();     
    };

    void print_avg_stats(){
        printf("=== AVG (us):\n");
        bool first = true;
        for(auto& tag : tagOrder){
            if(!first) printf(", ");
             printf("%s", tag.c_str());
            first = false;
        }
        printf(": ");
        first = true;
        for(auto& tag : tagOrder){
            if(!first) printf(", ");
            printf("%ld", timeTable[tag].count());
            first = false;
        }
        printf("\n");
    };
};
extern nvtxTimeTable nvTT;

//!
//! \class memoryProfiler
//!
//! \brief needed: #include <nvml.h>
//!
class memoryProfiler
{
private:
    nvmlDevice_t device;
    int gpu_id = 0;
    std::vector<std::pair<std::string, float>> memoryTable;
public:
    memoryProfiler(int device_id=0): gpu_id(device_id){
        nvmlInit();
        nvmlDeviceGetHandleByIndex(gpu_id, &device);
    };
    ~memoryProfiler() {
        nvmlShutdown();
    }
    
    // 入口名称建议 afOP, 即表示 OP 完成后的显存状态
    void mem_push(const std::string& tag) {
        nvmlMemory_t mem_info;
        nvmlDeviceGetMemoryInfo(device, &mem_info);
        memoryTable.push_back({tag, mem_info.used / (1024 * 1024)});// MB
    }
    void print_mem_info(){
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Memory Usage Summary:" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        int n = memoryTable.size();
        for(int i=0;i<n;i++){
            if (i > 0) {
                float prev = memoryTable[i-1].second;
                float cur = memoryTable[i].second;
                float delta = cur - prev;
                char sign = delta >= 0 ? '+' : ' ';
                
                std::cout << std::left << std::setw(20) << std::setfill(' ') << memoryTable[i].first << ": "
                          << std::right << std::fixed << std::setprecision(4) 
                          << std::setw(8) << cur << " MB ("
                          << sign << std::setw(6) << std::fixed << std::setprecision(2) 
                          << delta << " MB)" << std::endl;
            } else {
                std::cout << std::left << std::setw(20) << std::setfill(' ') << memoryTable[i].first << ": "
                          << std::right << std::fixed << std::setprecision(4) 
                          << std::setw(8) << memoryTable[i].second << " MB" << std::endl;
            }
        }
    }
};
extern nvtxTimeTable mP;
#endif


} // namespace base
#endif // INFER_INCLUDE_BASE_H