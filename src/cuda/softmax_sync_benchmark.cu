#include "kernel_lab/cuda_kernels.hpp"
#include "kernel_lab/softmax.hpp"
#include "kernel_lab/softmax_workspace.hpp"
#include "kernel_lab/validation.hpp"
#include "device_ops.cuh"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>

namespace {
using namespace kernel_lab;
using Clock = std::chrono::steady_clock;
constexpr int repeats = 24, warmup = 5;
constexpr unsigned seed = 20260829U;
const char* names[] = {"warp_wrapper", "workspace_staged", "workspace_kernel_sync", "workspace_copy_sync"};
constexpr int device_waits[] = {1,2,1,0}; // source-level calls, not profiler counts
const SoftmaxSyncMode modes[] = {SoftmaxSyncMode::Staged, SoftmaxSyncMode::KernelBoundary, SoftmaxSyncMode::CopyCompletion};
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    const auto n=values.size(); return n%2 ? values[n/2] : (values[n/2-1]+values[n/2])/2;
}
std::ofstream open_csv(const std::filesystem::path& directory,const char* filename) {
    const auto path=directory/filename;
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite sync results");
    std::ofstream output(path);
    if (!output) throw std::runtime_error("output directory missing/unwritable");
    output.imbue(std::locale::classic()); output<<std::setprecision(12); return output;
}
void run_case(std::size_t rows,std::size_t cols,std::ofstream& summary,std::ofstream& raw,std::ofstream& setup) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-2,2);
    std::vector<float> input(rows*cols);
    for (auto& v:input) v=distribution(generator);
    const auto reference=softmax_reference(input,rows,cols);
    std::array<double,4> errors{};
    const auto verify=[&](int v,const std::vector<float>& output) {
        errors[v]=std::max(errors[v],require_close(output,reference,2e-6,2e-5,names[v]));
        require_probability_rows(output,rows,cols);
    };
    const auto preparation=Clock::now();
    SoftmaxWorkspace workspace;
    const auto first=workspace.run(input,rows,cols); // initial staged allocation is excluded from steady timing
    const auto first_ms=elapsed(preparation);
    verify(1,first);
    setup<<rows<<','<<cols<<','<<first_ms<<','<<workspace.capacity()<<','<<workspace.allocation_events()<<','<<seed<<'\n';
    const auto call=[&](int variant) {
        if (!variant) return softmax_cuda_warp_register(input,rows,cols);
        return workspace.run(input,rows,cols,modes[variant-1]);
    };
    for (int i=0;i<warmup;++i) for(int v=0;v<4;++v) verify(v,call(v));
    const auto before=workspace.allocation_events();
    std::array<std::vector<double>,4> samples;
    for (int sample=0;sample<repeats;++sample) {
        for (int slot=0;slot<4;++slot) {
            const int v=(sample+slot)%4;
            const auto begin=Clock::now();
            auto output=call(v);
            const double ms=elapsed(begin);
            // CPU consumes every returned element with no external CUDA wait.
            verify(v,output);
            if (!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid elapsed time");
            samples[v].push_back(ms);
            raw<<rows<<','<<cols<<','<<names[v]<<','<<sample<<','<<slot<<','<<ms<<','<<seed<<'\n';
        }
    }
    if(before!=1||workspace.allocation_events()!=before) throw std::runtime_error("sync comparison reallocated");
    for (int v=0;v<4;++v) {
        const auto& values=samples[v]; const auto ms=median(values);
        summary<<rows<<','<<cols<<','<<names[v]<<",end_to_end_wall,"<<ms<<','
               <<*std::min_element(values.begin(),values.end())<<','<<*std::max_element(values.begin(),values.end())
               <<','<<median(samples[0])/ms<<','<<median(samples[1])/ms<<','<<errors[v]
               <<",2e-6,2e-5,"<<warmup<<','<<repeats<<','<<seed<<','<<device_waits[v]<<",PASS\n";
    }
    std::cout<<"SYNC_BENCH_CASE_PASS "<<rows<<'x'<<cols<<" reuse_growths=0 cpu_ready=PASS\n";
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=3||std::string(argv[1])!="--output-dir") throw std::invalid_argument("usage: softmax_sync_benchmark --output-dir EXISTING_EMPTY_DIR");
        device::check(cudaSetDevice(0),"sync benchmark device");
        device::check(cudaFree(nullptr),"initialize CUDA context");
        const std::filesystem::path directory(argv[2]);
        auto summary=open_csv(directory,"sync_summary.csv"),raw=open_csv(directory,"sync_samples.csv"),setup=open_csv(directory,"sync_setup.csv");
        summary<<"rows,cols,implementation,scope,median_ms,min_ms,max_ms,speedup_vs_wrapper,speedup_vs_staged,max_abs_error,atol,rtol,warmup,repeats,seed,explicit_device_waits,correctness\n";
        raw<<"rows,cols,implementation,sample,order_slot,elapsed_ms,seed\n";
        setup<<"rows,cols,first_call_ms,capacity,allocation_events,seed\n";
        const std::pair<std::size_t,std::size_t> shapes[]={{1,1},{1,128},{7,37},{32,128},{1024,128},{1024,257},
            {1024,512},{1024,1024},{1024,2048},{1024,4097}};
        for(const auto& shape:shapes) run_case(shape.first,shape.second,summary,raw,setup);
        for(auto* stream:{&summary,&raw,&setup}) {stream->flush();if(!*stream) throw std::runtime_error("CSV write failed");}
        std::cout<<"SYNC_BENCH_PASS summary=40 raw=960 setup=10\n"; return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
