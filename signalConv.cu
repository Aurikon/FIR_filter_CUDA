#include "cuda_runtime.h"
#include <vector>
#include <iostream>
#include "AudioFile/AudioFile.h"

#define FILTER_MAX_SIZE 1024
__constant__ float FILTER[FILTER_MAX_SIZE];


__global__ void sumLengthNoShared(float* signal, size_t signalLength, 
size_t filterLength, float* outSignal) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    float convSum = 0.0;
    if (pos < signalLength/* + filterLength - 1*/) {
        for (int i = 0; i < filterLength; ++i) {
            if (pos - i >= 0 && pos - i < signalLength) {
                convSum += FILTER[i] * signal[pos - i];
            }
        }
        outSignal[pos] = convSum;
    }
}

__global__ void sumLengthShared(float* signal, size_t signalLength,
size_t filterLength, float* outSignal) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedSignal[1024*8];  //[blockDim.x];
    __shared__ float prevSamplesShared[FILTER_MAX_SIZE - 1];//[filterLength - 1];
    
    
    int threadIx = threadIdx.x;
    if (pos < signalLength) {
        sharedSignal[threadIdx.x] = signal[pos];
    }
    if (threadIx - ((int)filterLength - 1) < 0 && pos - ((int)filterLength - 1) >= 0) {
        prevSamplesShared[threadIdx.x] = signal[pos - ((int)filterLength - 1)];
    }
    __syncthreads();

    if (pos < signalLength) {
        float convSum = 0.0f;
        for (int i = 0; i < filterLength; ++i) {
            if (threadIx - i >= 0) {
                convSum += FILTER[i] * sharedSignal[threadIx - i];
                // printf("%d %d %f\n", blockIdx.x, threadIdx.x, sharedSignal[threadIx - i]);
            }
            else {
                convSum += FILTER[i] * prevSamplesShared[(int)filterLength + (threadIdx.x - i) - 1];
                // printf("%d %d %f\n", blockIdx.x, threadIdx.x, prevSamplesShared[(int)filterLength + (threadIdx.x - i) - 1]);
                
            }
        }
        outSignal[pos] = convSum;
    }
}


// supose filter length is 32
__global__ void FIR(float* signal, size_t signalLength, size_t filterLength, 
                float* outSignal) {
    #define WARPSIZE 32
    #define LOG_WARP_SIZE 5

    auto pos = blockIdx.x * blockDim.x + threadIdx.x;
    auto laneId = threadIdx.x & (WARPSIZE - 1);
    auto warpId = threadIdx.x >> LOG_WARP_SIZE;


    for (int i = 0; i < 32; ++i) {
        float summand = signal[pos + i] * FILTER[threadIdx.x];

        float res = summand;
        res += __shfl_down_sync(0xFFFFFFFF, summand, 1);
        res += __shfl_down_sync(0x55555555, res, 2); 
        res += __shfl_down_sync(0x33333333, res, 4); 
        res += __shfl_down_sync(0x11111111, res, 8);
        res += __shfl_down_sync(0x80008000, res, 16);
        outSignal[pos + i] = res;
    }    
    // res += __shfl_down_sync(0xFFFFFFFF, res, 1);
    // res += __shfl_down_sync(0x55555555, res, 2);
    

    // #pragma unroll
    // for (int i = 1; i <= filterLength; i *= 2) {
    //     res += __shfl_down_sync(0xFFFFFFFF, res, i);
    // }
    
    // printf("%d %f\n", pos, res);
}

#ifndef DEBUG

int main() {


    AudioFile<float> audio;
    audio.load("audio.wav");


    for (int channel = 0; channel < audio.getNumChannels(); channel++) {
            size_t filterLength = 100;
            size_t signalLength = audio.getNumSamplesPerChannel();
            size_t outSignalLength = audio.getNumSamplesPerChannel();

            std::vector<float> signal(audio.samples[channel]);
            std::vector<float> filter/*(filterLength);*/ = {
               7.62939453125e-06, -0.0002288818359375, -0.00030517578125, -0.00048065185546875, -0.000701904296875, -0.0009613037109375, -0.00125885009765625, -0.00156402587890625, -0.0018768310546875, -0.00215911865234375, -0.0023956298828125, -0.00255584716796875, -0.00262451171875, -0.0025787353515625, -0.002410888671875, -0.0021209716796875, -0.0017242431640625, -0.0012359619140625, -0.00067901611328125, -9.918212890625e-05, 0.000457763671875, 0.0009613037109375, 0.0013580322265625, 0.00160980224609375, 0.00170135498046875, 0.00162506103515625, 0.00138092041015625, 0.00098419189453125, 0.00048065185546875, -8.392333984375e-05, -0.00064849853515625, -0.00115966796875, -0.00156402587890625, -0.00180816650390625, -0.001861572265625, -0.001708984375, -0.0013580322265625, -0.00084686279296875, -0.000213623046875, 0.00046539306640625, 0.001129150390625, 0.0016937255859375, 0.00209808349609375, 0.00228118896484375, 0.002227783203125, 0.0019073486328125, 0.00136566162109375, 0.000640869140625, -0.0001983642578125, -0.0010528564453125, -0.00183868408203125, -0.0024566650390625, -0.00283050537109375, -0.00290679931640625, -0.002655029296875, -0.0020904541015625, -0.00125885009765625, -0.00023651123046875, 0.0008697509765625, 0.00193023681640625, 0.002838134765625, 0.00347137451171875, 0.00374603271484375, 0.00360870361328125, 0.0030517578125, 0.0021209716796875, 0.00089263916015625, -0.0005035400390625, -0.00193023681640625, -0.00321197509765625, -0.00420379638671875, -0.0047760009765625, -0.0048370361328125, -0.0043487548828125, -0.00333404541015625, -0.0018768310546875, -0.00011444091796875, 0.00176239013671875, 0.00356292724609375, 0.00505828857421875, 0.006072998046875, 0.00646209716796875, 0.00612640380859375, 0.00506591796875, 0.00335693359375, 0.0011444091796875, -0.0013580322265625, -0.00386810302734375, -0.0061187744140625, -0.00782012939453125, -0.0087432861328125, -0.00872802734375, -0.0077056884765625, -0.005706787109375, -0.0028839111328125, 0.0005035400390625, 0.00412750244140625, 0.00757598876953125, 0.0104598999023438, 0.01239013671875, 0.0130615234375, 0.01226806640625, 0.00994873046875, 0.0062103271484375, 0.00131988525390625, -0.00431060791015625, -0.0101318359375, -0.0155105590820312, -0.0197982788085938, -0.0223388671875, -0.0225982666015625, -0.0201644897460938, -0.0148086547851562, -0.00653076171875, 0.004425048828125, 0.0176162719726562, 0.0323638916015625, 0.0478591918945312, 0.0631790161132812, 0.077362060546875, 0.089508056640625, 0.0988082885742188, 0.104652404785156, 0.106643676757812, 0.104652404785156, 0.0988082885742188, 0.089508056640625, 0.077362060546875, 0.0631790161132812, 0.0478591918945312, 0.0323638916015625, 0.0176162719726562, 0.004425048828125, -0.00653076171875, -0.0148086547851562, -0.0201644897460938, -0.0225982666015625, -0.0223388671875, -0.0197982788085938, -0.0155105590820312, -0.0101318359375, -0.00431060791015625, 0.00131988525390625, 0.0062103271484375, 0.00994873046875, 0.01226806640625, 0.0130615234375, 0.01239013671875, 0.0104598999023438, 0.00757598876953125, 0.00412750244140625, 0.0005035400390625, -0.0028839111328125, -0.005706787109375, -0.0077056884765625, -0.00872802734375, -0.0087432861328125, -0.00782012939453125, -0.0061187744140625, -0.00386810302734375, -0.0013580322265625, 0.0011444091796875, 0.00335693359375, 0.00506591796875, 0.00612640380859375, 0.00646209716796875, 0.006072998046875, 0.00505828857421875, 0.00356292724609375, 0.00176239013671875, -0.00011444091796875, -0.0018768310546875, -0.00333404541015625, -0.0043487548828125, -0.0048370361328125, -0.0047760009765625, -0.00420379638671875, -0.00321197509765625, -0.00193023681640625, -0.0005035400390625, 0.00089263916015625, 0.0021209716796875, 0.0030517578125, 0.00360870361328125, 0.00374603271484375, 0.00347137451171875, 0.002838134765625, 0.00193023681640625, 0.0008697509765625, -0.00023651123046875, -0.00125885009765625, -0.0020904541015625, -0.002655029296875, -0.00290679931640625, -0.00283050537109375, -0.0024566650390625, -0.00183868408203125, -0.0010528564453125, -0.0001983642578125, 0.000640869140625, 0.00136566162109375, 0.0019073486328125, 0.002227783203125, 0.00228118896484375, 0.00209808349609375, 0.0016937255859375, 0.001129150390625, 0.00046539306640625, -0.000213623046875, -0.00084686279296875, -0.0013580322265625, -0.001708984375, -0.001861572265625, -0.00180816650390625, -0.00156402587890625, -0.00115966796875, -0.00064849853515625, -8.392333984375e-05, 0.00048065185546875, 0.00098419189453125, 0.00138092041015625, 0.00162506103515625, 0.00170135498046875, 0.00160980224609375, 0.0013580322265625, 0.0009613037109375, 0.000457763671875, -9.918212890625e-05, -0.00067901611328125, -0.0012359619140625, -0.0017242431640625, -0.0021209716796875, -0.002410888671875, -0.0025787353515625, -0.00262451171875, -0.00255584716796875, -0.0023956298828125, -0.00215911865234375, -0.0018768310546875, -0.00156402587890625, -0.00125885009765625, -0.0009613037109375, -0.000701904296875, -0.00048065185546875, -0.00030517578125, -0.0002288818359375, 7.62939453125e-06
            };
            filterLength = filter.size();
            std::vector<float> outSignal(outSignalLength);
            for (int i = 0; i < filterLength; ++i) {
                // filter[i] = sin(0.5f*M_PI*(i - 40)) / (M_PI * (i - 40));
                // filter[i] = 1.0f / 10.0f;
            }

            std::cout << "Signal: ";
            for (auto elem : signal) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;

            dim3 threads(1024, 1, 1);
            dim3 blocks((outSignalLength + threads.x - 1)/threads.x, threads.y, threads.y);

            float* signalDevice;
            float* outDevice;
            
            auto cudaMallocStatus = cudaMalloc(&signalDevice, signalLength*sizeof(int));
            if (cudaMallocStatus != cudaSuccess) {
                std::cout << "Error in malloc1\n";
            }
            cudaMallocStatus = cudaMalloc(&outDevice, (outSignalLength)*sizeof(int));
            if (cudaMallocStatus != cudaSuccess) {
                std::cout << "Error in malloc2\n";
            }

            auto cudaMemcpyStatus = cudaMemcpy(signalDevice, signal.data(),
            signalLength*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaMemcpyStatus != cudaSuccess) {
                std::cout << "Error in memcpy1\n";
            }

            cudaMemcpyStatus = cudaMemcpyToSymbol(FILTER, filter.data(), sizeof(float)*filterLength);
            if (cudaMemcpyStatus != cudaSuccess) {
                std::cout << "Error in memcpy2\n";
            }
            sumLengthShared<<<blocks, threads>>>(signalDevice, signalLength, filterLength, outDevice);
            cudaMemcpyStatus = cudaMemcpy(outSignal.data(), 
                outDevice, outSignalLength * sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << cudaGetErrorString(cudaPeekAtLastError()) << "\n";
            cudaDeviceSynchronize();
            if (cudaMemcpyStatus != cudaSuccess) {
                std::cout << cudaGetErrorString(cudaMemcpyStatus) << "\n";
                std::cout << "Error in memcpy3\n";
            }

            audio.samples[channel] = outSignal;
            std::cout << "Output: ";
            for (auto elem : outSignal) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        
    }
    std::string outputFilePath = "shared.wav";
    audio.save (outputFilePath, AudioFileFormat::Wave);
    return 0;
}

#else

int main() {
    size_t filterLength = 3;
    size_t signalLength = 22;
    std::vector<float> signal(signalLength);
    for (int i = 0; i < signalLength; ++i) {
        signal[i] = (float)i;
    }
    std::reverse(signal.begin(), signal.end());

    std::vector<float> filter(filterLength);
    for (int i = 0; i < filterLength; ++i) {
        filter[i] = 1.0f;//(float)i / 1024.0f;
    }
    size_t outSignalLength = signalLength;
    std::vector<float> outSignal(outSignalLength);

    std::cout << "Signal: ";
    for (auto elem : signal) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    dim3 threads(15, 1, 1);
    dim3 blocks((outSignalLength + threads.x - 1)/threads.x, threads.y, threads.y);

    float* signalDevice;
    float* outDevice;

    cudaMalloc(&signalDevice, signalLength*sizeof(int));
    cudaMalloc(&outDevice, (outSignalLength)*sizeof(int));

    cudaMemcpy(signalDevice, signal.data(),
            signalLength*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(FILTER, filter.data(), sizeof(float)*filterLength);

    sumLengthShared<<<blocks, threads>>>(signalDevice, signalLength, filterLength, outDevice);
    cudaMemcpy(outSignal.data(), outDevice,
        outSignalLength * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << "\n";
    cudaDeviceSynchronize();

    // std::cout << "Output: ";
    // for (auto elem : outSignal) {
    //     std::cout << (int)elem << " ";
    // }
    // std::cout << "\n\n\n\n\n\n\n";


    // sumLengthNoShared<<<blocks, threads>>>(signalDevice, signalLength, filterLength, outDevice);
    // cudaMemcpy(outSignal.data(), outDevice,
    //     outSignalLength * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << cudaGetErrorString(cudaPeekAtLastError()) << "\n";
    // cudaDeviceSynchronize();

    // std::cout << "Output: ";
    // for (auto elem : outSignal) {
    //     std::cout << (int)elem << " ";
    // }
    // std::cout << std::endl;

    FIR<<<blocks, threads>>>(signalDevice, signalLength, filterLength, outDevice);
    cudaMemcpy(outSignal.data(), outDevice,
        outSignalLength * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Output: ";
    for (auto elem : outSignal) {
        std::cout << (int)elem << " ";
    }
    std::cout << "\n\n\n\n\n\n\n";
    return 0;
}

#endif