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

    if (pos < signalLength/* + filterLength - 1*/) {
        float convSum = 0.0f;
        for (int i = 0; i < filterLength; ++i) {
            if (threadIx - i >= 0) {
                convSum += FILTER[i] * sharedSignal[threadIx - i];
            }
            else {
                convSum += FILTER[i] * prevSamplesShared[threadIdx.x];
            }
        }
        outSignal[pos] = convSum;
    }
}

int main() {


    AudioFile<float> audio;
    audio.load("audio.wav");
    auto sampleRate = audio.getSampleRate();


    for (int channel = 0; channel < audio.getNumChannels(); channel++)
    {
            size_t filterLength = 100;
            size_t signalLength = audio.getNumSamplesPerChannel();
            size_t outSignalLength = audio.getNumSamplesPerChannel();

            std::vector<float> signal(audio.samples[channel]);
            std::vector<float> filter(filterLength);
            std::vector<float> outSignal(outSignalLength);
            for (int i = 0; i < filterLength; ++i) {
                filter[i] = 1.0f / 10.0f;
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
            sumLengthNoShared<<<blocks, threads>>>(signalDevice, signalLength, filterLength, outDevice);
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
    std::string outputFilePath = "filtered.wav";
    audio.save (outputFilePath, AudioFileFormat::Wave);
    return 0;
}