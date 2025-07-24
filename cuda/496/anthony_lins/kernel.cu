#include <cuda.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void rgb2gray(const unsigned char* rgb, unsigned char* gray, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ridx = i * 3;
        float r = rgb[ridx];
        float g = rgb[ridx + 1];
        float b = rgb[ridx + 2];
        gray[i] = (unsigned char)(0.2989f * r + 0.5870f * g + 0.1140f * b);
    }
}

int main(){
    int n;
    if (scanf("%d", &n) != 1) return 1;

    unsigned char *h_rgb = new unsigned char[n * 3];
    unsigned char *h_gray = new unsigned char[n];

    for (int i = 0; i < n * 3; i++) {
        int temp;
        if (scanf("%d", &temp) != 1) return 1;
        h_rgb[i] = (unsigned char)temp;
    }

    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, n * 3);
    cudaMalloc(&d_gray, n);
    cudaMemcpy(d_rgb, h_rgb, n * 3, cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rgb2gray<<<blocks, BLOCK_SIZE>>>(d_rgb, d_gray, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gray, d_gray, n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d\n", h_gray[i]);
    }

    delete[] h_rgb;
    delete[] h_gray;
    cudaFree(d_rgb);
    cudaFree(d_gray);
    return 0;
}
