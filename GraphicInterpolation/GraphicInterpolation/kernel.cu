#include <cstdlib>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NEW_IMAGE_SIZE_WIDTH 21 // 7 // 21
#define NEW_IMAGE_SIZE_HEIGHT 21 // 7 // 21
#define NEW_IMAGE_REPEAT_NUMBER 5

using namespace cv;
using namespace std;

/////////////////////////////////////////////////////////////////////////////////

void bilinearInterpolation(Mat &inputImage, Mat &outputImage)
{
	int blockSizeY = outputImage.rows / inputImage.rows;
	int blockSizeX = outputImage.cols / inputImage.cols;

	//problem z blokami krawêdziowymi
	for (int rowInd = blockSizeY; rowInd < outputImage.rows - blockSizeY; rowInd++)
	{
		for (int colInd = blockSizeX; colInd < outputImage.cols - blockSizeX; colInd++)
		{
			bool mask[3][3];
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					mask[i][j] = true;

			int myBlockY = rowInd / blockSizeY;
			int myBlockX = colInd / blockSizeX;

			int blockCenterRow = myBlockY * blockSizeY + blockSizeY / 2;
			int blockCenterCol = myBlockX * blockSizeX + blockSizeX / 2;

			///////////////wykreslanie blokow z maski
			////wiersze
			if (rowInd == blockCenterRow)
			{
				mask[0][0] = false;
				mask[0][1] = false;
				mask[0][2] = false;
				mask[2][0] = false;
				mask[2][1] = false;
				mask[2][2] = false;
			}
			else if (rowInd < blockCenterRow)
			{
				mask[2][0] = false;
				mask[2][1] = false;
				mask[2][2] = false;
			}
			else
			{
				mask[0][0] = false;
				mask[0][1] = false;
				mask[0][2] = false;
			}

			////kolumny
			if (colInd == blockCenterCol)
			{
				mask[0][0] = false;
				mask[1][0] = false;
				mask[2][0] = false;
				mask[0][2] = false;
				mask[1][2] = false;
				mask[2][2] = false;
			}
			else if (colInd < blockCenterCol)
			{
				mask[0][2] = false;
				mask[1][2] = false;
				mask[2][2] = false;
			}
			else
			{
				mask[0][0] = false;
				mask[1][0] = false;
				mask[2][0] = false;
			}

			for (int channel = 0; channel < outputImage.channels(); channel++)
			{
				/*
				outputImage.at<cv::Vec3b>(rowInd, colInd)[channel] =
					inputImage.at<cv::Vec3b>(myBlockY, myBlockX)[channel];
				*/

				double sum = 0.0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						if (mask[i][j] == true)
						{
							double maskPixelValue = inputImage.at<cv::Vec3b>(myBlockY - 1 + i, myBlockX - 1 + j)[channel];
							int maskPixelCenterRow = (myBlockY - 1 + i) * blockSizeY + blockSizeY / 2;
							int maskPixelCenterCol = (myBlockX - 1 + j) * blockSizeX + blockSizeX / 2;
							double Ydiff = ((double)(blockSizeY - abs(rowInd - maskPixelCenterRow))) /
								((double)blockSizeY);
							double Xdiff = ((double)(blockSizeX - abs(colInd - maskPixelCenterCol))) /
								((double)blockSizeX);
							sum += maskPixelValue * Ydiff * Xdiff;
						}
					}
				}

				outputImage.at<cv::Vec3b>(rowInd, colInd)[channel] = sum;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////

__device__ void CUDA_kernel_bilinearIntepolation_exec(
	uchar *d_imgIn, int d_imgIn_step, int d_imgIn_channels,
	uchar *d_imgOut, int d_imgOut_step,
	int d_blockSizeY, int d_blockSizeX,
	int rowInd, int colInd)
{
	bool mask[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			mask[i][j] = true;

	int myBlockY = rowInd / d_blockSizeY;
	int myBlockX = colInd / d_blockSizeX;

	int blockCenterRow = myBlockY * d_blockSizeY + d_blockSizeY / 2;
	int blockCenterCol = myBlockX * d_blockSizeX + d_blockSizeX / 2;

	///////////////wykreslanie blokow z maski
	////wiersze
	if (rowInd == blockCenterRow)
	{
		mask[0][0] = false;
		mask[0][1] = false;
		mask[0][2] = false;
		mask[2][0] = false;
		mask[2][1] = false;
		mask[2][2] = false;
	}
	else if (rowInd < blockCenterRow)
	{
		mask[2][0] = false;
		mask[2][1] = false;
		mask[2][2] = false;
	}
	else
	{
		mask[0][0] = false;
		mask[0][1] = false;
		mask[0][2] = false;
	}

	////kolumny
	if (colInd == blockCenterCol)
	{
		mask[0][0] = false;
		mask[1][0] = false;
		mask[2][0] = false;
		mask[0][2] = false;
		mask[1][2] = false;
		mask[2][2] = false;
	}
	else if (colInd < blockCenterCol)
	{
		mask[0][2] = false;
		mask[1][2] = false;
		mask[2][2] = false;
	}
	else
	{
		mask[0][0] = false;
		mask[1][0] = false;
		mask[2][0] = false;
	}

	for (int channel = 0; channel < d_imgIn_channels; channel++)
	{
		double sum = 0.0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (mask[i][j] == true)
				{
					//double maskPixelValue = inputImage.at<cv::Vec3b>(myBlockY - 1 + i, myBlockX - 1 + j)[channel];
					int input_tid = (myBlockY - 1 + i)*d_imgIn_step + (myBlockX - 1 + j)*d_imgIn_channels + channel;
					double maskPixelValue = d_imgIn[input_tid];
					int maskPixelCenterRow = (myBlockY - 1 + i) * d_blockSizeY + d_blockSizeY / 2;
					int maskPixelCenterCol = (myBlockX - 1 + j) * d_blockSizeX + d_blockSizeX / 2;
					double Ydiff = ((double)(d_blockSizeY - abs(rowInd - maskPixelCenterRow))) /
						((double)d_blockSizeY);
					double Xdiff = ((double)(d_blockSizeX - abs(colInd - maskPixelCenterCol))) /
						((double)d_blockSizeX);
					sum += maskPixelValue * Ydiff * Xdiff;
				}
			}
		}

		//outputImage.at<cv::Vec3b>(rowInd, colInd)[channel] = sum;
		int output_tid = rowInd * d_imgOut_step + colInd * d_imgIn_channels + channel;
		d_imgOut[output_tid] = sum;
	}
}

__global__ void CUDA_kernel_bilinearIntepolation(
	uchar *d_imgIn, int d_imgIn_step, int d_imgIn_channels,
	uchar *d_imgOut, int d_imgOut_step, int d_imgOut_rows, int d_imgOut_cols,
	int d_blockSizeY, int d_blockSizeX)
{
	//index of this pixel
	int initRowInd = (blockIdx.y * blockDim.y) + threadIdx.y; //height
	int initColInd = (blockIdx.x * blockDim.x) + threadIdx.x; //width

	int totalNumberOfThreadsInY = blockDim.y * gridDim.y;
	int totalNumberOfThreadsInX = blockDim.x * gridDim.x;

	//dzielenie z zaokr¹gleniem w góre
	int repeatsInY = (d_imgOut_rows + totalNumberOfThreadsInY - 1) / totalNumberOfThreadsInY;
	int repeatsInX = (d_imgOut_cols + totalNumberOfThreadsInX - 1) / totalNumberOfThreadsInX;

	for (int i = 0; i < repeatsInY; i++)
	{
		for (int j = 0; j < repeatsInX; j++)
		{
			int rowInd = initRowInd + i * totalNumberOfThreadsInY;
			int colInd = initColInd + j * totalNumberOfThreadsInX;

			//problem z blokami krawêdziowymi
			if (rowInd >= d_blockSizeY && rowInd < d_imgOut_rows - d_blockSizeY &&
				colInd >= d_blockSizeX && colInd < d_imgOut_cols - d_blockSizeX)
			{
				CUDA_kernel_bilinearIntepolation_exec(
					d_imgIn, d_imgIn_step, d_imgIn_channels,
					d_imgOut, d_imgOut_step,
					d_blockSizeY, d_blockSizeX,
					rowInd, colInd);
			}
		}
	}
}

cudaError_t CUDA_bilinearInterpolation(Mat inputImages[], Mat outputImages[], int numberOfImages)
{
	//cout << cv::getBuildInformation() << endl;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("AsyncEngineCount: %d\n", deviceProp.asyncEngineCount);

	cudaError_t cudaStatus;
	
	//Wybieranie GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	//Tworzenie streamów
	int numberOfStreams = 2;
	cudaStream_t streams[2];
	
	for (int i = 0; i < numberOfStreams; i++)
	{
		cudaStatus = cudaStreamCreate(&streams[i]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaStreamCreate returned error code %d after launching kernel!\n", cudaStatus);
			//goto Error;
		}
	}

	//Alokowanie macierzy z openCV na GPU
	uchar *h_imgInPin, *h_imgOutPin;
	uchar *d_imgIn, *d_imgOut;

	cudaStatus = cudaMallocHost((void**)&h_imgInPin, numberOfImages * inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	for (int i = 0; i < numberOfImages; i++)
	{
		memcpy(h_imgInPin + i * (inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar)),
			inputImages[0].data, inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar));
	}

	cudaStatus = cudaMallocHost((void**)&h_imgOutPin, outputImages[0].rows*outputImages[0].cols * outputImages[0].channels() * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_imgIn, inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_imgOut, outputImages[0].rows*outputImages[0].cols * outputImages[0].channels() * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//Wyliczanie potrzebnych stalych
	int d_imgIn_step = inputImages[0].step;
	int d_imgOut_step = outputImages[0].step;

	int blockSizeY = outputImages[0].rows / inputImages[0].rows;
	int blockSizeX = outputImages[0].cols / inputImages[0].cols;

	//Glowna petla programu - dla kazdego zdjecia wykonaj interpolacje

	//Pierwsze wyjatkowe kopiowanie dla pierwszego zdjecia
	int streamIndex = 0;
	
	cudaMemcpyAsync(d_imgIn, h_imgInPin, inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar),
		cudaMemcpyHostToDevice, streams[streamIndex]);

	for (int i = 0; i < numberOfImages; i++)
	{
		int nextStreamIndex = (streamIndex + 1) % numberOfStreams;

		//Wywolanie kernela
		const dim3 block(32, 32); //(16, 16);
		const dim3 grid(16, 16);

		CUDA_kernel_bilinearIntepolation<<<grid, block, 0, streams[streamIndex]>>>(
			d_imgIn, d_imgIn_step, inputImages[i].channels(),
			d_imgOut, d_imgOut_step, outputImages[i].rows, outputImages[i].cols,
			blockSizeY, blockSizeX);

		//Nie dla ostatniego zdjecia
		if (i+1 < numberOfImages)
		{
			//Laduj kolejne zdjecie
			cudaMemcpyAsync(d_imgIn, h_imgInPin + (i+1)*inputImages[i].rows*inputImages[i].cols * inputImages[i].channels() * sizeof(uchar), 
				inputImages[i+1].rows*inputImages[i+1].cols * inputImages[i+1].channels() * sizeof(uchar),
				cudaMemcpyHostToDevice, streams[nextStreamIndex]);
			
		}

		//Sprawdzenie czy wystapil blad w kernelu
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}
		
		//kopiowanie wyniku
		cudaMemcpyAsync(h_imgOutPin, d_imgOut, 
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar),
			cudaMemcpyDeviceToHost, streams[streamIndex]);
		
		streamIndex = nextStreamIndex;
	}

	//Czekanie na zakonczenie kerneli
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		//goto Error;
	}
	
	for (int i = 0; i < numberOfImages; i++)
	{
		memcpy(outputImages[i].data, h_imgOutPin, 
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
	}

	//CUDA FREE
//Error:
	cudaFreeHost(h_imgInPin);
	cudaFreeHost(h_imgOutPin);
	
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);

	for (int i = 0; i < numberOfStreams; i++)
		cudaStreamDestroy(streams[i]);

	return cudaStatus;
}

/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) 
{
    int Xresize, Yresize;
    string filename;
    int howManyRepeats;
    
    if(argc>1)
    {
        Xresize = atoi(argv[1]);
        Yresize = atoi(argv[2]);
        filename = argv[3];
        howManyRepeats = atoi(argv[4]);
    }
    else //domyœlnie
    {
        Xresize = NEW_IMAGE_SIZE_WIDTH;
        Yresize = NEW_IMAGE_SIZE_HEIGHT;
        filename = "./input/Lena.png";
        howManyRepeats = NEW_IMAGE_REPEAT_NUMBER;
    }
    
    printf("Arguments: %d %d %s %d \n", 
           Xresize, Yresize, filename.c_str(), howManyRepeats);

	Mat *imgsIn = new Mat[howManyRepeats];
	Mat *imgsOut = new Mat[howManyRepeats];

    for(int i=0; i<howManyRepeats; i++)
    {
		imgsIn[i] = imread(filename, CV_LOAD_IMAGE_COLOR);
		imgsOut[i] = Mat(Yresize*imgsIn[i].rows, Xresize*imgsIn[i].cols, imgsIn[i].type());
    }

	CUDA_bilinearInterpolation(imgsIn, imgsOut, howManyRepeats);

	char outputFilePath[] = "./output/result";
	char outputFileType[] = ".png";
	char fullFilename[100];
	for (int i = 0; i < howManyRepeats; i++)
	{
		sprintf(fullFilename,"%s%d%s",outputFilePath,i,outputFileType);
		imwrite(fullFilename, imgsOut[i]);
	}
	
	return 0;
}
