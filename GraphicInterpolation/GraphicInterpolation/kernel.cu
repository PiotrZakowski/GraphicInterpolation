#include <cstdlib>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NEW_IMAGE_SIZE_WIDTH 5 // 7 // 21
#define NEW_IMAGE_SIZE_HEIGHT 5 // 7 // 21
#define NEW_IMAGE_REPEAT_NUMBER 6

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

	uchar *h_imgInPin[2], **h_imgOutPin;
	uchar *d_imgIn[2], *d_imgOut[2];

	//przygotowanie
	h_imgOutPin = (uchar**)malloc(numberOfImages);
	for (int i = 0; i < numberOfImages; i++)
	{
		cudaStatus = cudaMallocHost((void**)&h_imgOutPin[i], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			//goto Error;
		}
	}

	//Glowna petla programu - dla kazdego zdjecia wykonaj interpolacje
	int streamIndex = 0;

	for (int i = 0; i < numberOfImages; i++)
	{
		int nextStreamIndex = (streamIndex + 1) % numberOfStreams;
		
		//Alokowanie macierzy z openCV na GPU
		if (i + 1 != numberOfImages)
		{
			cudaStatus = cudaMallocHost((void**)&h_imgInPin[nextStreamIndex], inputImages[i + 1].rows*inputImages[i + 1].cols * inputImages[i + 1].channels() * sizeof(uchar));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				//goto Error;
			}
			memcpy(h_imgInPin[nextStreamIndex], inputImages[i + 1].data, inputImages[i + 1].rows*inputImages[i + 1].cols * inputImages[i + 1].channels() * sizeof(uchar));

			cudaStatus = cudaMalloc((void**)&d_imgIn[1], inputImages[i + 1].rows*inputImages[i + 1].cols * inputImages[i + 1].channels() * sizeof(uchar));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				//goto Error;
			}
		}

		cudaStatus = cudaMalloc((void**)&d_imgOut[0], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			//goto Error;
		}

		//Pierwsze wyjatkowe kopiowanie dla pierwszego zdjecia
		if (i == 0)
		{
			cudaStatus = cudaMalloc((void**)&d_imgIn[i], inputImages[i].rows*inputImages[i].cols * inputImages[i].channels() * sizeof(uchar));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				//goto Error;
			}
			cudaMemcpy(d_imgIn[i], inputImages[i].data, inputImages[i].rows*inputImages[i].cols * inputImages[i].channels() * sizeof(uchar),
				cudaMemcpyHostToDevice);
		}

		//Wyliczanie potrzebnych stalych
		int d_imgIn_step = inputImages[i].step;
		int d_imgOut_step = outputImages[i].step;

		int blockSizeY = outputImages[i].rows / inputImages[i].rows;
		int blockSizeX = outputImages[i].cols / inputImages[i].cols;

		//Wywolanie kernela
		const dim3 block(32, 32); //(16, 16);
		const dim3 grid(16, 16);

		CUDA_kernel_bilinearIntepolation<<<grid, block, 0, streams[streamIndex]>>>(
			d_imgIn[0], d_imgIn_step, inputImages[i].channels(),
			d_imgOut[0], d_imgOut_step, outputImages[i].rows, outputImages[i].cols,
			blockSizeY, blockSizeX);

		//Nie dla ostatniego zdjecia
		if (i+1 < numberOfImages)
		{
			//Laduj kolejne zdjecie
			cudaStatus = cudaMemcpyAsync(d_imgIn[1], h_imgInPin[nextStreamIndex],
				inputImages[i+1].rows*inputImages[i+1].cols * inputImages[i+1].channels() * sizeof(uchar),
				cudaMemcpyHostToDevice, streams[nextStreamIndex]);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
		}

		//Sprawdzenie czy wystapil blad w kernelu
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}
		
		cudaFree(d_imgIn[0]);
		/**/
		cudaStatus = cudaMalloc((void**)&d_imgIn[0], inputImages[i+1].rows*inputImages[i+1].cols * inputImages[i+1].channels() * sizeof(uchar));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			//goto Error;
		}
		cudaMemcpy(d_imgIn[0], d_imgIn[1], inputImages[i+1].rows*inputImages[i+1].cols * inputImages[i+1].channels() * sizeof(uchar),
			cudaMemcpyDeviceToDevice);
		cudaFree(d_imgIn[1]);
		/**/
		/*
		d_imgIn[0] = d_imgIn[1];
		*/
		if(i!=0)
			cudaFree(d_imgOut[1]);
		/**/
		cudaStatus = cudaMalloc((void**)&d_imgOut[1], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			//goto Error;
		}
		cudaMemcpy(d_imgOut[1], d_imgOut[0], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar),
			cudaMemcpyDeviceToDevice);
		cudaFree(d_imgOut[0]);
		/**/
		/*
		d_imgOut[1] = d_imgOut[0];
		*/
		//kopiowanie wyniku
		cudaStatus = cudaMemcpyAsync(h_imgOutPin[i], d_imgOut[1],
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar),
			cudaMemcpyDeviceToHost, streams[streamIndex]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		if(i!=0)
			cudaFreeHost(h_imgInPin[streamIndex]);

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
		memcpy(outputImages[i].data, h_imgOutPin[i],
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
		cudaFreeHost(h_imgOutPin[i]);
	}

	//CUDA FREE
//Error:
	//free(h_imgOutPin);
	cudaFree(d_imgIn[1]);
	cudaFree(d_imgOut[1]);

	for (int i = 0; i < numberOfStreams; i++)
		cudaStreamDestroy(streams[i]);

	cudaStatus = cudaDeviceSynchronize();
	return cudaStatus;
}

/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) 
{
    int Xresize, Yresize;
    string filename1, filename2;
    int howManyImages;
    
    if(argc>1)
    {
        Xresize = atoi(argv[1]);
        Yresize = atoi(argv[2]);
        filename1 = argv[3];
		filename2 = argv[4];
        howManyImages = atoi(argv[5]);
    }
    else //domyœlnie
    {
        Xresize = NEW_IMAGE_SIZE_WIDTH;
        Yresize = NEW_IMAGE_SIZE_HEIGHT;
        filename1 = "./input/Lena.png";
		filename2 = "./input/Icon.png";
        howManyImages = NEW_IMAGE_REPEAT_NUMBER;
    }
    
    printf("Arguments: %d %d %s %s %d \n", 
           Xresize, Yresize, filename1.c_str(), filename2.c_str(), howManyImages);

	Mat *imgsIn = new Mat[howManyImages];
	Mat *imgsOut = new Mat[howManyImages];

    for(int i=0; i<howManyImages; i++)
    {
		imgsIn[i] = imread(i%2==0?filename1:filename2, CV_LOAD_IMAGE_COLOR);
		imgsOut[i] = Mat(Yresize*imgsIn[i].rows, Xresize*imgsIn[i].cols, imgsIn[i].type());
    }

	CUDA_bilinearInterpolation(imgsIn, imgsOut, howManyImages);

	char outputFilePath[] = "./output/result";
	char outputFileType[] = ".png";
	char fullFilename[100];
	for (int i = 0; i < howManyImages; i++)
	{
		sprintf(fullFilename,"%s%d%s",outputFilePath,i,outputFileType);
		imwrite(fullFilename, imgsOut[i]);
	}
	
	return 0;
}
