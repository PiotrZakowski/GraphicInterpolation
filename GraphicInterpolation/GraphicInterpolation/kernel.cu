#include <cstdlib>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NEW_IMAGE_SIZE_WIDTH 11 // 7 // 21
#define NEW_IMAGE_SIZE_HEIGHT 11 // 7 // 21
#define NEW_IMAGE_REPEAT_NUMBER 9

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
	
	printf("\nSetting GPU device...\n");
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "\t cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	else
		fprintf(stdout, "\t CUDA-capable GPU found.\n");

	printf("\nCreating streams...\n");
	int numberOfStreams = 2;
	cudaStream_t streams[2];
	
	for (int i = 0; i < numberOfStreams; i++)
	{
		cudaStatus = cudaStreamCreate(&streams[i]);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t cudaStreamCreate returned error code %d after launching kernel!\n", cudaStatus);
		else
			fprintf(stdout, "\t Stream created.\n");
	}

	printf("\nAllocating host memory for resources...\n");
	uchar **h_imgInPin, **h_imgOutPin;
	uchar *d_imgIn[2], *d_imgOut[2];

	h_imgInPin = (uchar**)malloc(numberOfImages*sizeof(uchar *));
	h_imgOutPin = (uchar**)malloc(numberOfImages*sizeof(uchar *));
	for (int i = 0; i < numberOfImages; i++)
	{
		printf("\t Allocating image nr %d...\n", i);
		cudaStatus = cudaMallocHost((void**)&h_imgInPin[i], inputImages[i].rows*inputImages[i].cols * inputImages[i].channels() * sizeof(uchar), cudaHostAllocPortable);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t\t cudaMalloc failed!\n");
		else
			fprintf(stdout, "\t\t Memory for input image allocated.\n");

		memcpy(h_imgInPin[i], inputImages[i].data, inputImages[i].rows*inputImages[i].cols * inputImages[i].channels() * sizeof(uchar));

		cudaStatus = cudaMallocHost((void**)&h_imgOutPin[i], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar), cudaHostAllocPortable);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t\t cudaMalloc failed!\n");
		else
			fprintf(stdout, "\t\t Memory for output image allocated.\n");
	}

	printf("\nProcesing images\n");
	int streamIndex = 0;
	for (int i = 0; i < numberOfImages; i++)
	{
		printf("\nProcessing image nr %d.\n", i+1);
		int nextStreamIndex = (streamIndex + 1) % numberOfStreams;
		
		printf("Allocating memory on GPU for resources...\n");
		if (i + 1 != numberOfImages)
		{
			cudaStatus = cudaMalloc((void**)&d_imgIn[1], inputImages[i + 1].rows*inputImages[i + 1].cols * inputImages[i + 1].channels() * sizeof(uchar));
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "\t cudaMalloc failed!\n");
			else
				fprintf(stdout, "\t Memory for input image allocated.\n");
		}

		cudaStatus = cudaMalloc((void**)&d_imgOut[0], outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t cudaMalloc failed!\n");
		else
			fprintf(stdout, "\t Memory for output image allocated.\n");

		if (i == 0)
		{
			printf("Handling up first image...\n");
			cudaStatus = cudaMalloc((void**)&d_imgIn[0], inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar));
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "\t cudaMalloc failed!\n");
			else
				fprintf(stdout, "\t Memory for input image allocated.\n");
			cudaMemcpyAsync(d_imgIn[0], h_imgInPin[0], inputImages[0].rows*inputImages[0].cols * inputImages[0].channels() * sizeof(uchar),
				cudaMemcpyHostToDevice, streams[streamIndex]);
		}

		printf("Launching computations...\n");
		int d_imgIn_step = inputImages[i].step;
		int d_imgOut_step = outputImages[i].step;

		int blockSizeY = outputImages[i].rows / inputImages[i].rows;
		int blockSizeX = outputImages[i].cols / inputImages[i].cols;

		
		const dim3 block(32, 32); //(16, 16);
		const dim3 grid(16, 16);

		CUDA_kernel_bilinearIntepolation<<<grid, block, 0, streams[streamIndex]>>>(
			d_imgIn[0], d_imgIn_step, inputImages[i].channels(),
			d_imgOut[0], d_imgOut_step, outputImages[i].rows, outputImages[i].cols,
			blockSizeY, blockSizeX);

		if (i+1 < numberOfImages)
		{
			printf("\t Prefetching next image.\n");
			cudaStatus = cudaMemcpyAsync(d_imgIn[1], h_imgInPin[i+1],
				inputImages[i+1].rows*inputImages[i+1].cols * inputImages[i+1].channels() * sizeof(uchar),
				cudaMemcpyHostToDevice, streams[nextStreamIndex]);
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "\t\t cudaMemcpyAsync failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		//printf("Sprawdzenie czy wystapil blad w kernelu\n");
		printf("\t Checking computations status...\n");
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t\t kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		else
			fprintf(stdout, "\t\t Computations completed.\n");
		
		cudaFree(d_imgIn[0]);
		if (i + 1 < numberOfImages)
			d_imgIn[0] = d_imgIn[1];

		if(i!=0)
			cudaFree(d_imgOut[1]);
		d_imgOut[1] = d_imgOut[0];

		//printf("Kopiowanie wyniku\n");
		printf("Coping output from device to host.\n");
		cudaStatus = cudaMemcpyAsync(h_imgOutPin[i], d_imgOut[1],
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar),
			cudaMemcpyDeviceToHost, streams[streamIndex]);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "\t cudaMemcpyAsync failed: %s\n", cudaGetErrorString(cudaStatus));

		streamIndex = nextStreamIndex;
	}

	printf("\nWaiting for computations to end...\n");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "\t cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	else
		fprintf(stdout, "\t Computations ended.\n");

	printf("\nReading outputs...\n");
	for (int i = 0; i < numberOfImages; i++)
	{
		memcpy(outputImages[i].data, h_imgOutPin[i],
			outputImages[i].rows*outputImages[i].cols * outputImages[i].channels() * sizeof(uchar));
	}
	printf("\t Outputs readed.\n");

	printf("\nFreeing allocated memory...\n");
	free(h_imgInPin);
	free(h_imgOutPin);
	cudaFree(d_imgOut[1]);

	for (int i = 0; i < numberOfImages; i++)
	{
		cudaFreeHost(h_imgInPin[i]);
		cudaFreeHost(h_imgOutPin[i]);
	}

	for (int i = 0; i < numberOfStreams; i++)
		cudaStreamDestroy(streams[i]);

	printf("\t Memory freed\n");
	return cudaStatus;
}

/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) 
{
	clock_t tStart = clock();

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
    else //domyslnie
    {
        Xresize = NEW_IMAGE_SIZE_WIDTH;
        Yresize = NEW_IMAGE_SIZE_HEIGHT;
        filename1 = "./input/Lena.png";
		filename2 = "./input/icon.png";
        howManyImages = NEW_IMAGE_REPEAT_NUMBER;
    }
    
    printf("Arguments: %d %d %s %s %d \n", 
           Xresize, Yresize, filename1.c_str(), filename2.c_str(), howManyImages);

	Mat *imgsIn = new Mat[howManyImages];
	Mat *imgsOut = new Mat[howManyImages];

    for(int i=0; i<howManyImages; i++)
    {
		imgsIn[i] = imread(i%2==0?filename1:filename2, CV_LOAD_IMAGE_COLOR);
		//Mat dst;
        //normalize(imgsIn[i], dst, 0, 1, cv::NORM_MINMAX);
        //imshow("test", dst);
        //waitKey(0);
		imgsOut[i] = Mat(Yresize*imgsIn[i].rows, Xresize*imgsIn[i].cols, imgsIn[i].type());
    }

	CUDA_bilinearInterpolation(imgsIn, imgsOut, howManyImages);

	char outputFilePath[] = "./output/result";
	char outputFileType[] = ".png";
	char fullFilename[100];
	for (int i = 0; i < howManyImages; i++)
	{
		sprintf(fullFilename,"%s%d%s",outputFilePath,i,outputFileType);
		try
		{
			imwrite(fullFilename, imgsOut[i]);
		}
		catch (const Exception e)
		{
			printf("%s", e.msg);
		}
		imgsIn[i].release();
		imgsOut[i].release();
	}
	
	double executionTime = (clock() - tStart) / (double)CLOCKS_PER_SEC;
	int minutes = (int)executionTime / 60;
	printf("\nExecution time: %dm%.3fs\n", minutes, executionTime - (double)(minutes * 60));

	delete[] imgsIn;
	delete[] imgsOut;
	return 0;
}
