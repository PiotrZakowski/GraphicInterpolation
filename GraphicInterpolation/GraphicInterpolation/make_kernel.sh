rm kernel.cu.o
rm kernel

/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc -m64 -o kernel.cu.o -c kernel.cu `pkg-config opencv --cflags --libs` -L/usr/lib/x86_64-linux-gnu/libopencv_core
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -o kernel kernel.cu.o `pkg-config opencv --cflags --libs` -L/usr/lib/x86_64-linux-gnu/libopencv_core
