**rupMC**

Version 1.0

**Overview**

The Marching Cubes (MC) algorithm is widely used for extracting iso-surface from volume data and 3D visualization because of its effectiveness and robustness. However, MC requires massive memory space and extensive computing time when dealing with large-scale applications. In addition, the iso-surface generated by MC lacks of topologic information, making it hard to be used in geologic applications, such as visualization of the geological models and simulation of the microstructures in rocks. 

This study proposes an enhanced MC, called ray-unit parallel Marching Cubes (rupMC) algorithm, to address the aforementioned limitations of MC. First, ray-units are used in rupMC as the basic voxel to determine how the surface intersects for reducing the repeated computations and enhancing the efficiency. Second, rupMC uses multiple computing processes and threads on a CPU/GPU heterogeneous architecture to process the points concurrently. Third, based on ray-units, the unique indices of the surface intersection are preserved to compose the surface triangles, and the topologic information of the surface are directly embedded by the triangle compositions. Experiments on five stratum datasets of various sizes showed that compared with VTK-implemented MC and C++-implemented MC, rupMC reduced the repeated surface intersections and generated the same surface triangles with unique intersection indices. The results also showed that rupMC achieved approximately 50 and 160 speed-up compared with VTK-implemented MC and C++-implemented MC. rupMC is highly scalable and adaptive for various CPUs/GPUs and datasets of varying sizes. rupMC is capable of extracting accurate surface intersections and triangles for large-scale and high-density applications with high efficiency and feasibility.

**Key features of rupMC**

- Supports a wide range of MPI-enabled CPUs and CUDA-enabled GPUs (https://developer.nvidia.com/cuda-gpus)
  - Automatic setting of the numbers of GPUs according to the available GPUs in the computing environment
  - Automatic setting of the numbers of threads and thread blocks according to the GPU’s available computing resources (e.g., memory, streaming multiprocessors, and warp)
  - Adaptive cyclic task assignment to achieve better load balance
  - Adaptive data domain decomposition when the size of images and temporary products exceeds the GPU’s memory
  - All above are completely transparent to users
- Supports both Windows and Linux/Unix operating systems

**References**

- Lorensen, W.E., Cline, H.E., 1987. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. Presented at the ACM SIGGRAPH Computer Graphics, pp. 163–169. https://doi.org/10.1145/37401.37422

**To Cite rupMC in Publications**

- Please cite the following reference:
- Yang, X., Yun, S., Guan, Q., Gao, H., 2024. rupMC: a ray-unit parallel marching cubes algorithm on CPU/GPU heterogeneous architectures. International Journal of Digital Earth, doi: 10.1080/17538947.2024.2340583.
- You may contact the e-mail aurora@cug.edu.cn if you have further questions about the usage of codes and datasets.
- For any possible research collaboration, please contact Prof. Qingfeng Guan (guanqf@cug.edu.cn).

**Compilation**

- Requirements:
  - A computer with MPI-enabled CPUs and CUDA-enabled GPUs (https://developer.nvidia.com/cuda-gpus)
  - A C/C++ compiler (e.g., Microsoft Visual Studio for Windows, and gcc/g++ for Linux/Unix) installed and tested
  - Nvidia CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) installed and tested
  - MPICH (https://www.mpich.org/downloads/) installed and tested
- For the Windows operating system (using MS Visual Studio as an example)

  1. Open all the source codes in Visual Studio
  2. Click menu Project -> Properties -> VC++ Directories -> Include Directories, and add the “include” directory of MPI (e.g., C:\Microsoft SDKs\MPI\Include)
  3. Click menu Project -> Properties -> VC++ Directories -> Lib Directories, and add the “lib” directory of GDAL (e.g., C:\Microsoft SDKs\MPI\Lib\x64)
  4. Click menu Build -> Build Solution
  5. Once successfully compiled, an executable file, rupMC.exe, is created.

- For the Linux/Unix operating system (using the MPI compiler --- mpic++, and the CUDA compiler --- nvcc)
  
  In a Linux/Unix terminal, type in:
  1. $ cd /the-directory-of-source-codes/
  2. $ mpic++ -o MPIrupMC.o -c rupMC.cpp
  3. $ nvcc -c rupMC.cu MC.cu
  4. $ mpic++ -o rupMC -L /usr/local/cuda/lib64 -lcudart -lcurand MPIrupMC.o rupMC.o MC.o
  5. Once successfully compiled, an executable file, rupMC, is created.

**Usage**

- Before running the program, make sure that the input file has the standard VTK DataFile form.
- Two parameters in rupMC.cpp must be manually altered to specify the input file and the values of iso-surfaces for rupMC.

  Example:
  
  string infilePath = "D:/rupMC/Data/geomodel.vtk";
  
  float values[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
