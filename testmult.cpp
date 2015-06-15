#include <iostream>
#include "Matrix.h"
#include "Vector.h"

#define BSIZE 50 //for my bay-trail 
//#define BSIZE 100 //for my core i7
#define CPU_TYPE double

//miminal checker
int main(int argc, char** argv){

	//args
	if (argc != 4){
		std::cout << "Please give matrix size (1000) , platform (0), and device (0)\n";
		return 1;
	}

	int N = atoi(argv[1]);
	size_t platform_choice = atoi(argv[2]);
	size_t device_choice = atoi(argv[3]);
	std::cout << "\n";
	std::cout << "Matrix dimension= " << N << "\n";
	std::cout << "buffer size= " << BSIZE << "\n\n";

	//contruct matrices
	//empty matrix constructor	
	Matrix<CPU_TYPE> myMatrix(N);
	Matrix<CPU_TYPE> myMatrix2(N);

	myMatrix.randomize();
	myMatrix2.randomize();


	//different multiplications
	/*
	//matmult naive
	//std::cout<<"Naive matmult\n";
	//Matrix<double> myMatrixN=matMultNaive(myMatrix,myMatrix2);
	*/


	/*
	//matmult transpon
	//std::cout<<"Naive matmult with transpon\n";
	Matrix<double> myMatrixNB=matMultNaiveTranspon(myMatrix,myMatrix2);
	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout<<(myMatrixNB.equals(myMatrixN) ?  "equal" : "different")<<std::endl;
	std::cout<<"the maximum elementwise difference is  ";
	std::cout<<myMatrixNB.maxDiff(myMatrixN)<<"\n\n";
	*/



	//matmult block 
	//std::cout<<"Block matmult\n";
	Matrix<CPU_TYPE> myMatrixB = matMultBlock0<BSIZE>(myMatrix, myMatrix2);
	/*
	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout<<(myMatrixNB.equals(myMatrixB) ?  "equal" : "different")<<std::endl;
	std::cout<<"the maximum elementwise difference is  ";
	std::cout<<myMatrixNB.maxDiff(myMatrixB)<<"\n\n";
	*/


	// BLOCK Multithreading test

	//matmult not naive 
	//std::cout<<"BLOCK Multithreaded matmult \n";
	Matrix<CPU_TYPE> myMatrixMTB = matMultMultiThreadBlock0<BSIZE>(myMatrix, myMatrix2);

	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout << (myMatrixB.equals(myMatrixMTB) ? "equal" : "different") << std::endl;
	std::cout << "the maximum elementwise difference is  ";
	std::cout << myMatrixB.maxDiff(myMatrixB) << "\n\n";



	//Strassen test
	std::cout << "Strassen matmult\n";
	Matrix<CPU_TYPE> myMatrixS = matMultStrassen0<BSIZE>(myMatrix, myMatrix2);

	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout << (myMatrixB.equals(myMatrixS) ? "equal" : "different") << std::endl;
	std::cout << "the maximum elementwise difference is  ";
	std::cout << myMatrixB.maxDiff(myMatrixS) << "\n\n";



//gpu test
	Matrix<CPU_TYPE> myMatrixGPU = gpu_test_mult(myMatrix, myMatrix2,platform_choice,device_choice);
	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout << (myMatrixB.equals(myMatrixGPU) ? "equal" : "different") << std::endl;
	std::cout << "the maximum elementwise difference is  ";
	std::cout << myMatrixB.maxDiff(myMatrixGPU) << "\n\n";
	std::cout<<"Done"<<std::endl;




//gpu block
	Matrix<CPU_TYPE> myMatrixGPU_C = gpu_matMultMultiThreadBlock("block_mult_2",myMatrix, myMatrix2, platform_choice, device_choice);
	//equality check
	//std::cout<<"multplication method equality check\nthe results are ";
	std::cout << (myMatrixB.equals(myMatrixGPU_C) ? "equal" : "different") << std::endl;
	std::cout << "the maximum elementwise difference is  ";
	std::cout << myMatrixB.maxDiff(myMatrixGPU_C) << "\n\n";

	std::cout << "Done" << std::endl;
	std::cin.get();
	


}
