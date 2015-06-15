//old opencl in a machine
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//type can be changed to float
#define CL_TYPE double

//optimal blocks size for GTX630 
#define BSIZE 4

////////////////////////////////////////////////////////////////////////
/*
	Naive matrix mult
		(blocking with blocksize=1)
*/
////////////////////////////////////////////////////////////////////////
__kernel void test_mult ( 
					__global CL_TYPE* vec1,
					__global CL_TYPE* vec2,
					__global CL_TYPE* vec3,
					__const unsigned int d
					)
{
	const int gid = get_global_id(0);
	int i,j,k;
	i=gid/d;
	j=gid%d;

	
	//zero it just for sure...
	vec3[i*d+j] = 0;
	CL_TYPE temp = 0;

	for(k=0; k<d ;k++)
	{
		temp += vec1[i*d+k] * vec2[k*d+j];
		//vec3[i*d+j] += vec1[i*d+k] * vec2[k*d+j];
	}
	vec3[i*d+j]=temp;
	return;
}

////////////////////////////////////////////////////////////////////////
/*
	Blocks without local copy
*/
////////////////////////////////////////////////////////////////////////

__kernel void block_mult_0 ( 
					__global CL_TYPE* vec1,
					__global CL_TYPE* vec2,
					__global CL_TYPE* vec3,
					__const unsigned int d
					)
{
	//number of full blocks
	int maxblock = (d+BSIZE-1)/BSIZE;

	//get global id
	const int gid = get_global_id(0);
	 
	//pointers
	int ii, jj, kk;
	int i,j,k;

	//there are just full blocks now
	//calculated subblock starting pos:
	ii=(gid/maxblock)*BSIZE;
	jj=(gid%maxblock)*BSIZE;

	//initialize to 0
	for( int i=ii;(i<ii+BSIZE) && i<d;i++){
		for( int j=jj;(j<jj+BSIZE) && j<d;j++){
			vec3[i*d+j]=0;
		}
	}

	//multiplication
	for(kk=0;kk<d;kk+=BSIZE){
		for( int i=ii;(i<ii+BSIZE) && i<d;i++){
			for( int j=jj;(j<jj+BSIZE) && j<d;j++){
				CL_TYPE sum = vec3[i*d+j];
				for( int k=kk;(k<kk+BSIZE) && k<d;k++){
					sum+=vec1[i*d+k] * vec2[k*d+j];
				}
				vec3[i*d+j]=sum;
			}
		}
	}
	return;
}


////////////////////////////////////////////////////////////////////////
/*
	blocked using local memory
		- faster with my small gpu above d=3000
		- only works with multiples of blocksize!!!
*/
////////////////////////////////////////////////////////////////////////

__kernel void block_mult_1 ( 
					__global CL_TYPE* vec1,
					__global CL_TYPE* vec2,
					__global CL_TYPE* vec3,
					__const unsigned int d
					)
{
	//number of full blocks
	int maxblock = (d+BSIZE-1)/BSIZE;
	
	//get global id
	const int gid = get_global_id(0);
	 
	//pointers
	int ii, jj, kk;
	int i,j,k;
	int row1, row2, offs1, offs2; //pointer offsets

	//there are just full blocks now
	//calculated subblock starting pos:
	ii=(gid/maxblock)*BSIZE;
	jj=(gid%maxblock)*BSIZE;


	//local buffers
	CL_TYPE A[BSIZE*BSIZE];
	CL_TYPE B[BSIZE*BSIZE];
	CL_TYPE C[BSIZE*BSIZE];

	
	//null C
	for (i = 0; i<BSIZE; i++){
		row1 = i*BSIZE;
		for (j = 0; j<BSIZE; j++){
			C[row1 + j] = 0;
		}
	}
	
	//loop over blocks
	for(kk=0;kk<d;kk+=BSIZE){
		//copy A
		for (i = 0; i<BSIZE; i++){
			row1 = i*BSIZE;
			offs2 = (ii + i)*d + kk;
			for (k = 0; k<BSIZE; k++){
				A[row1 + k] = vec1[offs2 + k];
			}
		}
		//copy B 
		for (k = 0; k<BSIZE; k++){
			offs2 = (kk + k)*d + jj;
			for (j = 0; j<BSIZE; j++){
				B[j*BSIZE + k] = vec2[offs2 + j]; //B transpon!!
			}
		}
		//multiply small blocks 
		for (i = 0; i<BSIZE; i++){
			row1 = i*BSIZE;
			for (j = 0; j<BSIZE; j++){
				row2 = j*BSIZE;
				CL_TYPE sum = C[i*BSIZE + j];
				for (k = 0; k<BSIZE; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}
	
	//copy back C
	for (i = 0; i<BSIZE; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*BSIZE;
		for (j = 0; j<BSIZE; j++){
			//vec3[0] = 0;
			vec3[offs1 + j] = C[row2 + j];
		}
	}
	
	return;
}

////////////////////////////////////////////////////////////////////////
/*
	blocked using local memory
		- faster with my small gpu (gtx630) above d=2000
			than corei7
		- zero pads local A,B matrices, at edges 
			- work with any d
			- (zero padding is more compact than different code
				for the edge blocks, and i think speed difference is negligible)
*/
////////////////////////////////////////////////////////////////////////

__kernel void block_mult_2 ( 
					__global CL_TYPE* vec1,
					__global CL_TYPE* vec2,
					__global CL_TYPE* vec3,
					__const unsigned int d
					)
{
	//number of full blocks
	int maxblock = (d+BSIZE-1)/BSIZE;
	
	//get global id
	const int gid = get_global_id(0);
	 
	//pointers
	int ii, jj, kk;
	int i,j,k;
	int row1, row2, offs1, offs2; //pointer offsets

	//there are just full blocks now
	//calculated subblock starting pos:
	ii=(gid/maxblock)*BSIZE;
	jj=(gid%maxblock)*BSIZE;


	//local buffers
	CL_TYPE A[BSIZE*BSIZE];
	CL_TYPE B[BSIZE*BSIZE];
	CL_TYPE C[BSIZE*BSIZE];

	
	//null C
	for (i = 0; i<BSIZE; i++){
		row1 = i*BSIZE;
		for (j = 0; j<BSIZE; j++){
			C[row1 + j] = 0;
		}
	}
	
	//loop over blocks
	for(kk=0;kk<d;kk+=BSIZE){

		//copy A
		for (i = 0; i<BSIZE; i++){
			row1 = i*BSIZE;
			offs2 = (ii + i)*d + kk;
			for (k = 0; k<BSIZE; k++){
				if( (ii+i)< d && (kk+k) < d){
					A[row1 + k] = vec1[offs2 + k];
				}
				else{
					A[row1 + k] = 0;
				}

			}
		}
		//copy B 
		for (k = 0; k<BSIZE; k++){
			offs2 = (kk + k)*d + jj;
			for (j = 0; j<BSIZE; j++){
				if( (jj+j)< d && (kk+k) < d){
					B[j*BSIZE + k] = vec2[offs2 + j]; //B transpon!!
				}
				else{
					B[j*BSIZE + k] = 0;
				}
				
			}
		}
		//multiply small blocks 
		for (i = 0; i<BSIZE; i++){
			row1 = i*BSIZE;
			for (j = 0; j<BSIZE; j++){
				row2 = j*BSIZE;
				CL_TYPE sum = C[i*BSIZE + j];
				for (k = 0; k<BSIZE; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}
	
	//copy back C
	for (i = 0; i<BSIZE; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*BSIZE;
		for (j = 0; j<BSIZE; j++){
			if( (ii+i)< d && (jj+j) < d){
				vec3[offs1 + j] = C[row2 + j];
			}			
		}
	}
	
	return;
}