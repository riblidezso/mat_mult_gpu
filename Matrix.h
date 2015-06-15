//include the opencl header
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif


#ifndef MATRIX_H 
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <initializer_list>
#include <stdlib.h>
#include <stdio.h>
#include <iterator>

#include <random>
#include <chrono>

#include<thread>

#include <string.h>

#include "Vector.h"

#include<math.h>




template<typename T>
class Matrix{
	//contructors and destr
public:
	Matrix(size_t ind);//empty constuctor
	Matrix(const T* inputData, size_t ind);//from array
	Matrix(std::initializer_list<std::initializer_list<T> > initVals);
	Matrix(const Matrix<T> &matrixToCopy);//copy constr
	Matrix(Matrix<T>&& matrixToCopy);//move constr
	~Matrix(); //destructor

	//data
private:
	T* data;
	size_t d; //dimension

	//methods	
public:
	//assignements
	void operator=(const Matrix<T>& matrixToCopy);//copy asssignement
	void operator=(Matrix<T>&& matrixToCopy);//move asssignement
	void releaseData();

	//get data
	T* getDataPointer() const;
	size_t getDim() const;

	//write matrix to console
	std::string toString() const; //toString

	//equality checkers
	bool equals(const Matrix<T>& toCompare); //equals method
	T max(); //max of elements
	T maxDiff(const Matrix<T>& toCompare); //maximum diff with other matrix

	//setters, getters with bound checking
	T get(size_t i, size_t j) const;//getters
	Vector<T> getRow(size_t i);
	Vector<T> getCol(size_t j);
	void set(size_t i, size_t j, T value);//setters
	void setRow(size_t i, const Vector<T>& inputRow);
	void setCol(size_t j, const Vector<T>& inputCol);

	//for random testing
	void randomize();	//make matrix random

	//math stuff
	T trace();
	T det();
	Matrix<T> transpon();
	Matrix<T> gauInv();




	///////////////////////////////////////////////////////////////////////////
	/*
	Forward iterators
	to row, col, and diagional iteration
	with implementation here
	*/
	///////////////////////////////////////////////////////////////////////////
	//iterators
	//rows
	//no need to know about the  dimension of Matrix instance
	//but for symettry reasons it will require it!
	class rowIterator : public std::iterator<std::forward_iterator_tag, int>{
	public:
		typedef rowIterator self_type;
		rowIterator(T* ptr, size_t ind) : ptr_(ptr), d(ind) { }
		self_type operator++() { self_type i = *this; ptr_++; return i; }
		self_type operator++(int junk) { ptr_++; return *this; }
		T& operator*() { return *ptr_; }
		T* operator->() { return ptr_; }
		bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
		bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
	private:
		T* ptr_;
		size_t d;
	};

	rowIterator rowBegin(int i){
		return rowIterator(data + i*d, d);
	}
	rowIterator rowEnd(int i){
		return rowIterator(data + (i + 1)*d, d);
	}

	//cols
	//needs to know the dimension of Matrix instance!!
	class colIterator : public std::iterator<std::forward_iterator_tag, int>{
	public:
		typedef colIterator self_type;
		colIterator(T* ptr, size_t ind) : ptr_(ptr), d(ind) { }
		self_type operator++() { self_type i = *this; ptr_ += d; return i; }
		self_type operator++(int junk) { ptr_ += d; return *this; }
		T& operator*() { return *ptr_; }
		T* operator->() { return ptr_; }
		bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
		bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
	private:
		T* ptr_;
		size_t d;
	};

	colIterator colBegin(int i){
		return colIterator(data + i, d);
	}
	colIterator colEnd(int i){
		return colIterator(data + i + d*d, d);
	}

	//diag
	//needs to know the dimension of Matrix instance!!
	class diagIterator : public std::iterator<std::forward_iterator_tag, int>{
	public:
		typedef diagIterator self_type;
		diagIterator(T* ptr, size_t ind) : ptr_(ptr), d(ind) { }
		self_type operator++() { self_type i = *this; ptr_ += d + 1; return i; }
		self_type operator++(int junk) { ptr_ += d + 1; return *this; }
		T& operator*() { return *ptr_; }
		T* operator->() { return ptr_; }
		bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
		bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
	private:
		T* ptr_;
		size_t d;
	};

	diagIterator diagBegin(){
		return diagIterator(data, d);
	}
	diagIterator diagEnd(){
		return diagIterator(data + d*(d + 1), d);
	}

};








////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
/*
Not member stuff
*/
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////
/*
Arithmetic operations declarations
*/


//diad
template<typename T>
Matrix<T> diad(const Vector<T>& lVec, const Vector<T>& rVec);

//matrix algebra as non-member operators
//core
template<typename T>
Matrix<T> operator+(Matrix<T> lMat, const Matrix<T>& rMat);
template<typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& rMat);

//combinations (rigths, + inverses)
template<typename T>
Matrix<T> operator-(Matrix<T> lMat, const Matrix<T>& rMat);
template<typename T>
Matrix<T> operator*(const Matrix<T>& lMat, T scalar);
template<typename T>
Matrix<T> operator/(const Matrix<T>& lMat, T scalar);

//matmult
template<typename T>
Matrix<T> operator*(Matrix<T> lMat, const Matrix<T>& rMat);

//vector arithmetics
template<typename T>
Vector<T> operator*(const Vector<T>& lVec, const Matrix<T>& rMat);
template<typename T>
Vector<T> operator*(const Matrix<T>& lMat, const Vector<T>& rVec);



////////////////////////////////////////////////////////////////////
/*
Advanced Matrix multiplication routines
-There are other methods which are commmented
they were experimental stages, but all works
fine, just have to be uncommented
*/

//Naive
template<typename T>
Matrix<T> matMultNaive(const Matrix<T>& lMat, const Matrix<T>& rMat);
//Naive with trasponation
template<typename T>
Matrix<T> matMultNaiveTranspon(const Matrix<T>& lMat, const Matrix<T>& rMat);
//Cache optimized Block	
template<size_t bsize, typename T>
Matrix<T> matMultBlock(const Matrix<T>& lMat, const Matrix<T>& rMat);
//Multithreaded block
template<size_t bsize, typename T>
Matrix<T> matMultMultiThreadBlock(const Matrix<T>& lMat, const Matrix<T>& rMat);
//Strassen algorithm (with block for small matrices)
template<size_t bsize, typename T>
Matrix<T> matMultStrassenRec(const Matrix<T>& lMat, const Matrix<T>& rMat);
////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
/*
Implementations
*/
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////
/*
Empty constructor
*/
template<typename T>
Matrix<T>::Matrix(size_t ind){
	d = ind;
	data = new T[d*d];
}


/*
Constructor from array
*/
template<typename T>
Matrix<T>::Matrix(const T* inputData, size_t ind){
	d = ind;
	data = new T[d*d];
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			set(i, j, inputData[i*d + j]);
		}
	}
}

/*
Constructor from std::initiializer_list
row continuous!!!
*/
template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T> > initVals){
	d = initVals.size();
	data = new T[d*d];
	size_t i = 0;
	for (auto row : initVals){
		size_t j = 0;
		for (auto val : row){
			set(i, j, val);
			j++;
		}
		i++;
	}
}


/*
Copy constructor
*/
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &matrixToCopy){
	d = matrixToCopy.getDim();
	data = new T[d*d];
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			set(i, j, matrixToCopy.get(i, j));
		}
	}
}

/*
Move constructor
*/
template<typename T>
Matrix<T>::Matrix(Matrix<T>&& matrixToCopy){
	d = matrixToCopy.getDim();
	data = matrixToCopy.getDataPointer();
	matrixToCopy.releaseData();
}


/*
Destructor
*/
template<typename T>
Matrix<T>::~Matrix(){
	delete[] data;
}



/*
Release data method
for move semantic
*/
template<typename T>
void Matrix<T>::releaseData(){
	data = NULL;
}

/*
Get data pointer
-for move semantic
-and fast Matmult
*/
template<typename T>
T* Matrix<T>::getDataPointer() const {
	return data;
}

/*
Copy assignement operator
*/
template<typename T>
void Matrix<T>::operator=(const Matrix<T> &matrixToCopy){
	delete[] data;
	d = matrixToCopy.getDim();
	data = new T[d*d];

	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			set(i, j, matrixToCopy.get(i, j));
		}
	}
}

/*
Move assignement operator
*/
template<typename T>
void Matrix<T>::operator=(Matrix<T>&& matrixToCopy){
	delete[] data;
	d = matrixToCopy.getDim();
	data = matrixToCopy.getDataPointer();
	matrixToCopy.releaseData();
}

/*
toString method
*/
template<typename T>
std::string Matrix<T>::toString() const {
	std::stringstream sstr;
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			sstr << get(i, j) << "\t";
		}
		sstr << "\n";
	}
	return sstr.str();
}

/*
equals method

Keep in mind that this uses exact value checking!!
- So if you invert a matrix, with two different
algorithms: naive, and gauss-elim for example,
Then there's a good chance that the inverse
matrices will not be the same!!!!
- This can be the same for matrix multiplication!!

*/
template<typename T>
bool Matrix<T>::equals(const Matrix<T>& toCompare){
	if (getDim() != toCompare.getDim()) //dimcheck
		return false;

	for (size_t i = 0; i<d; i++){	//valcheck
		for (size_t j = 0; j<d; j++){
			if (get(i, j) != toCompare.get(i, j))
				return false;
		}
	}
	return true;
}


/*
Get dimension method
*/
template<typename T>
size_t Matrix<T>::getDim() const {
	return d;
}


/*
Get element method
*/
template<typename T>
T Matrix<T>::get(size_t i, size_t j) const {
	if (i >= d || j >= d || i<0 || j<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::get(size_t i, size_t j) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong indices, i = " << i << ",  j = " << j << "\n";
		exit(1);
	}
	return data[i*d + j];
}

/*
Get row method
*/
template<typename T>
Vector<T> Matrix<T>::getRow(size_t i){
	if (i >= d || i<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::getRow(size_t i) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong index, i = " << i << "\n";
		exit(1);
	}
	Vector<T> result = Vector<T>(d);
	for (size_t j = 0; j<d; j++){
		result.set(j, get(i, j));
	}
	return result;
}

/*
Get col  method
*/
template<typename T>
Vector<T> Matrix<T>::getCol(size_t j){
	if (j >= d || j<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::getCol(size_t j) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong index, j = " << j << "\n";
		exit(1);
	}
	Vector<T> result = Vector<T>(d);
	for (size_t i = 0; i<d; i++){
		result.set(i, get(i, j));
	}
	return result;
}

/*
Set element method
*/
template<typename T>
void Matrix<T>::set(size_t i, size_t j, T value){
	if (i >= d || j >= d || i<0 || j<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::set(size_t i, size_t j, T value) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong indices, i = " << i << ",  j = " << j << "\n";
		exit(1);
	}
	data[i*d + j] = value;
}

/*
Set row method
*/
template<typename T>
void Matrix<T>::setRow(size_t i, const Vector<T>& inputRow) {
	if (i >= d || i<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::setRow(size_t i, const Vector<T>& inputCol) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong index, i = " << i << "\n";
		exit(1);
	}
	for (size_t j = 0; j<d; j++){
		set(i, j, inputRow.get(j));
	}
}

/*
Set column method
*/
template<typename T>
void Matrix<T>::setCol(size_t j, const Vector<T>& inputRow) {
	if (j >= d || j<0){
		std::cerr << "OUT OF BOUNDS ERROR!\n";
		std::cerr << "in Matrix::setCol(size_t j, const Vector<T>& inputRow) method\n";
		std::cerr << "square matrix dimension, d = " << d << "\n";
		std::cerr << "wrong index, j = " << j << "\n";
		exit(1);
	}
	for (size_t i = 0; i<d; i++){
		set(i, j, inputRow.get(i));
	}
}


/*
trace method
*/
template<typename T>
T Matrix<T>::trace(){
	T trace = 0;
	for (size_t i = 0; i<d; i++){
		trace += get(i, i);
	}
	return trace;
}

/*
transponation method
*/
template<typename T>
Matrix<T> Matrix<T>::transpon(){
	Matrix<T> mT = Matrix<T>(data, d);
	T temp = 0;
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<i; j++){
			temp = mT.get(i, j);
			mT.set(i, j, mT.get(j, i));
			mT.set(j, i, temp);
		}
	}
	return mT;
}

/*
Det calculating method
*/
template<typename T>
T Matrix<T>::det(){
	//upper triangulize matrix
	Matrix<T> uMat = Matrix<T>(data, d);
	T ratio = 0;
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<i; j++){
			ratio = uMat.get(i, j) / uMat.get(j, j);
			for (size_t k = 0; k<d; k++){
				uMat.set(i, k, uMat.get(i, k) - ratio*uMat.get(j, k));
			}
		}
	}
	//calclate det
	T determ = 1;
	for (size_t i = 0; i<d; i++){
		determ *= uMat.get(i, i);
	}
	return determ;
}

/*
Gaussian elimination inverter
Actually this is copied and modified from
old assignement, so style will be C-ish,
and different
Do not even think about using it on non floating point matrices!!
*/
template<typename T>
Matrix<T> Matrix<T>::gauInv(){
	//simplify new matrix class to C-style matrix (p,n)
	int n = d;
	T *p = new T[d*d];
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			p[n*i + j] = get(i, j);
		}
	}

	//rewrite old C-style memory allocs to templates
	//long double *inv,*sormax,*ideig,*ideiginv,max,a;
	T max, a;
	int i, j, j1, holmax;
	//inv=(long double*)calloc(n*n,sizeof(long double));
	T *inv = new T[d*d];
	//sormax=(long double*)calloc(n,sizeof(long double));
	T *sormax = new T[d];
	//ideig=(long double*)calloc(n,sizeof(long double));
	T *ideig = new T[d];
	//ideiginv=(long double*)calloc(n,sizeof(long double));
	T *ideiginv = new T[d];


	//from this part it is absolutely unmodified old C code
	//////////////////////////////////////////////////////////////////


	//Normálás

	for (i = 0; i<n; i++)
	{
		//max keresése
		for (j = 0; j<n; j++)
		{
			if (sormax[i]<p[n*i + j])
			{
				sormax[i] = p[n*i + j];
			}
		}

		if (sormax[i] == 0)
		{
			printf("A mátrix szinguláris!\n");
			exit(1);
		}

		//normálás
		for (j = 0; j<n; j++)
		{
			p[n*i + j] = p[n*i + j] / sormax[i];
		}

		//inverz normálása
		inv[i*n + i] = 1 / sormax[i];
	}

	//Invertálás

	//oszloponként haladva (i mindig sor, j mindig oszlop!, csak ugye ezek 0-tól mennek, nem 1-től!)
	for (j = 0; j<n; j++)
	{
		//numerikus hibák minimalizálása miatt olyan sor keresése, amiben maximális a j.edik elem. persze csak j.edik sortól keresve!
		max = 0;
		holmax = 0;
		for (i = j; i<n; i++)
		{
			if (max<p[n*i + j])
			{
				max = p[n*i + j];
				holmax = i;
			}
		}

		//sorcserék mindkét mátrixban	
		for (j1 = 0; j1<n; j1++)
		{
			ideig[j1] = p[n*j + j1]; //itt j sorindex, mert átlós elemekből kiindulva eliminálunk.
			ideiginv[j1] = inv[n*j + j1];
		}

		for (j1 = 0; j1<n; j1++)
		{
			p[n*j + j1] = p[n*holmax + j1];
			inv[n*j + j1] = inv[n*holmax + j1];
		}

		for (j1 = 0; j1<n; j1++)
		{
			p[n*holmax + j1] = ideig[j1];
			inv[n*holmax + j1] = ideiginv[j1];
		}

		//A j. sor normálása mindkét mátrixra, az átlóbeli elemmel.

		a = p[n*j + j];

		for (j1 = 0; j1<n; j1++)
		{
			p[n*j + j1] = p[n*j + j1] / a;
			inv[n*j + j1] = inv[n*j + j1] / a;
		}

		//j. oszlopok lenullázása

		for (i = 0; i<n; i++)
		{
			a = p[n*i + j];

			if (i != j)
			{
				for (j1 = 0; j1<n; j1++)
				{
					p[n*i + j1] = p[n*i + j1] - a*p[n*j + j1];// j sorindex most is!
					inv[n*i + j1] = inv[n*i + j1] - a*inv[n*j + j1];
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////
	//until now!

	//Now i turn old C-style matrix to new fancy matrix class and return it
	Matrix<T> invMat = Matrix<T>(inv, d);
	return invMat;
}
















///////////////////////////////////////////////////////
/*
Not member stuff
*/
///////////////////////////////////////////////////////

/*
diad multiplication
*/
template<typename T>
Matrix<T> diad(const Vector<T>& lVec, const Vector<T>& rVec){
	size_t d = lVec.getDim();
	Matrix<T> result = Matrix<T>(d);
	for (size_t j = 0; j<d; j++){
		for (size_t i = 0; i<d; i++){
			result.set(i, j, lVec.get(i)*rVec.get(j));
		}
	}
	return result;
}

/*
matrix addition method
*/
template<typename T>
Matrix<T> operator+(Matrix<T> lMat, const Matrix<T>& rMat){
	/*
	size_t d=rMat.getDim();
	Matrix<T> sumMat=Matrix<T>(d);
	for(int i=0;i<d;i++){
	for(int j=0;j<d;j++){
	sumMat.set(i,j,lMat.get(i,j)+rMat.get(i,j));
	}
	}
	return sumMat;
	*/
	size_t d = rMat.getDim();
	Matrix<T> sumMat = Matrix<T>(d);

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* sumPtr = sumMat.getDataPointer();

	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			*(sumPtr + i*d + j) = *(lPtr + i*d + j) + *(rPtr + i*d + j);
		}
	}
	return sumMat;

}

/*
left scalar multiplication method
*/
template<typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& rMat){
	/*	size_t d=rMat.getDim();
	Matrix<T> sumMat=Matrix<T>(d);
	for(int i=0;i<d;i++){
	for(int j=0;j<d;j++){
	sumMat.set(i,j,scalar*rMat.get(i,j));
	}
	}
	return sumMat;
	*/
	size_t d = rMat.getDim();
	Matrix<T> sumMat = Matrix<T>(d);

	T* rPtr = rMat.getDataPointer();
	T* sumPtr = sumMat.getDataPointer();

	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			*(sumPtr + i*d + j) = scalar * *(rPtr + i*d + j);
		}
	}
	return sumMat;


}

/*
matrix substraction method
*/
template<typename T>
Matrix<T> operator-(Matrix<T> lMat, const Matrix<T>& rMat){
	//return lMat + (-1.0 * rMat);
	return lMat + (((T)-1.0) * rMat);
}

/*
right scalar multiplication method
*/
template<typename T>
Matrix<T> operator*(const Matrix<T>& lMat, T scalar){
	return scalar*lMat;
}

/*
right scalar division method
!!note there is no left scalar method
*/
template<typename T>
Matrix<T> operator/(const Matrix<T>& lMat, T scalar){
	return (1.0 / scalar)*lMat;
}

/*
matrix multiplication method
-uses ikj multiplication
-for faster methods explicitly use
-block
-multithreaded
-strassen

*/
template<typename T>
Matrix<T> operator*(Matrix<T> lMat, const Matrix<T>& rMat){
	/*	size_t d=lMat.getDim();
	Matrix<T> mulMat=Matrix<T>(d);
	T temp;
	for(int i=0;i<d;i++){
	for(int j=0;j<d;j++){
	temp=0;
	for(int k=0;k<d;k++){
	temp+=lMat.get(i,k)*rMat.get(k,j);
	}
	mulMat.set(i,j,temp);
	}
	}
	return mulMat;
	*/

	/*
	ikj multiplication method:
	simple, fast, memory efficient
	*/
	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	for (size_t i = 0; i<d; i++){
		//nullify elemets
		for (size_t j = 0; j<d; j++){
			*(resPtr + i*d + j) = 0;
		}
		//mult
		for (size_t k = 0; k<d; k++){
			T temp = *(lPtr + i*d + k);
			for (size_t j = 0; j<d; j++){
				*(resPtr + i*d + j) += temp * *(rPtr + k*d + j);
			}
		}
	}
	return mulMat;
}

/*
Vector multiplication M*v
*/
template<typename T>
Vector<T> operator*(const Matrix<T>& lMat, const Vector<T>& rVec){
	size_t d = lMat.getDim();
	Vector<T> result = Vector<T>(d);
	T temp;
	for (size_t i = 0; i<d; i++){
		temp = 0;
		for (size_t j = 0; j<d; j++){
			temp += lMat.get(i, j)*rVec.get(j);
		}
		result.set(i, temp);
	}
	return result;
}


/*
Vector multiplication v*M
*/
template<typename T>
Vector<T> operator*(const Vector<T>& lVec, const Matrix<T>& rMat){
	size_t d = rMat.getDim();
	Vector<T> result = Vector<T>(d);
	T temp;
	for (size_t j = 0; j<d; j++){
		temp = 0;
		for (size_t i = 0; i<d; i++){
			temp += rMat.get(i, j)*lVec.get(j);
		}
		result.set(j, temp);
	}
	return result;
}





/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*
Matrix multiplication methods
*/

/////////////////////////////////////////////////////////////////////////
/*
randomize method for random matrix generation for testing
-using time seed
*/

template<typename T>
void Matrix<T>::randomize(){
	//generating double random numbers
	//idk how to generate any random template type
	//????
	std::default_random_engine generator;
	//std::uniform_real_distribution<T> distribution(0, 1);
	std::uniform_real_distribution<double> distribution(0, 1);

	// obtain a seed from the timer
	typedef std::chrono::high_resolution_clock myclock;
	myclock::time_point beginning = myclock::now();
	myclock::duration dur = myclock::now() - beginning;
	unsigned seed = dur.count();
	generator.seed(seed);

	//generate random numbers
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			set(i, j, distribution(generator));
		}
	}
}

//////////////////////////////////////////////////////////////
/*
Methods for comparing not exactly equal matrixes
*/

/*
Max element method
*/
template<typename T>
T Matrix<T>::max(){
	T max = get(0, 0);
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			if (max < get(i, j))
				max = get(i, j);
		}
	}
	return max;
}

/*
Max difference method
*/
template<typename T>
T Matrix<T>::maxDiff(const Matrix<T>& toCompare){
	return (*this - toCompare).max();
}

////////////////////////////////////////////////////////////////////////
/*
Numerous matrix multiplication methods
*/


///////////////////////////////////////////////////////////
//Naive methods

/*
Naive matrix multiplication method 1
-using gets and sets
-this is slow of course
*/
/*
template<typename T>
Matrix<T> matMultNaive1( const Matrix<T>& lMat,const Matrix<T>& rMat ){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
Matrix<T> mulMat=Matrix<T>(d);
for(int i=0;i<d;i++){
for(int j=0;j<d;j++){
T temp=0;
for(int k=0;k<d;k++){
temp+=lMat.get(i,k)*rMat.get(k,j);
}
mulMat.set(i,j,temp);
}
}
myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nNaive matrix mutiplication using\n";
std::cout<<"get() and set() took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
return mulMat;
}
*/

/*
Naive matrix multiplication method 2
-using pointer access
*/
template<typename T>
Matrix<T> matMultNaive(const Matrix<T>& lMat, const Matrix<T>& rMat){
	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			T temp = 0;
			for (size_t k = 0; k<d; k++){
				temp += *(lPtr + i*d + k) * *(rPtr + k*d + j);
			}
			*(resPtr + i*d + j) = temp;
		}
	}

	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "Naive matrix mutiplication\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return mulMat;
}


////////////////////////////////////////////////////////////
//Augmented naive methods

/*
Naive matrix multiplication method 2
-using trnasponation of rMat first!
-using pointer access
*/
template<typename T>
Matrix<T> matMultNaiveTranspon(const Matrix<T>& lMat, const Matrix<T>& rMat){
	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	//transpon
	T* temp = new T[d*d];
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			*(temp + j*d + i) = *(rPtr + i*d + j);
		}
	}

	//multip
	for (size_t i = 0; i<d; i++){
		for (size_t j = 0; j<d; j++){
			T sum = 0;
			for (size_t k = 0; k<d; k++){
				sum += *(lPtr + i*d + k) * *(temp + j*d + k);
			}
			*(resPtr + i*d + j) = sum;
		}
	}
	delete[] temp;

	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "Naive matrix mutiplication w transponation\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return mulMat;
}


/*
Loop order changed matrix multiplication method
- we only change the loop order!
- taken the idea from Linpack dgemm.f
*/
/*
template<typename T>
Matrix<T> matMultikj(const Matrix<T>& lMat, const Matrix<T>& rMat ){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
Matrix<T> mulMat=Matrix<T>(d);
T* lPtr=lMat.getDataPointer();
T* rPtr=rMat.getDataPointer();
T* resPtr=mulMat.getDataPointer();
for(int i=0;i<d;i++){
//nullify elemets
for(int j=0;j<d;j++){
*(resPtr+i*d+j)=0;
}
//mult
for(int k=0;k<d;k++){
T temp= *(lPtr+i*d+k);
for(int j=0;j<d;j++){
*(resPtr+i*d+j) += temp * *(rPtr+k*d+j);
}
}
}
myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nNot so Naive matrix mutiplication \n";
std::cout<<"took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
return mulMat;
}
*/


//////////////////////////////////////////////////////////
//Block matrix methods


/*
Block matrix multiplication method
- using pointer access
- no temporary container
*/
/*
template<typename T>
Matrix<T> matMultBlock0A(const Matrix<T>& lMat, const Matrix<T>& rMat, int bsize ){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
Matrix<T> mulMat=Matrix<T>(d);
T* lPtr=lMat.getDataPointer();
T* rPtr=rMat.getDataPointer();
T* resPtr=mulMat.getDataPointer();
//nullify
for(int i=0;i<d;i++){
for (int j=0;j<d;j++){
*(resPtr+i*d+j)=0;
}
}

//blocks
for(int ii=0;ii<d;ii+=bsize){
for(int jj=0;jj<d;jj+=bsize){
for(int kk=0;kk<d;kk+=bsize){
//multiply small blocks
for( int i=ii;(i<ii+bsize) && i<d;i++){
for( int j=jj;(j<jj+bsize) && j<d;j++){
T sum=*(resPtr+i*d+j);
for( int k=kk;(k<kk+bsize) && k<d;k++){
sum+=*(lPtr+i*d+k) * *(rPtr+k*d+j);
}
*(resPtr+i*d+j)=sum;
}
}
}
}
}

myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nBlock matrix mutiplication using\n";
std::cout<<"Block size = "<<bsize<<" took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
return mulMat;
}
*/



/*
Block matrix multiplication method
- using pointer access
- using temporary containers

*/
/*
template<typename T>
Matrix<T> matMultBlock0B(const Matrix<T>& lMat, const Matrix<T>& rMat, int bsize ){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
Matrix<T> mulMat=Matrix<T>(d);
T* lPtr=lMat.getDataPointer();
T* rPtr=rMat.getDataPointer();
T* resPtr=mulMat.getDataPointer();
T C[bsize*bsize]; //temp cont for C
T A[bsize*bsize]; //temp cont for A
T B[bsize*bsize]; //temp cont for B
//blocks
for(int ii=0;ii<d;ii+=bsize){
for(int jj=0;jj<d;jj+=bsize){
//null C (is it necessary?)
for(int i=0;i<bsize && i+ii<d ;i++){
for(int j=0;j<bsize && j+jj<d ;j++){
C[i*bsize+j]=0;
}
}
for(int kk=0;kk<d;kk+=bsize){
//copy A
for(int i=0;i<bsize && i+ii<d ;i++){
for(int k=0;k<bsize && k+kk<d ;k++){
A[i*bsize+k]=*(lPtr+(ii+i)*d+kk+k);
}
}
//copy B
for(int k=0;k<bsize && k+kk<d ;k++){
for(int j=0;j<bsize && j+jj<d ;j++){
B[j*bsize+k]=*(rPtr+(kk+k)*d+jj+j);
}
}
//multiply small blocks
for(int i=0;i<bsize && i+ii<d ;i++){
for(int j=0;j<bsize && j+jj<d ;j++){
T sum=C[i*bsize+j];
for(int k=0;k<bsize && k+kk<d ;k++){
sum+=A[i*bsize+k]*B[j*bsize+k];
}
C[i*bsize+j]=sum;
}
}
}
//copy back C
for(int i=0;i<bsize && i+ii<d ;i++){
for(int j=0;j<bsize && j+jj<d ;j++){
*(resPtr+(ii+i)*d+jj+j)=C[i*bsize+j];
}
}
}
}

myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nBlock matrix mutiplication using\n";
std::cout<<"Block size = "<<bsize<<" took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
return mulMat;
}
*/

/*
Block matrix multiplication method
- using pointer access
- using temporary containers
- using precalculated pointer offsets
int the innermost loops
-only works when d=n*bsize !!!!!

*/
/*
template<size_t bsize,typename T>
Matrix<T> matMultBlock0C(const Matrix<T>& lMat, const Matrix<T>& rMat){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
//checking exact size match
if(d%bsize != 0){
std::cout<<"\t\t--------------------------------\n";
std::cout<<"\t\tERROR\n";
std::cout<<"\t\tMatrix size is not a multiplier of block size\n";
std::cout<<"\t\tResult will be false\n\n";
exit(1);
}

Matrix<T> mulMat=Matrix<T>(d);
T* lPtr=lMat.getDataPointer();
T* rPtr=rMat.getDataPointer();
T* resPtr=mulMat.getDataPointer();
T C[bsize*bsize]; //temp cont for C
T B[bsize*bsize]; //temp cont for B
T A[bsize*bsize]; //temp cont for A
int maxblock=bsize*(d/bsize);
size_t row1,row2,offs1,offs2;
// calculate full blocks
for(int ii=0;ii<maxblock;ii+=bsize){
for(int jj=0;jj<maxblock;jj+=bsize){
//null C
for(int i=0;i<bsize ;i++){
row1=i*bsize;
for(int j=0;j<bsize ;j++){
C[row1+j]=0;
}
}
//multiply full blocks
for(int kk=0;kk<maxblock;kk+=bsize){
//copy A
for(int i=0;i<bsize;i++){
row1=i*bsize;
offs2=(ii+i)*d+kk;
for(int k=0;k<bsize;k++){
A[row1+k]=*(lPtr+offs2+k);
}
}
//copy B
for(int k=0;k<bsize;k++){
offs2=(kk+k)*d+jj;
for(int j=0;j<bsize;j++){
B[j*bsize+k]=*(rPtr+offs2+j); //B transpon!!
}
}
//multiply small blocks
for(int i=0;i<bsize;i++){
row1=i*bsize;
for(int j=0;j<bsize;j++){
row2=j*bsize;
T sum=C[i*bsize+j];
for(int k=0;k<bsize;k++){
sum+=A[row1+k]*B[row2+k];
}
C[row1+j]=sum;
}
}
}
//copy back C
for(int i=0;i<bsize;i++){
offs1=(ii+i)*d+jj;
row2=i*bsize;
for(int j=0;j<bsize;j++){
*(resPtr+offs1+j)=C[row2+j];
}
}
}
}
myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nBlock matrix mutiplication using\n";
std::cout<<"Block size = "<<bsize<<" took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
return mulMat;
}
*/



/*
Block matrix multiplication method
- using pointer access
- using temporary containers
- using precalculated pointer offsets
int the innermost loops
- using distinct for loops for border blocks
to avoid extensive border checking
with runtime variables
*/

//subroutine declarations
template<size_t bsize, typename T>
void matMultBlockFullBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t ii, size_t jj);
template<size_t bsize, typename T>
void matMultBlockRightBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t ii);
template<size_t bsize, typename T>
void matMultBlockBottomBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t jj);
template<size_t bsize, typename T>
void matMultBlockRightBottom(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat);


/*
Benchmark
*/
template<size_t bsize, typename T>
Matrix<T> matMultBlock0(const Matrix<T>& lMat, const Matrix<T>& rMat){

	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();


	Matrix<T> resMat = matMultBlock<bsize, T>(lMat, rMat);

	//benchmark 
	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "Block matrix mutiplication\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return resMat;
}



template<size_t bsize, typename T>
Matrix<T> matMultBlock(const Matrix<T>& lMat, const Matrix<T>& rMat){

	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	size_t maxblock = bsize*(d / bsize);

	////////////////////////////////////////////////////////////////////
	// calculate full blocks	
	for (size_t ii = 0; ii<maxblock; ii += bsize){
		for (size_t jj = 0; jj<maxblock; jj += bsize){
			matMultBlockFullBlocks<bsize>(lMat, rMat, mulMat, ii, jj);
		}
	}
	//////////////////////////////////////////////////////////////////////
	// calculate right end blocks	
	for (size_t ii = 0; ii<maxblock; ii += bsize){
		matMultBlockRightBlocks<bsize>(lMat, rMat, mulMat, ii);
	}
	//////////////////////////////////////////////////////////////////////
	// calculate bottom end blocks	
	for (size_t jj = 0; jj<maxblock; jj += bsize){
		matMultBlockBottomBlocks<bsize>(lMat, rMat, mulMat, jj);
	}
	//////////////////////////////////////////////////////////////////////
	// calculate bottom right end block	
	matMultBlockRightBottom<bsize>(lMat, rMat, mulMat);

	return mulMat;
}


/*
Full blocks
*/
template<size_t bsize, typename T>
void matMultBlockFullBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t ii, size_t jj){

	size_t d = lMat.getDim();

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	T A[bsize*bsize]; //left
	T B[bsize*bsize]; //right
	T C[bsize*bsize]; //result

	size_t maxblock = bsize*(d / bsize);
	size_t kk; //block counters
	size_t kbsize; //block sizes at borders
	size_t row1, row2, offs1, offs2; //pointer offsets


	//null C
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			C[row1 + j] = 0;
		}
	}
	//multiply full blocks
	for (size_t kk = 0; kk<maxblock; kk += bsize){
		//copy A 
		for (size_t i = 0; i<bsize; i++){
			row1 = i*bsize;
			offs2 = (ii + i)*d + kk;
			for (size_t k = 0; k<bsize; k++){
				A[row1 + k] = *(lPtr + offs2 + k);
			}
		}
		//copy B 
		for (size_t k = 0; k<bsize; k++){
			offs2 = (kk + k)*d + jj;
			for (size_t j = 0; j<bsize; j++){
				B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
			}
		}
		//multiply small blocks 
		for (size_t i = 0; i<bsize; i++){
			row1 = i*bsize;
			for (size_t j = 0; j<bsize; j++){
				row2 = j*bsize;
				T sum = C[i*bsize + j];
				for (size_t k = 0; k<bsize; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}

	//multiply last 2 blocks
	kk = maxblock;
	kbsize = d - maxblock;
	//copy A 
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		offs2 = (ii + i)*d + kk;
		for (size_t k = 0; k<kbsize; k++){
			A[row1 + k] = *(lPtr + offs2 + k);
		}
	}
	//copy B 
	for (size_t k = 0; k<kbsize; k++){
		offs2 = (kk + k)*d + jj;
		for (size_t j = 0; j<bsize; j++){
			B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
		}
	}
	//multiply the 2 small blocks 
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			row2 = j*bsize;
			T sum = C[i*bsize + j];
			for (size_t k = 0; k<kbsize; k++){
				sum += A[row1 + k] * B[row2 + k];
			}
			C[row1 + j] = sum;
		}
	}

	//copy back C
	for (size_t i = 0; i<bsize; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			*(resPtr + offs1 + j) = C[row2 + j];
		}
	}
}


/*
Right end blocks
*/
template<size_t bsize, typename T>
void matMultBlockRightBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t ii){

	size_t d = lMat.getDim();

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	T A[bsize*bsize]; //left
	T B[bsize*bsize]; //right
	T C[bsize*bsize]; //result

	size_t maxblock = bsize*(d / bsize);
	size_t jj, kk; //block counters
	size_t jbsize, kbsize; //block sizes at borders
	size_t row1, row2, offs1, offs2; //pointer offsets

	jj = maxblock;
	jbsize = d - maxblock;

	//null C
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			C[row1 + j] = 0;
		}
	}
	//multiply full blocks
	for (size_t kk = 0; kk<maxblock; kk += bsize){
		//copy A 
		for (size_t i = 0; i<bsize; i++){
			row1 = i*bsize;
			offs2 = (ii + i)*d + kk;
			for (size_t k = 0; k<bsize; k++){
				A[row1 + k] = *(lPtr + offs2 + k);
			}
		}
		//copy B 
		for (size_t k = 0; k<bsize; k++){
			offs2 = (kk + k)*d + jj;
			for (size_t j = 0; j<jbsize; j++){
				B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
			}
		}
		//multiply small blocks 
		for (size_t i = 0; i<bsize; i++){
			row1 = i*bsize;
			for (size_t j = 0; j<jbsize; j++){
				row2 = j*bsize;
				T sum = C[i*bsize + j];
				for (size_t k = 0; k<bsize; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}
	//multiply last 2 blocks
	kk = maxblock;
	kbsize = d - maxblock;
	//copy A 
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		offs2 = (ii + i)*d + kk;
		for (size_t k = 0; k<kbsize; k++){
			A[row1 + k] = *(lPtr + offs2 + k);
		}
	}
	//copy B 
	for (size_t k = 0; k<kbsize; k++){
		offs2 = (kk + k)*d + jj;
		for (size_t j = 0; j<jbsize; j++){
			B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
		}
	}
	//multiply the 2 small blocks 
	for (size_t i = 0; i<bsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			row2 = j*bsize;
			T sum = C[i*bsize + j];
			for (size_t k = 0; k<kbsize; k++){
				sum += A[row1 + k] * B[row2 + k];
			}
			C[row1 + j] = sum;
		}
	}

	//copy back C
	for (size_t i = 0; i<bsize; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			*(resPtr + offs1 + j) = C[row2 + j];
		}
	}
}


/*
Bottom end blocks
*/
template<size_t bsize, typename T>
void matMultBlockBottomBlocks(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t jj){

	size_t d = lMat.getDim();

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	T A[bsize*bsize]; //left
	T B[bsize*bsize]; //right
	T C[bsize*bsize]; //result

	size_t maxblock = bsize*(d / bsize);
	size_t ii, kk; //block counters
	size_t ibsize, kbsize; //block sizes at borders
	size_t row1, row2, offs1, offs2; //pointer offsets


	ii = maxblock;
	ibsize = d - maxblock;
	//null C
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			C[row1 + j] = 0;
		}
	}
	//multiply full blocks
	for (size_t kk = 0; kk<maxblock; kk += bsize){
		//copy A 
		for (size_t i = 0; i<ibsize; i++){
			row1 = i*bsize;
			offs2 = (ii + i)*d + kk;
			for (size_t k = 0; k<bsize; k++){
				A[row1 + k] = *(lPtr + offs2 + k);
			}
		}
		//copy B 
		for (size_t k = 0; k<bsize; k++){
			offs2 = (kk + k)*d + jj;
			for (size_t j = 0; j<bsize; j++){
				B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
			}
		}
		//multiply small blocks 
		for (size_t i = 0; i<ibsize; i++){
			row1 = i*bsize;
			for (size_t j = 0; j<bsize; j++){
				row2 = j*bsize;
				T sum = C[i*bsize + j];
				for (size_t k = 0; k<bsize; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}
	//multiply last 2 blocks
	kk = maxblock;
	kbsize = d - maxblock;
	//copy A 
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		offs2 = (ii + i)*d + kk;
		for (size_t k = 0; k<kbsize; k++){
			A[row1 + k] = *(lPtr + offs2 + k);
		}
	}
	//copy B 
	for (size_t k = 0; k<kbsize; k++){
		offs2 = (kk + k)*d + jj;
		for (size_t j = 0; j<bsize; j++){
			B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
		}
	}
	//multiply the 2 small blocks 
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			row2 = j*bsize;
			T sum = C[i*bsize + j];
			for (size_t k = 0; k<kbsize; k++){
				sum += A[row1 + k] * B[row2 + k];
			}
			C[row1 + j] = sum;
		}
	}

	//copy back C
	for (size_t i = 0; i<ibsize; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*bsize;
		for (size_t j = 0; j<bsize; j++){
			*(resPtr + offs1 + j) = C[row2 + j];
		}
	}
}


/*
Bottom-right block
*/
template<size_t bsize, typename T>
void matMultBlockRightBottom(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat){

	size_t d = lMat.getDim();

	T* lPtr = lMat.getDataPointer();
	T* rPtr = rMat.getDataPointer();
	T* resPtr = mulMat.getDataPointer();

	T A[bsize*bsize]; //left
	T B[bsize*bsize]; //right
	T C[bsize*bsize]; //result

	size_t maxblock = bsize*(d / bsize);
	size_t ii, jj, kk; //block counters
	size_t ibsize, jbsize, kbsize; //block sizes at borders
	size_t row1, row2, offs1, offs2; //pointer offsets


	//////////////////////////////////////////////////////////////////////
	// calculate the bottom-right end block	
	ii = maxblock;
	jj = maxblock;
	ibsize = d - maxblock;
	jbsize = d - maxblock;
	//null C
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			C[row1 + j] = 0;
		}
	}
	//multiply full blocks
	for (size_t kk = 0; kk<maxblock; kk += bsize){
		//copy A 
		for (size_t i = 0; i<ibsize; i++){
			row1 = i*bsize;
			offs2 = (ii + i)*d + kk;
			for (size_t k = 0; k<bsize; k++){
				A[row1 + k] = *(lPtr + offs2 + k);
			}
		}
		//copy B 
		for (size_t k = 0; k<bsize; k++){
			offs2 = (kk + k)*d + jj;
			for (size_t j = 0; j<jbsize; j++){
				B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
			}
		}
		//multiply small blocks 
		for (size_t i = 0; i<ibsize; i++){
			row1 = i*bsize;
			for (size_t j = 0; j<jbsize; j++){
				row2 = j*bsize;
				T sum = C[i*bsize + j];
				for (size_t k = 0; k<bsize; k++){
					sum += A[row1 + k] * B[row2 + k];
				}
				C[row1 + j] = sum;
			}
		}
	}
	//multiply last 2 blocks
	kk = maxblock;
	kbsize = d - maxblock;
	//copy A 
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		offs2 = (ii + i)*d + kk;
		for (size_t k = 0; k<kbsize; k++){
			A[row1 + k] = *(lPtr + offs2 + k);
		}
	}
	//copy B 
	for (size_t k = 0; k<kbsize; k++){
		offs2 = (kk + k)*d + jj;
		for (size_t j = 0; j<jbsize; j++){
			B[j*bsize + k] = *(rPtr + offs2 + j); //B transpon!!
		}
	}
	//multiply the 2 small blocks 
	for (size_t i = 0; i<ibsize; i++){
		row1 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			row2 = j*bsize;
			T sum = C[i*bsize + j];
			for (size_t k = 0; k<kbsize; k++){
				sum += A[row1 + k] * B[row2 + k];
			}
			C[row1 + j] = sum;
		}
	}

	//copy back C
	for (size_t i = 0; i<ibsize; i++){
		offs1 = (ii + i)*d + jj;
		row2 = i*bsize;
		for (size_t j = 0; j<jbsize; j++){
			*(resPtr + offs1 + j) = C[row2 + j];
		}
	}
}









////////////////////////////////////////////////////////////////////////////////////////////////

/*
Multithreading matrix multiplication
*/

////////////////////////////////////////////////////////////////////////////////////////////////
/*
Block divided method
- uses many threads
- no option to control maximum cpu usage
*/


//////////////////////////////////////////////////////////////////////
//The FASTEST METHOD YET
//////////////////////////////////////////////////////////////////////
/*
Block matrix multiplication method
Benchmark wrapper
*/
template<size_t bsize, typename T>
Matrix<T> matMultMultiThreadBlock0(const Matrix<T>& lMat, const Matrix<T>& rMat){
	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	Matrix<T> resMat = matMultMultiThreadBlock<bsize, T>(lMat, rMat);

	//benchmark 
	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "Multithreaded Block matrix mutiplication using\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return resMat;
}


/*
Block matrix multiplication method
- using pointer access
- using temporary containers
- using precalculated pointer offsets
int the innermost loops
- using distinct for loops for border blocks
to avoid extensive border checking
with runtime variables
*/
template<size_t bsize, typename T>
Matrix<T> matMultMultiThreadBlock(const Matrix<T>& lMat, const Matrix<T>& rMat){

	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	size_t maxblock = bsize*(d / bsize);

	size_t noThreads = (d / bsize + 1)*(d / bsize + 1);
	std::thread *t = new std::thread[noThreads];	//thread array

	////////////////////////////////////////////////////////////////////
	// calculate full blocks	
	size_t tid = 0;
	for (size_t ii = 0; ii<maxblock; ii += bsize){
		for (size_t jj = 0; jj<maxblock; jj += bsize){
			t[tid] = std::thread(matMultBlockFullBlocks<bsize, T>, std::ref(lMat), std::ref(rMat), std::ref(mulMat), ii, jj);
			tid++;
		}
	}
	//////////////////////////////////////////////////////////////////////
	// calculate right end blocks	
	for (size_t ii = 0; ii<maxblock; ii += bsize){
		t[tid] = std::thread(matMultBlockRightBlocks<bsize, T>, std::ref(lMat), std::ref(rMat), std::ref(mulMat), ii);
		tid++;
	}
	//////////////////////////////////////////////////////////////////////
	// calculate bottom end blocks	
	for (size_t jj = 0; jj<maxblock; jj += bsize){
		t[tid] = std::thread(matMultBlockBottomBlocks<bsize, T>, std::ref(lMat), std::ref(rMat), std::ref(mulMat), jj);
		tid++;
	}
	//////////////////////////////////////////////////////////////////////
	// calculate bottom right end block	
	t[tid] = std::thread(matMultBlockRightBottom<bsize, T>, std::ref(lMat), std::ref(rMat), std::ref(mulMat));

	for (size_t tid = 0; tid<noThreads; tid++){	//wait for threads to finish
		t[tid].join();
	}

	delete[] t;

	return mulMat;
}




////////////////////////////////////////////////////////////////////////////////////////////////
/*
Row divided method
- In this method one can control the maximum number of threads
- In block divided that would not be elegant
*/

/*
Subroutine of multithreaded matrix multiplication
Calculates one row
Slave thread
*/
/*
template<typename T>
void matMultOneThreadNN(const Matrix<T>& lMat, const Matrix<T>& rMat, Matrix<T>& mulMat, size_t tid ,size_t bsize){
size_t d=lMat.getDim();
T* lPtr=lMat.getDataPointer();
T* rPtr=rMat.getDataPointer();
T* resPtr=mulMat.getDataPointer();
for(int i=tid*bsize;i<(tid+1)*bsize && i<d ;i++){ //loop over rows assigned
for(int j=0;j<d;j++){ 	//nullify elemets
*(resPtr+i*d+j)=0;
}
for(int k=0;k<d;k++){ 	//mult
T temp= *(lPtr+i*d+k);
for(int j=0;j<d;j++){
*(resPtr+i*d+j) += temp * *(rPtr+k*d+j);
}
}
}
}
*/
/*
Multithreaded matrix multiplication
1 thread - 1 row
Master thread
*/
/*
template<typename T>
Matrix<T> matMultMultiThreadNN(const Matrix<T>& lMat, const Matrix<T>& rMat ,size_t noThreads){
//benchmark
typedef std::chrono::high_resolution_clock myclock;
typedef std::chrono::microseconds microseconds;
myclock::time_point beg = myclock::now();
size_t d=lMat.getDim();
Matrix<T> mulMat=Matrix<T>(d);
size_t bsize=(d/noThreads)+1; 	//block size
std::thread t[noThreads];	//thread array
for(int tid=0;tid<noThreads;tid++){	//start threads
t[tid] = std::thread(matMultOneThreadNN<T> ,std::ref(lMat),std::ref(rMat),std::ref(mulMat),tid,bsize);
}
for(int tid=0;tid<noThreads;tid++){	//wait for threads to finish
t[tid].join();
}
//benchmark
myclock::time_point end = myclock::now();
microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
std::cout<<"\n\nMultithreaded matrix mutiplication \n";
std::cout<<"\n\nNumber of threads = "<<noThreads<<"\n";
std::cout<<"It took: "<<ms.count()/double(1e6)<<" s"<<"\n\n\n";
//std::cout<<d<<"\t"<<ms.count()/double(1e6)<<" s"<<"\n";
return mulMat;
}
*/









////////////////////////////////////////////////////////////////////////////////////////////////

/*
Strassen algorithm implementations
*/

////////////////////////////////////////////////////////////////////////////////////////////////


/*
Matrix multiplication using Strassen-algorithm
This in only the master function for benchmarking

*/
template<size_t bsize, typename T>
Matrix<T> matMultStrassen0(const Matrix<T>& lMat, const Matrix<T>& rMat){
	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	Matrix<T> resMat = matMultStrassenRec<bsize>(lMat, rMat);

	//benchmark 
	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "Strassen matrix mutiplication\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return resMat;
}


/*
Strassen matrix multiplication
- at END_OF_REC_STRASSEN size it changes back
to blocking multiplication
-faster from 4-500
*/
#define END_OF_REC_STRASSEN 256
template<size_t bsize, typename T>
Matrix<T> matMultStrassenRec(const Matrix<T>& lMat, const Matrix<T>& rMat){

	size_t d = lMat.getDim();
	Matrix<T> mulMat = Matrix<T>(d);

	//if matrices are "small" multiply them w block method
	if (d <= END_OF_REC_STRASSEN){
		mulMat = matMultBlock<bsize>(lMat, rMat);
	}
	else{
		//get zero padded blocks
		size_t db = (d + 1) / 2;
		Matrix<T> A11(db), A12(db), A21(db), A22(db), B11(db), B12(db), B21(db), B22(db);
		getPaddedBlockMats(lMat, A11, A12, A21, A22);
		getPaddedBlockMats(rMat, B11, B12, B21, B22);

		//run Strassen algorithm to get the M matrices	
		Matrix<T> M1 = matMultStrassenRec<bsize>(A11 + A22, B11 + B22);
		Matrix<T> M2 = matMultStrassenRec<bsize>(A21 + A22, B11);
		Matrix<T> M3 = matMultStrassenRec<bsize>(A11, B12 - B22);
		Matrix<T> M4 = matMultStrassenRec<bsize>(A22, B21 - B11);
		Matrix<T> M5 = matMultStrassenRec<bsize>(A11 + A12, B22);
		Matrix<T> M6 = matMultStrassenRec<bsize>(A21 - A11, B11 + B12);
		Matrix<T> M7 = matMultStrassenRec<bsize>(A12 - A22, B21 + B22);

		//sum and write back blocks	
		setPaddedBlockMats(mulMat, M1 + M4 - M5 + M7, M3 + M5, M2 + M4, M1 - M2 + M3 + M6);
	}

	return mulMat;
}

/*
gets 2x2 block matrices of  a matrix
-if d is even its ok
-if d is odd then
-first 1 column, and row is "added"
-then it is even
*/
template<typename T>
void getPaddedBlockMats(const Matrix<T>& inMat, Matrix<T>& A11, Matrix<T>& A12, Matrix<T>& A21, Matrix<T>& A22){

	size_t d0 = inMat.getDim();
	size_t d = (d0 + 1) / 2;

	//get pointers
	T* inPtr = inMat.getDataPointer();
	T* A11Ptr = A11.getDataPointer();
	T* A12Ptr = A12.getDataPointer();
	T* A21Ptr = A21.getDataPointer();
	T* A22Ptr = A22.getDataPointer();

	//if its even
	if (d0 % 2 == 0){
		//copy data
		for (size_t i = 0; i<d; i++){
			size_t offset = i*d;
			//the no zero part
			for (size_t j = 0; j<d; j++){
				*(A11Ptr + offset + j) = *(inPtr + i*d0 + j);
				*(A12Ptr + offset + j) = *(inPtr + i*d0 + j + d);
				*(A21Ptr + offset + j) = *(inPtr + (i + d)*d0 + j);
				*(A22Ptr + offset + j) = *(inPtr + (i + d)*d0 + j + d);
			}
		}
	}
	//if odd
	else{
		//copy data
		for (size_t i = 0; i<d - 1; i++){
			size_t offset = i*d;
			//the no zero part
			for (size_t j = 0; j<d - 1; j++){
				*(A11Ptr + offset + j) = *(inPtr + i*d0 + j);
				*(A12Ptr + offset + j) = *(inPtr + i*d0 + j + d);
				*(A21Ptr + offset + j) = *(inPtr + (i + d)*d0 + j);
				*(A22Ptr + offset + j) = *(inPtr + (i + d)*d0 + j + d);
			}
			//last columns
			*(A11Ptr + offset + d - 1) = *(inPtr + i*d0 + d - 1);
			*(A12Ptr + offset + d - 1) = 0;
			*(A21Ptr + offset + d - 1) = *(inPtr + (i + d)*d0 + d - 1);
			*(A22Ptr + offset + d - 1) = 0;
		}
		//last rows 
		for (size_t j = 0; j<d - 1; j++){
			//last rows 
			*(A11Ptr + (d - 1)*d + j) = *(inPtr + (d - 1)*d0 + j);
			*(A12Ptr + (d - 1)*d + j) = *(inPtr + (d - 1)*d0 + j + d);
			*(A21Ptr + (d - 1)*d + j) = 0;
			*(A22Ptr + (d - 1)*d + j) = 0;
		}
		//right bottom
		*(A11Ptr + (d - 1)*d + d - 1) = *(inPtr + (d - 1)*d0 + d - 1);
		*(A12Ptr + (d - 1)*d + d - 1) = 0;
		*(A21Ptr + (d - 1)*d + d - 1) = 0;
		*(A22Ptr + (d - 1)*d + d - 1) = 0;
	}
}


/*
sets 2x2 block matrices of  a matrix
-only sets values not padded in input matrices
*/
template<typename T>
void setPaddedBlockMats(Matrix<T>& inMat, const Matrix<T>& A11, const  Matrix<T>& A12, const Matrix<T>& A21, const Matrix<T>& A22){

	size_t d0 = inMat.getDim();
	size_t d = (d0 + 1) / 2;

	//get pointers
	T* inPtr = inMat.getDataPointer();
	T* A11Ptr = A11.getDataPointer();
	T* A12Ptr = A12.getDataPointer();
	T* A21Ptr = A21.getDataPointer();
	T* A22Ptr = A22.getDataPointer();

	//if its even
	if (d0 % 2 == 0){
		//copy data
		for (size_t i = 0; i<d; i++){
			size_t offset = i*d;
			//the no zero part
			for (size_t j = 0; j<d; j++){
				*(inPtr + i*d0 + j) = *(A11Ptr + offset + j);
				*(inPtr + i*d0 + j + d) = *(A12Ptr + offset + j);
				*(inPtr + (i + d)*d0 + j) = *(A21Ptr + offset + j);
				*(inPtr + (i + d)*d0 + j + d) = *(A22Ptr + offset + j);
			}
		}
	}
	//if odd
	else {
		//copy data
		for (size_t i = 0; i<d - 1; i++){
			size_t offset = i*d;
			//the no zero part
			for (size_t j = 0; j<d - 1; j++){
				*(inPtr + i*d0 + j) = *(A11Ptr + offset + j);
				*(inPtr + i*d0 + j + d) = *(A12Ptr + offset + j);
				*(inPtr + (i + d)*d0 + j) = *(A21Ptr + offset + j);
				*(inPtr + (i + d)*d0 + j + d) = *(A22Ptr + offset + j);
			}
			//last columns
			*(inPtr + i*d0 + d - 1) = *(A11Ptr + offset + d - 1);
			*(inPtr + (i + d)*d0 + d - 1) = *(A21Ptr + offset + d - 1);
		}
		//last rows 
		for (size_t j = 0; j<d - 1; j++){
			//last rows 
			*(inPtr + (d - 1)*d0 + j) = *(A11Ptr + (d - 1)*d + j);
			*(inPtr + (d - 1)*d0 + j + d) = *(A12Ptr + (d - 1)*d + j);
		}
		//right bottom
		*(inPtr + (d - 1)*d0 + d - 1) = *(A11Ptr + (d - 1)*d + d - 1);
	}
}





//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/*
	OpenCL matmult
*/
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//type can be changed
#define TYPE cl_double
//blocks size on GPU
//has to be set here, and in kernel too!!!
	//here for worksize calculation
#define CL_BLOCK_SIZE 4


//////////////////////////////////////////////////////////////////////////
/*
	Class for easier work with opencl
*/
//////////////////////////////////////////////////////////////////////////

template<typename T>
class Matmult_opencl{

	/*
		Constr, destr
			no copy, this object should not be copied
	*/
public:
	Matmult_opencl(size_t d) : resMat(d) {}
	~Matmult_opencl() {}


	/* 
		data 
	*/
private:
	//opencl related objects, and data
	cl_context context;
	cl_program program;
	cl_command_queue commandQueue;
	cl_kernel my_kernel;
	//buffers on the GPU (other device)
	cl_mem dev_buf_1, dev_buf_2, dev_buf_3;

public:
	//result matrix
	Matrix<T> resMat;

	/*
		Methods
	*/
public:
	/*
		Init function, to deal with most opencl stuff
			- choose opencl platfrom, and device
			- create context, commandque
			- load kernel files, and build them

		Its quite long and verbose
			- might be changed in the future
	*/
	int opencl_initialize(size_t platform_choice, size_t device_choice)
	{
		cl_int	status = 0;

		//Getting OpenCL platforms and choose an available one.
		cl_uint numPlatforms;				//the NO. of platforms
		cl_platform_id* platforms = NULL; 	//id of available platforms
		cl_platform_id 	platform = NULL;	//id of the chosen platform

		//getting NO. of platforms
		status = clGetPlatformIDs(0, NULL, &numPlatforms);
		if (status != CL_SUCCESS)
		{
			std::cerr << "Error: Getting platforms!" << std::endl;
			std::cerr << "Error number= " << status << std::endl;
			return status;
		}

		//Choosing platform
		if (numPlatforms > 0)
		{
			//getting platform ids
			platforms = new cl_platform_id[numPlatforms];
			status = clGetPlatformIDs(numPlatforms, platforms, NULL);

			//printing platform names
			std::cout << "\nPlatform info:" << std::endl;
			for (unsigned int i = 0; i<numPlatforms; i++)
			{
				//get platform name size
				size_t platform_name_size;
				status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_size);

				//get platform name
				char* platform_name = new char[platform_name_size];
				status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);


				//get platform version size
				size_t platform_version_size;
				status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, NULL, &platform_version_size);

				//get platform version
				char* platform_version = new char[platform_version_size];
				status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, platform_version_size, platform_version, NULL);


				//print info
				std::cout << i << ". platform:\t" << platform_name << "\n";
				std::cout << i << ". version:\t " << platform_version << "\n";
				std::cout << std::endl;

				delete[] platform_name;
				delete[] platform_version;
			}


			//choosing platform
			std::cout << "\nChoose platform: (0)" << std::endl;
			/*int platform_choice = 0;
			std::string temp_line;
			getline(std::cin, temp_line);
			std::stringstream temp_sstr;
			temp_sstr << temp_line;
			temp_sstr >> platform_choice;
			std::cout << "platform choice" << platform_choice << std::endl;
			*/
			platform = platforms[platform_choice];

			delete[] platforms;
		}

		//Query the platform and choose the  device
		cl_uint		numDevices = 0; 	//NO. of devices
		cl_device_id	*devices;		// device ids
		cl_device_id	device;			//id of chosen device

		//getting number of devices
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		devices = new cl_device_id[numDevices];

		//getting device ids
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

		//printing device info
		std::cout << "\nDevice info:" << std::endl;
		for (unsigned int i = 0; i<numDevices; i++)
		{
			//get device vendor size
			size_t device_vendor_size;
			status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, 0, NULL, &device_vendor_size);

			//get device vendor
			char* device_vendor = new char[device_vendor_size];
			status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, device_vendor_size, device_vendor, NULL);


			//get device name size
			size_t device_name_size;
			status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &device_name_size);

			//get device name
			char* device_name = new char[device_name_size];
			status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, device_name_size, device_name, NULL);


			//get devicetype 
			cl_device_type device_type;
			status = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);


			//get device version size
			size_t device_version_size;
			status = clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 0, NULL, &device_version_size);

			//get device version
			char* device_version = new char[device_version_size];
			status = clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, device_version_size, device_version, NULL);


			//print info
			std::cout << i << ". device vendor:\t" << device_vendor << std::endl;
			std::cout << "            name:\t" << device_name << std::endl;

			//device type 
			if (device_type == CL_DEVICE_TYPE_CPU)
				std::cout << "            type:\tCPU" << std::endl;
			if (device_type == CL_DEVICE_TYPE_GPU)
				std::cout << "            type:\tGPU" << std::endl;
			if (device_type == CL_DEVICE_TYPE_ACCELERATOR)
				std::cout << "            type:\tACCELERATOR" << std::endl;
			if (device_type == CL_DEVICE_TYPE_DEFAULT)
				std::cout << "            type:\tDEFAULT" << std::endl;


			std::cout << "            version:\t" << device_version << std::endl;

			delete[] device_vendor;
			delete[] device_name;
			delete[] device_version;
		}

		//choosing device
		std::cout << "\nChoose device: (0)" << std::endl;
		/*int device_choice = 0;
		std::string temp_line1;
		getline(std::cin, temp_line1);
		std::stringstream temp_sstr1;
		temp_sstr1 << temp_line1;
		temp_sstr1 >> device_choice;
		*/
		device = devices[device_choice];


		//Create context
		context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
		if (status != 0)
		{
			std::cerr << "ERROR creating context: " << status << std::endl;
			return status;
		}

		//Creating command queue associated with the context
		commandQueue = clCreateCommandQueue(context, device, 0, &status);
		if (status != 0)
		{
			std::cerr << "ERROR creating commandqueue: " << status << std::endl;
			return status;
		}

		//open kernel file and convert it to char array
		//const char *filename = kernel_filename.c_str();
		std::string sourceStr = convertToString("test_mult.cl");
		const char *source = sourceStr.c_str();
		size_t sourceSize[] = { strlen(source) };

		//Create program object
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
		if (status != 0)
		{
			std::cout << "ERROR creating program: " << status << std::endl;
			return status;
		}

		//Building program 
		//only prints log if there was an error
		//if there are only warnings it is not printed
		status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
		if (status != 0)
		{
			//print ERROR but do not quit, there may be just warnings
			std::cerr << "ERROR building program: " << status << std::endl;

			//Getting build log size
			size_t logsize = 0;
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
			std::cout << logsize << std::endl;

			//Getting build log
			char* log = new char[logsize];
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);

			//print log info
			std::cout << "log:\n " << log << std::endl;
			delete[] log;

			return status;
		}

		return status;
	}
	
	//create kernel
	int opencl_create_kernel(std::string kernel_name)
	{
		//error variable
		cl_int status = 0;

		// Create kernel object
		my_kernel = clCreateKernel(program, kernel_name.c_str(), &status);
		if (status != 0)
		{
			std::cerr << "ERROR creating kernel: " << status << std::endl;
			return status;
		}

		return status;
	}

private:
	//load the kernel file into a null terminated string 
	std::string convertToString(std::string infilename)
	{
		std::string str;

		// load the kernel file into a null terminated string 
		//open file in binary i/o 
		std::ifstream infile(infilename.c_str(), std::ios::binary | std::ios::ate);
		//check file 
		if (!(infile))
		{
			std::cout << "\nERROR CAN'T OPEN KERNEL FILE: " << infilename << "\n" << std::endl;
			return NULL;
		}

		//get the size of the file 
		std::ifstream::pos_type size;
		size = infile.tellg();
		//go to the begginging of file								 
		infile.seekg(0, std::ios::beg);

		//read file
		str.resize(size);
		infile.read((char*)(str.c_str()), size);
		//append "\0"
		str += '\0';

		return str;
	}

public:
	//allocates buffers on devices
	int opencl_copy_mem(const Matrix<T>& lMat, const Matrix<T>& rMat)
	{
		//error variable
		cl_int status = 0;

		// Allocate memory on device

		size_t d0 = lMat.getDim();

		//data1
		//double not template!!!
		dev_buf_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d0 * d0 * sizeof(TYPE), lMat.getDataPointer(), &status);
		//error check
		if (status != 0)
		{
			std::cerr << "ERROR creating buffers: " << status << std::endl;
			return status;
		}
		//data2
		dev_buf_2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d0 * d0 * sizeof(TYPE), rMat.getDataPointer(), &status);
		//error check
		if (status != 0)
		{
			std::cerr << "ERROR creating buffers: " << status << std::endl;
			return status;
		}
		//data3
		dev_buf_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, d0 * d0 * sizeof(TYPE), NULL, &status);
		//error check
		if (status != 0)
		{
			std::cerr << "ERROR creating buffers: " << status << std::endl;
			return status;
		}

		return status;

	}

	//Setting kernel arguments
	//with the buffers on the GPU (device)
	int set_kern_arg(size_t d)
	{
		//error variable
		cl_int status = 0;

		//there is no size_t in opencl kernel
		unsigned int dui = (unsigned int)d;

		//kernel1

		status = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), &dev_buf_1);
		status |= clSetKernelArg(my_kernel, 1, sizeof(cl_mem), &dev_buf_2);
		status |= clSetKernelArg(my_kernel, 2, sizeof(cl_mem), &dev_buf_3);
		status |= clSetKernelArg(my_kernel, 3, sizeof(unsigned int), &dui);

		//error check
		if (status != 0)
		{
			std::cerr << "ERROR setting kernel arguments: " << status << std::endl;
			return status;
		}

		return status;
	}

	//this function launches the kernel
	//and reads back result from host(GPU) if needed
	int call_kernel(size_t d, size_t worksize)
	{
		//error variable
		cl_int status = 0;

		//size_t worksize = d*d;
		// Running the kernel.
		size_t global_work_size[1] = { worksize };
		status = clEnqueueNDRangeKernel(commandQueue, my_kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		if (status != 0)
		{
			std::cerr << "ERROR running kernel1: " << status << std::endl;
			return status;
		}

		//Read the result back to host memory.
		status = clEnqueueReadBuffer(commandQueue, dev_buf_3, CL_TRUE, 0, d*d * sizeof(TYPE), this->resMat.getDataPointer(), 0, NULL, NULL);
		if (status != 0)
		{
			std::cout << "ERROR reading buffer: " << status << std::endl;
		}
		return status;
	}

};


//////////////////////////////////////////////////////////////////////////
/*
	the test multiplication method
		naive matmult
*/
//////////////////////////////////////////////////////////////////////////
template<typename T>
Matrix<T> gpu_test_mult(const Matrix<T>& lMat, const Matrix<T>& rMat, size_t platform_choice, size_t device_choice){

	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	std::string kernel_name = "test_mult";

	Matmult_opencl<T> my_mat_mult( lMat.getDim() );
	//init
	my_mat_mult.opencl_initialize( platform_choice, device_choice);
	//compile kernel, copy data
	my_mat_mult.opencl_copy_mem(lMat, rMat);
	//create kernel
	my_mat_mult.opencl_create_kernel(kernel_name);
	//set args
	my_mat_mult.set_kern_arg(lMat.getDim());
	//do the mutplications
	my_mat_mult.call_kernel(lMat.getDim(), lMat.getDim() * lMat.getDim());

	//benchmark 
	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "GPU naive matrix mutiplication\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return my_mat_mult.resMat;
}



//////////////////////////////////////////////////////////////////////////
/*
	Block matrix multiplication method
		- kernel name is parameter, so it can use different kernels
			Best is block_mult_2	
			- using pointer access
			- using temporary containers
			- using precalculated pointer offsets
				in the innermost loops
			- using zero padding for border blocks
*/
//////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T> gpu_matMultMultiThreadBlock(std::string kernel_name,const Matrix<T>& lMat, const Matrix<T>& rMat, size_t platform_choice, size_t device_choice){
	//benchmark 
	typedef std::chrono::high_resolution_clock myclock;
	typedef std::chrono::microseconds microseconds;
	myclock::time_point beg = myclock::now();

	Matmult_opencl<T> my_mat_mult(lMat.getDim());
	//init
	my_mat_mult.opencl_initialize(platform_choice, device_choice);
	//compile kernel, copy data
	my_mat_mult.opencl_copy_mem(lMat, rMat);
	//create kernel
	my_mat_mult.opencl_create_kernel(kernel_name);
	//set args
	my_mat_mult.set_kern_arg(lMat.getDim());

	//worksize calculation
	size_t d = lMat.getDim();
	size_t worksize = ((d + CL_BLOCK_SIZE - 1) / CL_BLOCK_SIZE)*((d + CL_BLOCK_SIZE - 1)/ CL_BLOCK_SIZE);

	//do the mutplications
	my_mat_mult.call_kernel(d, worksize);

	//benchmark 
	myclock::time_point end = myclock::now();
	microseconds ms = std::chrono::duration_cast<microseconds>(end - beg);
	std::cout << "GPU blocked matrix mutiplication\n";
	std::cout << "took: " << ms.count() / double(1e6) << " s" << "\n\n";

	return my_mat_mult.resMat;
}


#endif
