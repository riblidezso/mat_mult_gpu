#ifndef VECTOR_H 
#define VECTOR_H

#include <iostream>
#include <string>
//#include <fstream>
//#include <vector>
#include <sstream>
//#include <random>
//#include <map>
//#include <float.h>

//minimal vector class
//for vector operations w matrices
template<typename T>
class Vector{
	//constructors and destructors
	public:
	Vector(size_t  ind);//empty
	~Vector();
	
	//data
	private:
	T* data;
	size_t d;

	//methods
	public:
	T get(int i) const;
	size_t getDim() const;
	void set(int i,T value);
	void operator=(const Vector<T>& VectorToCopy);
	std::string toString(); //toString

	};

/*
	 Empty constructor
*/
template<typename T>
Vector<T>::Vector(size_t ind){
	d=ind;
	data = new T[d];
	}
	
/*
	 Destructor
*/
template<typename T>
Vector<T>::~Vector(){
	delete[] data;
	}
	

/*
	Get element
*/
template<typename T>
T Vector<T>::get(int i) const{
	return data[i];
	}
		
/*
	Getdim
*/
template<typename T>
size_t Vector<T>::getDim() const {
	return d;
	}
	

/*
	Set element
*/
template<typename T>
void Vector<T>::set(int i,T value){
	data[i]=value;
	}


/*
	Assignement with operator
*/
template<typename T>
void Vector<T>::operator=(const Vector<T>& VectorToCopy){
	for(int i=0;i<d;i++){
		data[i]=VectorToCopy.get(i);
		}
	}

/*
	toString method
*/
template<typename T>
std::string Vector<T>::toString(){
	std::stringstream sstr;
	for(int i=0;i<d;i++){
		sstr<<data[i]<<"\t";
		}
	return sstr.str();
	}
#endif
