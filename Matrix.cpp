#include "matrix.h"
#include <iostream>
#include <cstdlib> 
#include <ctime>
#include <cmath>

Matrix::Matrix(int r, int c) : rows(r) , col(c) {data = std::vector<std::vector<double>>(r, std::vector<double>(c,0.0));}

Matrix::Matrix() : rows(0), col(0) {data.clear();}

Matrix::Matrix(vector<vector<double>> d){
	if(d.empty()){
        std::cerr << "Input vector cannot be empty!"<< std::endl;
    }
    size_t num_cols = d[0].size();
    for (const auto& row : d) {
        if (row.size() != num_cols) {
        std::cerr << "All rows in the input vector must have the same length." << std::endl;
        }
    }
    rows = static_cast<int>(d.size());
    col = static_cast<int>(num_cols);
    data = d;
}

Matrix::Matrix(int r, int c, bool isRandom){ 
	rows = r;
    col = c;
    data = std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0));
    if(isRandom){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }
}

void Matrix::print_matrix() const {
	for(const auto& row : data){
        for(double val : row){
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::operator*(const Matrix& mat) const{
	int r1 = rows;
    int c1 = col;
    int r2 = mat.rows;
    int c2 = mat.col;
    if(c1 != r2){
        std::cerr << "Invalid Input: Matrix dimensions incompatible for multiplication." << std::endl;
        return Matrix();
    }
    std::vector<std::vector<double>> result(r1, std::vector<double>(c2, 0.0));
    for(int i = 0; i < r1; i++){
        for(int j = 0; j < c2; j++){
            for(int k = 0; k < c1; k++){
                result[i][j] += data[i][k] * mat.data[k][j];
            }
        }
    }
	return Matrix(result);
}

Matrix Matrix::operator+(const Matrix& mat) const{
	if(rows != mat.rows || col != mat.col){
        std::cerr << "Invalid Input: Matrix dimensions must be equal for addition." << std::endl;
        return Matrix();
    }
    std::vector<std::vector<double>> result(rows, std::vector<double>(col));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result[i][j] = data[i][j] + mat.data[i][j];
        }
    }
	return Matrix(result);
}

Matrix Matrix::operator-(const Matrix& mat) const{
	if(rows != mat.rows || col != mat.col){
        std::cerr << "Invalid Input: Matrix dimensions must be equal for subtraction." << std::endl;
        return Matrix();
    }
    std::vector<std::vector<double>> result(rows, std::vector<double>(col));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result[i][j] = data[i][j] - mat.data[i][j];
        }
    }
	return Matrix(result);
}

Matrix Matrix::operator*(double num) const{
	std::vector<std::vector<double>> result(rows, std::vector<double>(col));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result[i][j] = data[i][j] * num;
        }
    }
	return Matrix(result);
}

Matrix Matrix::exp_multiply() const{
	std::vector<std::vector<double>> result(rows, std::vector<double>(col));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result[i][j] = std::exp(data[i][j]);
        }
    }
	return Matrix(result);
}

Matrix Matrix::scalar_subtraction(double num) const{
    std::vector<std::vector<double>> result(rows, std::vector<double>(col));
    for(int i=0; i< rows; i++){
        for(int j=0; j<col ; j++){
            result[i][j] = data[i][j] - num;
        }
    }
    return Matrix(result);
}

Matrix Matrix::transpose() const{
    std::vector<std::vector<double>> result_data(col, std::vector<double>(rows));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result_data[j][i] = data[i][j];
        }
    }
    return Matrix(result_data);
   }
   
Matrix Matrix::element_sqrt() const{
    std::vector<std::vector<double>> result_data(rows, std::vector<double>(col, 0.0));
       for(int i = 0; i < rows; i++){
           for(int j = 0; j < col; j++){
               result_data[i][j] = std::sqrt(data[i][j]);
           }
       }
    return Matrix(result_data);
}
   
Matrix Matrix::element_divide(const Matrix& mat) const{
    if(rows != mat.rows || col != mat.col){
        std::cerr << "Invalid Input: Matrices must be the same size for element-wise division." << std::endl;
        return Matrix();
    }
    std::vector<std::vector<double>> result_data(rows, std::vector<double>(col, 0.0));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            if (mat.data[i][j] != 0.0) {
                result_data[i][j] = data[i][j] / mat.data[i][j];
            } else {
                result_data[i][j] = 0.0; 
            }
        }
    }
    return Matrix(result_data);
}
   
Matrix Matrix::operator/(double num) const{
    std::vector<std::vector<double>> result_data(rows, std::vector<double>(col, 0.0));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result_data[i][j] = data[i][j] / num;
        }
    }
    return Matrix(result_data);
}

Matrix Matrix::element_multiply(const Matrix& mat) const{
	if(rows != mat.rows || col != mat.col){
        std::cerr << "Invalid Input: Matrices must be the same size for element-wise multiplication." << std::endl;
        return Matrix();
    }
    std::vector<std::vector<double>> result_data(rows, std::vector<double>(col, 0.0));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < col; j++){
            result_data[i][j] = data[i][j] * mat.data[i][j];
        }
    }
	return Matrix(result_data);
}