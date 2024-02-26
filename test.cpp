// Kalme.cpp : 定义控制台应用程序的入口点。
//
 
// #include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>//包含Eigen矩阵运算库，用于矩阵计算
#include <cmath>
#include <limits>//用于生成随机分布数列
 
using namespace std;
using Eigen::MatrixXd;
double generateGaussianNoise(double mu, double sigma);

int main(int argc, char* argv[])
{
	ofstream fout("..//kalman//result.txt");
	
	double generateGaussianNoise(double mu, double sigma);//随机高斯分布数列生成器函数
 
	const double delta_t = 0.1;//控制周期，100ms
	const int num = 100;//迭代次数
	const double acc = 10;//加速度，ft/m
 
	MatrixXd A(2,2);
	A(0,0) = 1;
	A(1,0) = 0;
	A(0,1) = delta_t;
	A(1,1) = 1;
 
	MatrixXd B(2,1);
	B(0,0) = pow(delta_t,2)/2;
	B(1,0) = delta_t;
 
	MatrixXd H(1,2);//测量的是小车的位移，速度为0
	H(0,0) = 1;
	H(0,1) = 0;
	
	MatrixXd Q(2,2);//过程激励噪声协方差，假设系统的噪声向量只存在速度分量上，且速度噪声的方差是一个常量0.01，位移分量上的系统噪声为0
	Q(0,0) = 0;
	Q(1,0) = 0;
	Q(0,1) = 0;
	Q(1,1) = 0.01;
 
	MatrixXd R(1,1);//观测噪声协方差，测量值只有位移，它的协方差矩阵大小是1*1，就是测量噪声的方差本身。
	R(0,0) = 10;
 
	//time初始化，产生时间序列
	vector<double> time(100, 0);
	for(decltype(time.size()) i = 0; i != num; ++i){
		time[i] = i * delta_t;
		//cout<<time[i]<<endl;
	}
 
	MatrixXd X_real(2,1);
	vector<MatrixXd> x_real, rand;
	//生成高斯分布的随机数
	for(int i = 0; i<100;++i){
		MatrixXd a(1,1);
		a(0,0) = generateGaussianNoise(0,sqrt(10));
		rand.push_back(a);
	}
	//生成真实的位移值
	for(int i = 0; i < num; ++i){
		X_real(0,0) = 0.5 * acc * pow(time[i],2);
		X_real(1,0) = 0;
		x_real.push_back(X_real);
	}
 
	//变量定义，包括状态预测值，状态估计值，测量值，预测状态与真实状态的协方差矩阵，估计状态和真实状态的协方差矩阵，初始值均为零
	MatrixXd X_evlt = MatrixXd::Constant(2,1,0), X_pdct = MatrixXd::Constant(2,1,0), Z_meas = MatrixXd::Constant(1,1,0), 
		Pk = MatrixXd::Constant(2,2,0), Pk_p = MatrixXd::Constant(2,2,0), K = MatrixXd::Constant(2,1,0);
	vector<MatrixXd> x_evlt, x_pdct, z_meas, pk, pk_p, k;
	x_evlt.push_back(X_evlt);
	x_pdct.push_back(X_pdct);
	z_meas.push_back(Z_meas);
	pk.push_back(Pk);
	pk_p.push_back(Pk_p);
	k.push_back(K);
 
	//开始迭代
	for(int i = 1; i < num; ++i){
		//预测值
		X_pdct = A * x_evlt[i-1] + B * acc;
		x_pdct.push_back(X_pdct);
		//预测状态与真实状态的协方差矩阵，Pk'
		Pk_p = A * pk[i-1] * A.transpose() + Q;
		pk_p.push_back(Pk_p);
		//K:2x1
		MatrixXd tmp(1,1);
		tmp = H * pk_p[i] * H.transpose() + R;
		K = pk_p[i] * H.transpose() * tmp.inverse();
		k.push_back(K);
		//测量值z
		Z_meas = H * x_real[i] + rand[i];
		z_meas.push_back(Z_meas);
		//估计值
		X_evlt = x_pdct[i] + k[i] * (z_meas[i] - H * x_pdct[i]);
		x_evlt.push_back(X_evlt);
		//估计状态和真实状态的协方差矩阵，Pk
		Pk = (MatrixXd::Identity(2,2) - k[i] * H) * pk_p[i];
		pk.push_back(Pk);
	}
	
	cout<<"含噪声测量"<<"  "<<"后验估计"<<"  "<<"真值"<<"  "<<endl;
	for(int i = 0; i < num; ++i){
		//cout<<z_meas[i]<<"  "<<x_evlt[i](0,0)<<"  "<<x_real[i](0,0)<<endl;
		fout<<z_meas[i]<<"  "<<x_evlt[i](0,0)<<"  "<<x_real[i](0,0)<<endl;//输出到txt文档，用于matlab绘图
		//cout<<k[i](1,0)<<endl;
		//fout<<rand[i](0,0)<<endl;
		//fout<<x_pdct[i](0,0)<<endl;
	}
 
	fout.close();
 
	return 0;
}
 
//生成高斯分布随机数的函数，网上找的
double generateGaussianNoise(double mu, double sigma)
{
    const double epsilon = std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;
 
    static double z0, z1;
    static bool generate;
    generate = !generate;
 
    if (!generate)
       return z1 * sigma + mu;
 
    double u1, u2;
    do
     {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
     }
    while ( u1 <= epsilon );
 
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}