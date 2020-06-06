#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>



class Accel {
	std::string fp;
	std::string kernels;
	std::vector<cl::Platform> platforms;
	cl::Platform default_platform;
	std::vector<cl::Device> devices;
	cl::Device default_device;
	cl::Context* context;
	cl::Program::Sources sources;
	cl::Program* program;
	cl::CommandQueue* queue;

public:
	Accel(std::string filepath);

	std::string ParseKernels(const std::string filepath);

	std::vector<float> call1v(std::vector<float> a, std::string operation, float val);

	std::vector<float> call1v(std::vector<float> a, std::string operation);

	std::vector<float> call2v(std::vector<float> a, std::vector<float> b, std::string operation);

	std::vector<std::vector<float>> callActivationFunction(std::vector<float> v, std::string operation);

	std::vector<std::vector<float>> callActivationFunction(std::vector<float> v, std::string operation, float val);

	std::vector<float> vec(int val, int size);

	void print_kernels() {
		std::cout << kernels << std::endl;
	}
};

