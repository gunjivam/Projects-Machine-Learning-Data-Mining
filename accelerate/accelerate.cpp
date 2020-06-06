#include "accelerate.h"


Accel::Accel(std::string filepath) : fp(filepath) {
	kernels = ParseKernels(filepath);

	cl::Platform::get(&platforms);
	default_platform = platforms[0];

	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	default_device = devices[0];

	context = new cl::Context({ default_device });


	sources.push_back({ kernels.c_str(), kernels.length() });

	program = new cl::Program(*context, sources);

	if ((*program).build({ default_device }) != CL_SUCCESS) {
		std::cout << " error building: " << (*program).getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

	queue = new cl::CommandQueue(*context, default_device);

}

std::vector<float> Accel::call1v(std::vector<float> a, std::string operation, float val) {
	const int LIST_SIZE = a.size();
	cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);

	std::vector<float>c(LIST_SIZE);

	queue->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &a[0]);

	cl::Kernel k = cl::Kernel(*program, operation.c_str());

	k.setArg(1, val);
	k.setArg(0, buffer_A);
	k.setArg(2, buffer_C);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &c[0]);
	queue->finish();

	return c;
}

std::vector<float> Accel::call1v(std::vector<float> a, std::string operation) {
	const int LIST_SIZE = a.size();
	cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);

	std::vector<float>c(LIST_SIZE);

	queue->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &a[0]);

	cl::Kernel k = cl::Kernel(*program, operation.c_str());

	k.setArg(0, buffer_A);
	k.setArg(1, buffer_C);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &c[0]);
	queue->finish();

	return c;
}

std::vector<float> Accel::call2v(std::vector<float> a, std::vector<float> b, std::string operation) {
	const int LIST_SIZE = a.size();

	cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_B(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);

	std::vector<float>c(LIST_SIZE);


	queue->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &a[0]);
	queue->enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &b[0]);

	cl::Kernel k = cl::Kernel(*program, operation.c_str());

	k.setArg(1, buffer_B);
	k.setArg(0, buffer_A);
	k.setArg(2, buffer_C);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &c[0]);
	queue->finish();

	return c;
}


std::vector<float> Accel::vec(int val, int size) {
	std::vector<float> c(size);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, size*sizeof(float));

	cl::Kernel k = cl::Kernel(*program, "init_vec");
	k.setArg(1, buffer_C);
	k.setArg(0, val);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * size, &c[0]);
	queue->finish();
	return c;
}

std::vector<std::vector<float>> Accel::callActivationFunction(std::vector<float> v, std::string operation) {
	const int LIST_SIZE = v.size();

	cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_D(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);

	std::vector<float>c(LIST_SIZE);
	std::vector<float>d(LIST_SIZE);

	queue->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &v[0]);

	cl::Kernel k = cl::Kernel(*program, operation.c_str());

	k.setArg(1, buffer_C);
	k.setArg(0, buffer_A);
	k.setArg(2, buffer_D);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &c[0]);
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &d[0]);
	queue->finish();

	std::vector<std::vector<float>> res = { c, d };
	return res;
}

std::vector<std::vector<float>> Accel::callActivationFunction(std::vector<float> v, std::string operation, float val) {
	const int LIST_SIZE = v.size();

	cl::Buffer buffer_A(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_C(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);
	cl::Buffer buffer_D(*context, CL_MEM_READ_WRITE, sizeof(float) * LIST_SIZE);

	std::vector<float>c(LIST_SIZE);
	std::vector<float>d(LIST_SIZE);

	queue->enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &v[0]);

	cl::Kernel k = cl::Kernel(*program, operation.c_str());

	k.setArg(2, buffer_C);
	k.setArg(1, val);
	k.setArg(0, buffer_A);
	k.setArg(3, buffer_D);

	queue->enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &c[0]);
	queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * LIST_SIZE, &d[0]);
	queue->finish();

	std::vector<std::vector<float>> res = { c, d };
	return res;
}


std::string Accel::ParseKernels(const std::string filepath) {
	std::ifstream stream(filepath);
	std::string line;
	std::stringstream ss;

	while (std::getline(stream, line)) {
		ss << line << '\n';
	}
	return ss.str();
}
