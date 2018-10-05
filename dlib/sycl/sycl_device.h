#ifndef SYCLDEVICE_H
#define SYCLDEVICE_H
#include <mutex>
#include <CL/sycl.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

/*
 * This Class implement a singleton of a sycl device to keep
 * the same code structure than the cuda one.
 * 'cudaSetDevice' is global, a singleton allow us to call the
 * device globally too.
 */
class sycl_device
{
private:
    static sycl_device* _instance;

public:
    static sycl_device* get();
    Eigen::SyclDevice* device();
    void set_device(int dev);
    int get_device_idx();
    std::string get_device_name(int device);
    int get_num_devices();

private:
    sycl_device();
    sycl_device(const sycl_device &) = delete;
    sycl_device operator=(const sycl_device &) = delete;

private:
    int current_device_idx = 0;
    Eigen::SyclDevice current_device;
    Eigen::QueueInterface queue_interface;
};

#endif // SYCLDEVICE_H
