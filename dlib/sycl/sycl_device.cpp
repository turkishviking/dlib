#include "sycl_device.h"

sycl_device *sycl_device::_instance = nullptr;

sycl_device *sycl_device::get()
{
    std::lock_guard<std::mutex> lock(initialisationMutex);
    if (!_instance)
        _instance = new SyclDevice();
    return _instance;
}

sycl_device::sycl_device()
{
    set_device(0);
}

Eigen::SyclDevice* device()
{
    return &current_device;
}

void sycl_device::set_device (int dev)
{
    auto devices = Eigen::get_sycl_supported_devices();
    const auto device_selector = *devices[dev];
    queue_interface(device_selector);
    currentDeviceIdx = dev;
    current_device = Eigen::SyclDevice(&queue_interface);
}

int sycl_device::get_num_devices()
{
    return Eigen::get_sycl_supported_devices().size();
}

int sycl_device::get_device_idx()
{
    return current_device_idx;
}

std::string sycl_device::get_device_name(int device)
{
    return (cl::sycl::device)(Eigen::get_sycl_supported_devices()[device]).get_info<cl::sycl::info::device::name>();
}
