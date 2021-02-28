#include <iostream>
#include <torch/torch.h>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/*We can find the mnist file foramt at http://yann.lecun.com/exdb/mnist/
Ubyte file consists values like MagicNumber, NumItems, NumRows, NumCols, Data
Here, MagicNumber is unique to type like images or labels */
torch::Tensor read_mnist_images(const std::string& image_filename){
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);

    //2051 is magic number for images
    if(magic != 2051){
        std::cout<<"Incorrect image file magic: "<<magic<< std::endl;
        return torch::tensor(-1);
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    auto tensor = torch::empty({num_items, 1, rows, cols}, torch::kByte);
    image_file.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());

    return tensor.to(torch::kFloat32).div_(255);
}

torch::Tensor read_mnist_labels(const std::string& label_filename) {
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    uint32_t magic;
    uint32_t num_labels;

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);

    //2049 is magic number for images
    if(magic != 2049){
        std::cout<<"Incorrect image file magic: "<<magic<< std::endl;
        return torch::tensor(-1);
    }

    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);

    auto tensor = torch::empty(num_labels, torch::kByte);
    label_file.read(reinterpret_cast<char*>(tensor.data_ptr()), num_labels);
    return tensor.to(torch::kInt64);
}