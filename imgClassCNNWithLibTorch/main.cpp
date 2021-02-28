#include <iostream>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "read-mnist.h"

struct Options {
    int batchSize = 256; //Batch size
    size_t epochs = 20; // Number of epochs
    size_t logInterval = 20;
    std::ofstream loss_acc_train;
    std::ofstream loss_acc_test;
    //Paths to train and test images and labels
    const char* trainImagesPath = "train-images-idx3-ubyte";
    const char* trainLabelsPath = "train-labels-idx1-ubyte";
    const char* testImagesPath = "t10k-images-idx3-ubyte";
    const char* testLabelsPath = "t10k-labels-idx1-ubyte";
    torch::DeviceType device = torch::kCPU;
};

static Options options;

//Read images from ubyte format and convert to tensors
torch::Tensor process_images(const std::string& root, bool train) {
    const auto path = root + (train ? options.trainImagesPath: options.testImagesPath); //images_path
    auto images = read_mnist_images(path);

    return images;
}

//Read labels from ubyte format and convert to tensors
torch::Tensor process_labels(const std::string& root, bool train) {
    const auto path = root + (train ? options.trainLabelsPath: options.testLabelsPath); //labels_path
    auto labels = read_mnist_labels(path);

    return labels;
}


//Use CustomDataset class to load any type of dataset other than inbuilt datasets
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
private:
    // data should be 2 tensors
    torch::Tensor images, labels;
    size_t img_size;
public:
    CustomDataset(const std::string& root, bool train) {
        images = process_images(root, train);
        labels = process_labels(root, train);
        img_size = images.size(0);
    }

    //Returns the data sample at the given `index
    torch::data::Example<> get(size_t index) override {
        // This should return {torch::Tensor, torch::Tensor}
        torch::Tensor img = images[index];
        torch::Tensor label = labels[index];
        return {img.clone(), label.clone()};
    };

    torch::optional<size_t> size() const override {
        return img_size;
    };
};


struct Net: torch::nn::Module {
    Net() {
        //Convolutional layer 1: 32 filters, 3x3 Kernel size, padding=1
        //torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(p).stride(s) and similarly other options
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
        //Convolutional layer 2: 32 filters, 3x3 Kernel size, padding=0
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)));
        // Insert pool layer (poolsize-[2,2])
        dp1 = register_module("dp1", torch::nn::Dropout(0.25));

        //Convolutional layer 3: 64 filters, 3x3 Kernel size, padding=1
        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        //Convolutional layer 4: 64 filters, 3x3 Kernel size, padding=0
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        // Insert pool layer (poolsize-[2,2])
        dp2 = register_module("dp2", torch::nn::Dropout(0.25));

        //Convolutional layer 5: 64 filters, 3x3 Kernel size, padding=1
        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        //Convolutional layer 6: 32 filters, 3x3 Kernel size, padding=0
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        // Insert pool layer (poolsize-[2,2])
        dp3 = register_module("dp3", torch::nn::Dropout(0.25));

        //Fully connected layer - torch::nn:Linear(input_size, output_size(number of neurons))
        fc1 = register_module("fc1", torch::nn::Linear(64, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, 10));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1_1->forward(x));
        x = torch::relu(conv1_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp1(x);

        x = torch::relu(conv2_1->forward(x));
        x = torch::relu(conv2_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp2(x);

        x = torch::relu(conv3_1->forward(x));
        x = torch::relu(conv3_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp3(x);

        x = x.view({-1, 64});

        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);

        return torch::log_softmax(x, 1);
    }

    // Declare layers
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Dropout dp1{nullptr}, dp2{nullptr}, dp3{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

template <typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size) {
    size_t index = 0;
    //Set network in the training mode
    network->train();
    float Loss = 0, Acc = 0;

    for (auto& batch : loader) {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).view({-1});
        // Execute the model on the input data
        auto output = network->forward(data);

        //Using mean square error loss function to compute loss
        auto loss = torch::nll_loss(output, targets);
        auto acc = output.argmax(1).eq(targets).sum();

        // Reset gradients
        optimizer.zero_grad();
        // Compute gradients
        loss.backward();
        //Update the parameters
        optimizer.step();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    if (index++ % options.logInterval == 0) {
        auto end = data_size;
        options.loss_acc_train << std::to_string(Loss/data_size) + "," + std::to_string(Acc/data_size) << std::endl;

        std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                  << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                  << std::endl;
    }
};


template <typename DataLoader>
void test(std::shared_ptr<Net> network, DataLoader& loader, size_t epoch, size_t data_size) {
    network->eval();
    size_t index = 0;
    float Loss = 0, Acc = 0;
    int display_count = 0;

    for (const auto& batch : loader) {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).view({-1});

        auto output = network->forward(data);

        //To display 3 test image and its output
        if (display_count < 3 && epoch== options.epochs) {
            cv::Mat test_image(28,28,CV_8UC1);
            torch::Tensor tensor = data[display_count].mul_(255).clamp(0,255).to(torch::kU8);
            tensor = tensor.to(torch::kCPU);
            std::memcpy((void*)test_image.data,tensor.data_ptr(),sizeof(torch::kU8)*tensor.numel());

            std::cout << "***** TESTING on TEST IMAGE " << display_count << " *****" << std::endl;
            std::cout << "GroundTruth: " << targets[display_count].template item<float>()
                      << ", Prediction: " << output[display_count].argmax() << std::endl;
            std::cout << "Output Probabilities" << std::endl;
            for (int i =0; i < output[display_count].size(0); i++) {
                std::cout << "Class: " << i << " " << torch::exp(output[display_count])[i].template item<float>()  << std::endl;
            }
            cv::imwrite("OUTPUT" + std::to_string(display_count) + "_GT_" + std::to_string(targets[display_count].template item<int>()) +
                        "_Pred_" + std::to_string(output[display_count].argmax().template item<int>()) + ".jpg", test_image);
            std::cout << "Outputs saved, Please checkout the output images" << std::endl;

            display_count++;
        }

        auto loss = torch::nll_loss(output, targets);
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    //This block can be used inside for loop to see the training within each epoch
    if (index++ % options.logInterval == 0) {
        options.loss_acc_test << std::to_string(Loss/data_size) + "," + std::to_string(Acc/data_size) << std::endl;
        std::cout << "Val Epoch: " << epoch
                  << "\tVal Loss: " << Loss/data_size << "\tVal ACC:"<< Acc/data_size << std::endl;
    }
}


int main() {
    //Use CUDA for computation if available
    if (torch::cuda::is_available())
        options.device = torch::kCUDA;
    std::cout << "Running on: "
              << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    //Path to Fashion Mnist
    std::string root_string = "../fashion-mnist/";
    bool isTrain = true; //Flag to create train or test data

    //Uses Custom Dataset Class to load train data. Apply stack collation which takes
    //batch of tensors and stacks them into single tensor along the first dimension
    auto train_dataset = CustomDataset(root_string, isTrain).map(torch::data::transforms::Stack<>());
    //Data Loader provides options to speed up the data loading like batch size, number of workers
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), options.batchSize);
    auto train_size = train_dataset.size().value();

    //Process and load test dat similar to above
    auto test_dataset = CustomDataset(root_string, false).map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(test_dataset), options.batchSize);
    auto test_size = test_dataset.size().value();

    //Create Feed forward network
    auto net = std::make_shared<Net>();
    //Moving model parameters to correct device
    net->to(options.device);
    // torch::load(net, "net.pt"); /*To use trained model*/

    options.loss_acc_train.open("loss_acc_train.txt");
    options.loss_acc_test.open("loss_acc_test.txt");

    //Using Adam optimizer with beta1 as 0.5
    //torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.999)));


    for (size_t i = 0; i < options.epochs; i++) {
        /*Run the training for all epochs*/
        train(net, *train_loader, optimizer, i + 1, train_size);
        std::cout << std::endl;
        /*Run on the validation set for all epochs*/
        test(net, *test_loader, i+1, test_size);
        /*Save the network*/
        torch::save(net, "net.pt");
    }

    return 0;
}
