#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>

//creating an alias so that the code is short.
namespace F = torch::nn::functional;

// Where to find the CIFAR10 dataset.
const std::string kDataRoot = "../cifar-10-batches-bin";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 5;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

const uint32_t kImageRows = 32;
const uint32_t kImageColumns = 32;
const uint32_t kImageChannels = 3;
const uint32_t kNumTrainBatchFiles = 5;
const std::string kTrainBatchFileNames[] = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
const uint32_t kTrainSizePerBatchFile = 10000;
const uint32_t kTrainSize = 50000;
const uint32_t kTestSize = 10000;
const std::string kTestFilename = "test_batch.bin";

/// The CIFAR10 dataset.
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
    /// The mode in which the dataset is loaded.
    enum class Mode { kTrain, kTest };
    /// Loads the CIFAR dataset from the `root` path.
    ///
    /// The supplied `root` path should contain the *content* of the unzipped
    /// CIFAR binary version dataset, available from https://wwww.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
    explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain)	{
        char label;
        torch::Tensor image = torch::empty({ kImageChannels, kImageRows, kImageColumns }, torch::kByte);
        if (Mode::kTrain == mode) {
            targets_ = torch::empty(kTrainSize, torch::kByte);
            images_ = torch::empty({ kTrainSize, kImageChannels, kImageRows, kImageColumns }, torch::kByte);

            size_t sample_index = 0;

            for (size_t batch = 0; batch < kNumTrainBatchFiles; ++batch) {
                std::string path = join_paths(root, kTrainBatchFileNames[batch]);
                std::ifstream fid(path, std::ios::binary);
                TORCH_CHECK(fid, "Error opening images file at ", path);

                for (size_t img_index = 0; img_index < kTrainSizePerBatchFile; ++img_index) {
                    fid.read(reinterpret_cast<char*>(&label), sizeof(label));
                    targets_[sample_index] = label;
                    fid.read(reinterpret_cast<char*>(image.data_ptr()), image.numel());
                    images_[sample_index] = image.clone();
                    sample_index = sample_index + 1;
                }
                fid.close();
            }
        }
        else {
            targets_ = torch::empty(kTestSize, torch::kByte);
            images_ = torch::empty({ kTestSize, kImageChannels, kImageRows, kImageColumns }, torch::kByte);

            size_t sample_index = 0;

            std::string path = join_paths(root, kTestFilename);
            std::ifstream fid(path, std::ios::binary);
            TORCH_CHECK(fid, "Error opening images file at ", path);

            for (size_t img_index = 0; img_index < kTestSize; ++img_index) {

                fid.read(reinterpret_cast<char*>(&label), sizeof(label));
                targets_[sample_index] = label;

                fid.read(reinterpret_cast<char*>(image.data_ptr()), image.numel());
                images_[sample_index] = image.clone();
                sample_index = sample_index + 1;
            }
            fid.close();
        }
        images_ = images_.to(torch::kFloat32).div_(255);
        targets_ = targets_.to(torch::kInt64);
    }

    /// Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) {
        return { images_[index], targets_[index] };
    }

    /// Returns the size of the dataset.
    torch::optional<size_t> size() const {
        return images_.size(0);
    }

    /// Returns true if this is the training subset of MNIST.
    bool is_train() const noexcept {
        return images_.size(0) == kTrainSize;
    }

    /// Returns all images stacked into a single tensor.
    const torch::Tensor& images() const {
        return images_;
    }

    /// Returns all targets stacked into a single tensor.
    const torch::Tensor& targets() const {
        return targets_;
    }

private:
    torch::Tensor images_, targets_;

    std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }
};

//////////// Specify the architecture.

struct Net : torch::nn::Module
{
    Net()
    {
        // https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
        //torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(p).stride(s) and similary other options
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)));
        dp1 = register_module("dp1", torch::nn::Dropout(0.25)); // 0.25
        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        dp2 = register_module("dp2", torch::nn::Dropout(0.30));
        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
        dp3 = register_module("dp3", torch::nn::Dropout(0.30));
        fc1 = register_module("fc1", torch::nn::Linear(2 * 2 * 64, 512));
        dp4 = register_module("dp4", torch::nn::Dropout(0.4));
        fc2 = register_module("fc2", torch::nn::Linear(512, 10));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::elu(conv1_1->forward(x));
        x = torch::elu(conv1_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp1(x);

        x = torch::elu(conv2_1->forward(x));
        x = torch::elu(conv2_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp2(x);

        x = torch::elu(conv3_1->forward(x));
        x = torch::elu(conv3_2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = dp3(x);

        x = x.view({-1, 2 * 2 * 64});

        x = torch::elu(fc1->forward(x));
        x = dp4(x);
        x = torch::log_softmax(fc2->forward(x), 1);

        return x;
    }

    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Dropout dp1{nullptr};
    torch::nn::Dropout dp2{nullptr};
    torch::nn::Dropout dp3{nullptr};
    torch::nn::Dropout dp4{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

////////////////////////////////////////////////////////////
template <typename DataLoader>
void train(int32_t epoch, Net& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size)
{
    model.train();
    double train_loss = 0;
    int32_t correct = 0;
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = F::nll_loss(output, targets);


        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % kLogInterval == 0) {
            std::printf(
                    "\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
                    epoch,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
        }
        train_loss += loss.template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }
    train_loss /= dataset_size;
    std::printf(
            "\n   Train set: Average loss: %.4f | Accuracy: %.3f",
            train_loss,
            static_cast<double>(correct) / dataset_size);
}

template <typename DataLoader>
void test(Net& model, torch::Device device, DataLoader& data_loader, size_t dataset_size)
{
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model.forward(data);
        test_loss += F::nll_loss(
                output,
                targets,
                F::NLLLossFuncOptions().ignore_index(-100).reduction(torch::kSum))
                .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::printf(
            "\n    Test set: Average loss: %.4f | Accuracy: %.3f\n",
            test_loss,
            static_cast<double>(correct) / dataset_size);
}

int main() {
    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model;
    model.to(device);

    auto train_dataset = CIFAR10(kDataRoot, CIFAR10::Mode::kTrain).map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =	torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

    auto test_dataset = CIFAR10(kDataRoot, CIFAR10::Mode::kTest).map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);


    // You must specify the kind of optimizer in the following code

    //torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(/*lr=*/0.001).momentum(0.9));
    //torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.999)));

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.6, 0.999)));
    //torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.6, 0.999))); -> 0.533
    // torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.4, 0.999))); -> result 0.518
    ////////////////////////////////////////////////////////////////

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *test_loader, test_dataset_size);
    }

    return 0;
}

void disp(const torch::Tensor &t)
{
    std::cout << t << std::endl;
}
void size(const torch::Tensor &t)
{
    std::cout << t.sizes() << std::endl;
}