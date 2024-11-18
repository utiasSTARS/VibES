#include <torch/script.h> // One-stop header for TorchScript
#include <iostream>

int main() {
    // Load the scripted model
    torch::jit::Module model;
    try {
        model = torch::jit::load("/home/viciopoli/STARS/courses/CSC2529 computational imagin/Project_proposal/project/resnet1d_scripted.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n"<< e.what() << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully\n";

    // Prepare example input
    std::vector<float> input_data(100 * 3, 0.0); // Example input
    auto input_tensor = torch::from_blob(input_data.data(), {1, 100, 3}).to(torch::kFloat32).to(torch::kCUDA);

    // Perform inference
    at::Tensor output = model.forward({input_tensor}).toTensor();
    std::cout << "Model output: " << output << "\n";

    return 0;
}
