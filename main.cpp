#include "ANN.h"
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>

const std::string RESET = "\033[0m";
const std::string GREEN = "\033[32m";

using namespace std;

Matrix intTobinary(int num) {
    vector<vector<double>> binary(4, vector<double>(1)); 
    for (int i = 3; i >= 0; i--) {
        binary[i][0] = num % 2;  
        num /= 2;
    }
    return Matrix(binary);
}

Matrix createTarget(int class_num, int num_classes) {
    vector<vector<double>> target(num_classes, vector<double>(1, 0.0));
    if (class_num >= 0 && class_num < num_classes) {
        target[class_num][0] = 1.0;
    }
    return Matrix(target);
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));
    int target_class = rand() % 16; 
    Matrix input_pattern = intTobinary(target_class);
    
    cout << "\nRandomly generated number: " << target_class << endl;
    cout << "Binary input: ";
    input_pattern.transpose().print_matrix(); 
    
    vector<int> layer_sizes = {4, 16, 16};
    ANN network(layer_sizes, nn_utils::softmax, nullptr);
    Matrix target_pattern = createTarget(target_class, 16);
    
    cout << "----------------------------------------------------------" << endl;
    
    int epoch = 0;
    int max_epochs = 10000; 
    vector<double> losses;

    while (epoch < max_epochs) {
        epoch++;
        double loss = network.train_step(input_pattern, target_pattern);
        losses.push_back(loss);
        double target_loss = 0.005;
        Matrix predictions = network.feedforward(input_pattern);
        int predicted = nn_utils::argmax(predictions);
        cout << "Epoch: " << setw(4) << epoch
             << " | Loss: " << fixed << setprecision(4) << loss
             << " | Prediction: " << setw(2) << predicted << endl;
        
        if (loss < target_loss) {
        cout << "----------------------------------------------------------" << endl;
        cout << GREEN << "SUCCESS! Reached target loss after " << epoch << " epochs." << RESET << endl;
        break;
        }
    }
    
    if (epoch == max_epochs) {cout << "Maximum epochs reached." << endl;}

    cout << "\n--- Probabilities ---" << endl;
    Matrix final_predictions = network.feedforward(input_pattern);
    int final_predicted = nn_utils::argmax(final_predictions);

    for (int i = 0; i < final_predictions.getRows(); i++) {
        cout << (i == final_predicted ? GREEN : RESET);
        cout << setw(2) << i << ": " << fixed << setprecision(4) << final_predictions.data[i][0] << endl;
    }

    ofstream loss_file("training_loss.csv");
    for (double loss : losses) {
        loss_file << loss << "\n";
    }
    loss_file.close();

    system("matlab -nosplash -nodesktop -r \""
            "try; " 
            "    losses = csvread('training_loss.csv'); "
            "    figure('Position', [100, 100, 1000, 600]); "
            "    plot(losses, 'b-', 'LineWidth', 1.5); "
            "    xlabel('Epoch'); "
            "    ylabel('Cross-Entropy Loss'); "
            "    title('Loss Curve'); "
            "    grid on; " 
            "    waitfor(gcf); " 
            "catch ME; " 
            "    fprintf('Error: %s\\n', ME.message); " 
            "    pause(5); " 
            "end; "
            "\"");
    return 0;
}