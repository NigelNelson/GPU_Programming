/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

float cpu_time(timespec* start, timespec* end){
	/**
	 * Function responsible for returning the ellapsed time in
	 * milliseconds
	 */
	return ((1e9*end->tv_sec + end->tv_nsec) - (1e9*start->tv_sec + 
	start->tv_nsec))/1e6;
}


int main(void) {
  //data
  MNIST dataset("../data/mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn;
  Layer* conv1 = new Conv(1, 28, 28, 4, 5, 5, 2, 2, 2);
  Layer* pool1 = new MaxPooling(4, 14, 14, 2, 2, 2);
  Layer* conv2 = new Conv(4, 7, 7, 16, 5, 5, 1, 2, 2);
  Layer* pool2 = new MaxPooling(16, 7, 7, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 32);
  Layer* fc4 = new FullyConnected(32, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* softmax = new Softmax;
  dnn.add_layer(conv1);
  dnn.add_layer(relu1);
  dnn.add_layer(pool1);
  dnn.add_layer(conv2);
  dnn.add_layer(relu2);
  dnn.add_layer(pool2);
  dnn.add_layer(fc3);
  dnn.add_layer(relu3);
  dnn.add_layer(fc4);
  dnn.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);

  	// Begin timing the CPU implementation
	timespec ts, te;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

  dnn.forward(dataset.test_data);

	clock_gettime(CLOCK_MONOTONIC_RAW, &te);
	printf("CPU ellapsed time: %f\n", cpu_time(&ts, &te));


  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << std::endl;
  std::cout << "random init test acc: " << acc << std::endl;
  std::cout << std::endl;

  return 0;
}

