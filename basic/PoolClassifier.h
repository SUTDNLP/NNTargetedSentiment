/*
 * PoolClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_PoolClassifier_H_
#define SRC_PoolClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "Metric.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class PoolClassifier {
public:
  PoolClassifier() {
    _dropOut = 0.5;
  }
  ~PoolClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _token_representation_size;
  int _inputsize;
  int _hiddensize;

  UniLayer<xpu> _olayer_linear;
  //UniLayer<xpu> _tanh_project;
  int _poolmanners;
  int _poolfunctions;
  int _targetdim;
  int _leftdim;
  int _rightdim;

  int _poolsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

public:

  inline void init(const NRMat<dtype>& wordEmb, int labelSize) {
    _wordcontext = 0;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;
    _poolmanners = _poolfunctions * 3; //( left, right, target) * (avg, max, min, std, pro)
    _inputsize = _wordwindow * _token_representation_size;
    _hiddensize = _inputsize;

    _targetdim = _hiddensize;
    _leftdim = _hiddensize;
    _rightdim = _hiddensize;

    _poolsize = _poolmanners * _hiddensize;

    _words.initial(wordEmb);

    //_tanh_project.initial(_hiddensize, _inputsize, true, 3, 0);

    _olayer_linear.initial(_labelSize, _poolsize, false, 70, 2);

    _remove = 0;
  }

  inline void release() {
    _words.release();
    _olayer_linear.release();
    //_tanh_project.release();

  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;

      hash_set<int> targetIndex, leftIndex, rightIndex;
      Tensor<xpu, 2, dtype> targetrepresent, targetrepresentLoss;
      Tensor<xpu, 2, dtype> leftrepresent, leftrepresentLoss;
      Tensor<xpu, 2, dtype> rightrepresent, rightrepresentLoss;

      static hash_set<int>::iterator it;

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);
      wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);
      wordrepresentLoss = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

      project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
      projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolLoss[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
        poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
      }

      targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      targetrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      leftrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
      rightrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      //forward propagation
      //input setting, and linear setting
      int left = example.left, right = example.right;
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;
        if (idx < left) {
          leftIndex.insert(idx);
        } else if (idx > right) {
          rightIndex.insert(idx);
        } else {
          targetIndex.insert(idx);
        }

       _words.GetEmb(words[0], wordprime[idx]);

        //dropout
        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
      }

      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);

      // do we need a convolution? future work, currently needn't
      for (int idx = 0; idx < seq_size; idx++) {
        //_tanh_project.ComputeForwardScore(input[idx], project[idx]);
        project[idx] += input[idx];
      }

      offset = 0;
      //target
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], targetIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], targetIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], targetIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], targetIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], targetIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], targetrepresent);

      offset = _poolfunctions;
      //left
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], leftIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], leftIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], leftIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], leftIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], leftIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], leftrepresent);

      offset = 2 * _poolfunctions;
      //right
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(project, pool[offset], poolIndex[offset], rightIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], rightIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], rightIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], rightIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(project, pool[offset + 4], poolIndex[offset + 4], rightIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], rightrepresent);

      concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);

      _olayer_linear.ComputeForwardScore(poolmerge, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(poolmerge, output, outputLoss, poolmergeLoss);

      unconcat(targetrepresentLoss, leftrepresentLoss, rightrepresentLoss, poolmergeLoss);

      //overallrepresentLoss = 0.0; leftrepresentLoss = 0.0; rightrepresentLoss = 0.0; leftsentrepresentLoss = 0.0; rightsentrepresentLoss = 0.0;

      offset = 0;
      //target
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], targetrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  projectLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], projectLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], projectLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], projectLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], projectLoss);
      }

      offset = _poolfunctions;
      //left
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], leftrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  projectLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], projectLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], projectLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], projectLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], projectLoss);
      }

      offset = 2 * _poolfunctions;
      //right
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], rightrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  projectLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], projectLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], projectLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], projectLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], projectLoss);
      }

      for (int idx = 0; idx < seq_size; idx++) {
        //_tanh_project.ComputeBackwardLoss(input[idx], project[idx], projectLoss[idx], inputLoss[idx]);
        inputLoss[idx] += projectLoss[idx];
      }
	  
      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);

      for (int idx = 0; idx < seq_size; idx++) {
        wordprimeLoss[idx] += wordrepresentLoss[idx];
      }

      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }

      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordrepresent);
      FreeSpace(&wordrepresentLoss);

      FreeSpace(&input);
      FreeSpace(&inputLoss);

      FreeSpace(&project);
      FreeSpace(&projectLoss);

      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolLoss[idm]));
        FreeSpace(&(poolIndex[idm]));
      }

      FreeSpace(&targetrepresent);
      FreeSpace(&targetrepresentLoss);
      FreeSpace(&leftrepresent);
      FreeSpace(&leftrepresentLoss);
      FreeSpace(&rightrepresent);
      FreeSpace(&rightrepresentLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);

    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const vector<Feature>& features, const int& left, const int& right, vector<dtype>& results) {
    int seq_size = features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> output;

    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    hash_set<int> targetIndex, leftIndex, rightIndex;
    Tensor<xpu, 2, dtype> targetrepresent;
    Tensor<xpu, 2, dtype> leftrepresent;
    Tensor<xpu, 2, dtype> rightrepresent;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (idx < left) {
        leftIndex.insert(idx);
      } else if (idx > right) {
        rightIndex.insert(idx);
      } else {
        targetIndex.insert(idx);
      }

      _words.GetEmb(words[0], wordprime[idx]);

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      //_tanh_project.ComputeForwardScore(input[idx], project[idx]);
      project[idx] += input[idx];
    }

    offset = 0;
    //target
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], targetIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], targetIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], targetIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], targetIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], targetIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], targetrepresent);

    offset = _poolfunctions;
    //left
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], leftIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], leftIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], leftIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], leftIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], leftIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], leftrepresent);

    offset = 2 * _poolfunctions;
    //right
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], rightIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], rightIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], rightIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], rightIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], rightIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], rightrepresent);

    concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);

    _olayer_linear.ComputeForwardScore(poolmerge, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);
    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    FreeSpace(&targetrepresent);
    FreeSpace(&leftrepresent);
    FreeSpace(&rightrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&output);

    return optLabel;
  }

  void dumpfeatures(const vector<Feature>& features, const int& left, const int& right, NRVec<dtype>& poolfeatures) {
    int seq_size = features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;

    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    hash_set<int> targetIndex, leftIndex, rightIndex;
    Tensor<xpu, 2, dtype> targetrepresent;
    Tensor<xpu, 2, dtype> leftrepresent;
    Tensor<xpu, 2, dtype> rightrepresent;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (idx < left) {
        leftIndex.insert(idx);
      } else if (idx > right) {
        rightIndex.insert(idx);
      } else {
        targetIndex.insert(idx);
      }

      _words.GetEmb(words[0], wordprime[idx]);

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      //_tanh_project.ComputeForwardScore(input[idx], project[idx]);
      project[idx] += input[idx];
    }

    offset = 0;
    //target
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], targetIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], targetIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], targetIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], targetIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], targetIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], targetrepresent);

    offset = _poolfunctions;
    //left
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], leftIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], leftIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], leftIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], leftIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], leftIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], leftrepresent);

    offset = 2 * _poolfunctions;
    //right
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], rightIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], rightIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], rightIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], rightIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], rightIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], rightrepresent);

    concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);

    poolfeatures.resize(_poolsize);
    poolfeatures = 0.0;
    for (int i = 0; i < _poolsize; i++) {
      poolfeatures[i] = poolmerge[0][i];
    }

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);
    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    FreeSpace(&targetrepresent);
    FreeSpace(&leftrepresent);
    FreeSpace(&rightrepresent);

    FreeSpace(&poolmerge);
  }

  dtype computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> output;

    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

    hash_set<int> targetIndex, leftIndex, rightIndex;
    Tensor<xpu, 2, dtype> targetrepresent;
    Tensor<xpu, 2, dtype> leftrepresent;
    Tensor<xpu, 2, dtype> rightrepresent;

    static hash_set<int>::iterator it;

    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    int left = example.left, right = example.right;
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = example.m_features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
      if (idx < left) {
        leftIndex.insert(idx);
      } else if (idx > right) {
        rightIndex.insert(idx);
      } else {
        targetIndex.insert(idx);
      }

      _words.GetEmb(words[0], wordprime[idx]);

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      //_tanh_project.ComputeForwardScore(input[idx], project[idx]);
      project[idx] += input[idx];
    }

    offset = 0;
    //target
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], targetIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], targetIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], targetIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], targetIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], targetIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], targetrepresent);

    offset = _poolfunctions;
    //left
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], leftIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], leftIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], leftIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], leftIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], leftIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], leftrepresent);

    offset = 2 * _poolfunctions;
    //right
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(project, pool[offset], poolIndex[offset], rightIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(project, pool[offset + 1], poolIndex[offset + 1], rightIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(project, pool[offset + 2], poolIndex[offset + 2], rightIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(project, pool[offset + 3], poolIndex[offset + 3], rightIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(project, pool[offset + 4], poolIndex[offset + 4], rightIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], rightrepresent);

    concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);

    _olayer_linear.ComputeForwardScore(poolmerge, output);

    // get delta for each output
    dtype cost = softmax_cost(output, example.m_labels);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);
    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    FreeSpace(&targetrepresent);
    FreeSpace(&leftrepresent);
    FreeSpace(&rightrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    //_tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols, idThirds;
    idRows.clear();
    idCols.clear();
    idThirds.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int i = 0; i < Wd.size(1); i++)
      idCols.push_back(i);
    for (int i = 0; i < Wd.size(2); i++)
      idThirds.push_back(i);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());
    random_shuffle(idThirds.begin(), idThirds.end());

    int check_i = idRows[0], check_j = idCols[0], check_k = idThirds[0];

    dtype orginValue = Wd[check_i][check_j][check_k];

    Wd[check_i][check_j][check_k] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j][check_k] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j][check_k];

    printf("Iteration %d, Checking gradient for %s[%d][%d][%d]:\t", iter, mark.c_str(), check_i, check_j, check_k);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j][check_k] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    //checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    //checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);
    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }
};

#endif /* SRC_PoolClassifier_H_ */
