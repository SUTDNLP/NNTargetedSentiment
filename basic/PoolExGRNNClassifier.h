/*
 * PoolExGRNNClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_PoolExGRNNClassifier_H_
#define SRC_PoolExGRNNClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class PoolExGRNNClassifier {
public:
  PoolExGRNNClassifier() {
    _dropOut = 0.5;
  }
  ~PoolExGRNNClassifier() {

  }

public:
  LookupTable<xpu> _words;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _token_representation_size;
  int _inputsize;
  int _hiddensize;
  int _rnnHiddenSize;

  //gated interaction part
  UniLayer<xpu> _represent_transform[3];
  //overall
  AttRecursiveGatedNN<xpu> _target_attention;

  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;

  GRNN<xpu> _rnn_left;
  GRNN<xpu> _rnn_right;

  int _poolmanners;
  int _poolfunctions;
  int _targetdim;
  int _leftdim;
  int _rightdim;

  int _poolsize;
  int _gatedsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

public:

  inline void init(const NRMat<dtype>& wordEmb, int wordcontext, int rnnHiddenSize, int hiddensize, int labelSize) {
    _wordcontext = wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = labelSize;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;
    _poolmanners = _poolfunctions * 3; //( left, right, target) * (avg, max, min, std, pro)
    _inputsize = _wordwindow * _token_representation_size;
    _hiddensize = hiddensize;
    _rnnHiddenSize = rnnHiddenSize;

    _targetdim = _hiddensize;
    _leftdim = _hiddensize;
    _rightdim = _hiddensize;

    _poolsize = _poolmanners * _hiddensize;
    _gatedsize = _targetdim;

    _words.initial(wordEmb);

    for (int idx = 0; idx < 3; idx++) {
      _represent_transform[idx].initial(_targetdim, _poolfunctions * _hiddensize, true, (idx + 1) * 100 + 60, 0);
    }

    _target_attention.initial(_targetdim, _targetdim, 100);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);

    _tanh_project.initial(_hiddensize, 2 * _rnnHiddenSize, true, 70, 0);
    _olayer_linear.initial(_labelSize, _poolsize + _gatedsize, false, 80, 2);

    _remove = 0;
  }

  inline void release() {
    _words.release();
    _olayer_linear.release();
    _tanh_project.release();
    _rnn_left.release();
    _rnn_right.release();

    for (int idx = 0; idx < 3; idx++) {
      _represent_transform[idx].release();
    }

    _target_attention.release();

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

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> gatedmerge, gatedmergeLoss;
      Tensor<xpu, 2, dtype> allmerge, allmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //gated interaction part
      Tensor<xpu, 2, dtype> target_input_span[3], target_input_spanLoss[3];
      Tensor<xpu, 2, dtype> target_reset_left, target_reset_right, target_interact_middle;
      Tensor<xpu, 2, dtype> target_update_left, target_update_right, target_update_interact;
      Tensor<xpu, 2, dtype> target_interact, target_interactLoss;

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

      rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);
      rnn_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

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

      for (int idm = 0; idm < 3; idm++) {
        target_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
        target_input_spanLoss[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }

      target_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      target_interactLoss = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLoss = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize), 0.0);
      allmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize), 0.0);
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

      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

      for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);
      }

      // do we need a convolution? future work, currently needn't
      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);
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

      _represent_transform[0].ComputeForwardScore(leftrepresent, target_input_span[0]);
      _represent_transform[1].ComputeForwardScore(rightrepresent, target_input_span[1]);
      _represent_transform[2].ComputeForwardScore(targetrepresent, target_input_span[2]);

      _target_attention.ComputeForwardScore(target_input_span[0], target_input_span[1], target_input_span[2],
          target_reset_left, target_reset_right, target_interact_middle, 
          target_update_left, target_update_right, target_update_interact, 
          target_interact);


      concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);
      gatedmerge += target_interact;
      concat(poolmerge, gatedmerge, allmerge);

      _olayer_linear.ComputeForwardScore(allmerge, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(allmerge, output, outputLoss, allmergeLoss);

      unconcat(poolmergeLoss, gatedmergeLoss, allmergeLoss);
      target_interactLoss += gatedmergeLoss;
      unconcat(targetrepresentLoss, leftrepresentLoss, rightrepresentLoss, poolmergeLoss);

      _target_attention.ComputeBackwardLoss(target_input_span[0], target_input_span[1], target_input_span[2],
          target_reset_left, target_reset_right, target_interact_middle, 
          target_update_left, target_update_right, target_update_interact, 
          target_interact, target_interactLoss,
          target_input_spanLoss[0], target_input_spanLoss[1], target_input_spanLoss[2]);

      _represent_transform[0].ComputeBackwardLoss(leftrepresent, target_input_span[0], target_input_spanLoss[0], leftrepresentLoss);
      _represent_transform[1].ComputeBackwardLoss(rightrepresent, target_input_span[1], target_input_spanLoss[1], rightrepresentLoss);
      _represent_transform[2].ComputeBackwardLoss(targetrepresent, target_input_span[2], target_input_spanLoss[2], targetrepresentLoss);

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
        _tanh_project.ComputeBackwardLoss(rnn_hidden_merge[idx], project[idx], projectLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(rnn_hidden_leftLoss[idx], rnn_hidden_rightLoss[idx], rnn_hidden_mergeLoss[idx]);
      }

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);
	  
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

      FreeSpace(&rnn_hidden_left_reset);
      FreeSpace(&rnn_hidden_left_update);
      FreeSpace(&rnn_hidden_left_afterreset);
      FreeSpace(&rnn_hidden_left_current);
      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_leftLoss);

      FreeSpace(&rnn_hidden_right_reset);
      FreeSpace(&rnn_hidden_right_update);
      FreeSpace(&rnn_hidden_right_afterreset);
      FreeSpace(&rnn_hidden_right_current);
      FreeSpace(&rnn_hidden_right);
      FreeSpace(&rnn_hidden_rightLoss);

      FreeSpace(&rnn_hidden_merge);
      FreeSpace(&rnn_hidden_mergeLoss);

      FreeSpace(&project);
      FreeSpace(&projectLoss);

      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolLoss[idm]));
        FreeSpace(&(poolIndex[idm]));
      }

      for (int idm = 0; idm < 3; idm++) {
        FreeSpace(&(target_input_span[idm]));
        FreeSpace(&(target_input_spanLoss[idm]));
      }

      FreeSpace(&target_reset_left);
      FreeSpace(&target_reset_right);
      FreeSpace(&target_interact_middle);
      FreeSpace(&target_update_left);
      FreeSpace(&target_update_right);
      FreeSpace(&target_update_interact);
      FreeSpace(&(target_interact));
      FreeSpace(&(target_interactLoss));

      FreeSpace(&targetrepresent);
      FreeSpace(&targetrepresentLoss);
      FreeSpace(&leftrepresent);
      FreeSpace(&leftrepresentLoss);
      FreeSpace(&rightrepresent);
      FreeSpace(&rightrepresentLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&gatedmerge);
      FreeSpace(&gatedmergeLoss);
      FreeSpace(&allmerge);
      FreeSpace(&allmergeLoss);
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

    Tensor<xpu, 3, dtype> rnn_hidden_left_update;
    Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_left_current;

    Tensor<xpu, 3, dtype> rnn_hidden_right_update;
    Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_right;
    Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_right_current;

    Tensor<xpu, 3, dtype> rnn_hidden_merge;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> gatedmerge;
    Tensor<xpu, 2, dtype> allmerge;
    Tensor<xpu, 2, dtype> output;

    //gated interaction part
    Tensor<xpu, 2, dtype> target_input_span[3];
    Tensor<xpu, 2, dtype> target_reset_left, target_reset_right, target_interact_middle;
    Tensor<xpu, 2, dtype> target_update_left, target_update_right, target_update_interact;
    Tensor<xpu, 2, dtype> target_interact;

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

    rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

    for (int idm = 0; idm < 3; idm++) {
      target_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }
    
    target_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize), 0.0);
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

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);
    }

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);
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

    _represent_transform[0].ComputeForwardScore(leftrepresent, target_input_span[0]);
    _represent_transform[1].ComputeForwardScore(rightrepresent, target_input_span[1]);
    _represent_transform[2].ComputeForwardScore(targetrepresent, target_input_span[2]);

    _target_attention.ComputeForwardScore(target_input_span[0], target_input_span[1], target_input_span[2],
        target_reset_left, target_reset_right, target_interact_middle, 
        target_update_left, target_update_right, target_update_interact, 
        target_interact);


    concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);
    gatedmerge += target_interact;
    concat(poolmerge, gatedmerge, allmerge);


    _olayer_linear.ComputeForwardScore(allmerge, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);

    FreeSpace(&rnn_hidden_left_reset);
    FreeSpace(&rnn_hidden_left_update);
    FreeSpace(&rnn_hidden_left_afterreset);
    FreeSpace(&rnn_hidden_left_current);
    FreeSpace(&rnn_hidden_left);

    FreeSpace(&rnn_hidden_right_reset);
    FreeSpace(&rnn_hidden_right_update);
    FreeSpace(&rnn_hidden_right_afterreset);
    FreeSpace(&rnn_hidden_right_current);
    FreeSpace(&rnn_hidden_right);

    FreeSpace(&rnn_hidden_merge);

    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    for (int idm = 0; idm < 3; idm++) {
      FreeSpace(&(target_input_span[idm]));
    }
    
    FreeSpace(&target_reset_left);
    FreeSpace(&target_reset_right);
    FreeSpace(&target_interact_middle);
    FreeSpace(&target_update_left);
    FreeSpace(&target_update_right);
    FreeSpace(&target_update_interact);
    FreeSpace(&(target_interact));

    FreeSpace(&targetrepresent);
    FreeSpace(&leftrepresent);
    FreeSpace(&rightrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&gatedmerge);
    FreeSpace(&allmerge);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
    int seq_size = example.m_features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    Tensor<xpu, 3, dtype> rnn_hidden_left_update;
    Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_left_current;

    Tensor<xpu, 3, dtype> rnn_hidden_right_update;
    Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_right;
    Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_right_current;

    Tensor<xpu, 3, dtype> rnn_hidden_merge;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> gatedmerge;
    Tensor<xpu, 2, dtype> allmerge;
    Tensor<xpu, 2, dtype> output;

    //gated interaction part
    Tensor<xpu, 2, dtype> target_input_span[3];
    Tensor<xpu, 2, dtype> target_reset_left, target_reset_right, target_interact_middle;
    Tensor<xpu, 2, dtype> target_update_left, target_update_right, target_update_interact;
    Tensor<xpu, 2, dtype> target_interact;

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

    rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2 * _rnnHiddenSize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _hiddensize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
    }

    targetrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    leftrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);
    rightrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _hiddensize), 0.0);

    for (int idm = 0; idm < 3; idm++) {
      target_input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }
    
    target_reset_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_reset_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_interact_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_left = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_right = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_update_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    target_interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);

    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize), 0.0);
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

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

    for (int idx = 0; idx < seq_size; idx++) {
      concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);
    }

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);
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

    _represent_transform[0].ComputeForwardScore(leftrepresent, target_input_span[0]);
    _represent_transform[1].ComputeForwardScore(rightrepresent, target_input_span[1]);
    _represent_transform[2].ComputeForwardScore(targetrepresent, target_input_span[2]);

    _target_attention.ComputeForwardScore(target_input_span[0], target_input_span[1], target_input_span[2],
        target_reset_left, target_reset_right, target_interact_middle, 
        target_update_left, target_update_right, target_update_interact, 
        target_interact);


    concat(targetrepresent, leftrepresent, rightrepresent, poolmerge);
    gatedmerge += target_interact;
    concat(poolmerge, gatedmerge, allmerge);


    _olayer_linear.ComputeForwardScore(allmerge, output);

    // get delta for each output
    dtype cost = softmax_cost(output, example.m_labels);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);

    FreeSpace(&rnn_hidden_left_reset);
    FreeSpace(&rnn_hidden_left_update);
    FreeSpace(&rnn_hidden_left_afterreset);
    FreeSpace(&rnn_hidden_left_current);
    FreeSpace(&rnn_hidden_left);

    FreeSpace(&rnn_hidden_right_reset);
    FreeSpace(&rnn_hidden_right_update);
    FreeSpace(&rnn_hidden_right_afterreset);
    FreeSpace(&rnn_hidden_right_current);
    FreeSpace(&rnn_hidden_right);

    FreeSpace(&rnn_hidden_merge);

    FreeSpace(&project);

    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }

    for (int idm = 0; idm < 3; idm++) {
      FreeSpace(&(target_input_span[idm]));
    }
    
    FreeSpace(&target_reset_left);
    FreeSpace(&target_reset_right);
    FreeSpace(&target_interact_middle);
    FreeSpace(&target_update_left);
    FreeSpace(&target_update_right);
    FreeSpace(&target_update_interact);
    FreeSpace(&(target_interact));

    FreeSpace(&targetrepresent);
    FreeSpace(&leftrepresent);
    FreeSpace(&rightrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&gatedmerge);
    FreeSpace(&allmerge);
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _target_attention.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    for (int idx = 0; idx < 3; idx++) {
      _represent_transform[idx].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }

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

    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

    checkgrad(examples, _rnn_left._rnn_update._WL, _rnn_left._rnn_update._gradWL, "_rnn_left._rnn_update._WL", iter);
    checkgrad(examples, _rnn_left._rnn_update._WR, _rnn_left._rnn_update._gradWR, "_rnn_left._rnn_update._WR", iter);
    checkgrad(examples, _rnn_left._rnn_update._b, _rnn_left._rnn_update._gradb, "_rnn_left._rnn_update._b", iter);
    checkgrad(examples, _rnn_left._rnn_reset._WL, _rnn_left._rnn_reset._gradWL, "_rnn_left._rnn_reset._WL", iter);
    checkgrad(examples, _rnn_left._rnn_reset._WR, _rnn_left._rnn_reset._gradWR, "_rnn_left._rnn_reset._WR", iter);
    checkgrad(examples, _rnn_left._rnn_reset._b, _rnn_left._rnn_reset._gradb, "_rnn_left._rnn_reset._b", iter);
    checkgrad(examples, _rnn_left._rnn._WL, _rnn_left._rnn._gradWL, "_rnn_left._rnn._WL", iter);
    checkgrad(examples, _rnn_left._rnn._WR, _rnn_left._rnn._gradWR, "_rnn_left._rnn._WR", iter);
    checkgrad(examples, _rnn_left._rnn._b, _rnn_left._rnn._gradb, "_rnn_left._rnn._b", iter);

    checkgrad(examples, _rnn_right._rnn_update._WL, _rnn_right._rnn_update._gradWL, "_rnn_right._rnn_update._WL", iter);
    checkgrad(examples, _rnn_right._rnn_update._WR, _rnn_right._rnn_update._gradWR, "_rnn_right._rnn_update._WR", iter);
    checkgrad(examples, _rnn_right._rnn_update._b, _rnn_right._rnn_update._gradb, "_rnn_right._rnn_update._b", iter);
    checkgrad(examples, _rnn_right._rnn_reset._WL, _rnn_right._rnn_reset._gradWL, "_rnn_right._rnn_reset._WL", iter);
    checkgrad(examples, _rnn_right._rnn_reset._WR, _rnn_right._rnn_reset._gradWR, "_rnn_right._rnn_reset._WR", iter);
    checkgrad(examples, _rnn_right._rnn_reset._b, _rnn_right._rnn_reset._gradb, "_rnn_right._rnn_reset._b", iter);
    checkgrad(examples, _rnn_right._rnn._WL, _rnn_right._rnn._gradWL, "_rnn_right._rnn._WL", iter);
    checkgrad(examples, _rnn_right._rnn._WR, _rnn_right._rnn._gradWR, "_rnn_right._rnn._WR", iter);
    checkgrad(examples, _rnn_right._rnn._b, _rnn_right._rnn._gradb, "_rnn_right._rnn._b", iter);

    checkgrad(examples, _target_attention._reset_left._WL, _target_attention._reset_left._gradWL, "_target_attention._reset_left._WL", iter);
    checkgrad(examples, _target_attention._reset_left._WR, _target_attention._reset_left._gradWR, "_target_attention._reset_left._WR", iter);
    checkgrad(examples, _target_attention._reset_left._b, _target_attention._reset_left._gradb, "_target_attention._reset_left._b", iter);

    checkgrad(examples, _target_attention. _reset_right._WL, _target_attention. _reset_right._gradWL, "_target_attention. _reset_right._WL", iter);
    checkgrad(examples, _target_attention. _reset_right._WR, _target_attention. _reset_right._gradWR, "_target_attention. _reset_right._WR", iter);
    checkgrad(examples, _target_attention. _reset_right._b, _target_attention. _reset_right._gradb, "_target_attention. _reset_right._b", iter);

    checkgrad(examples, _target_attention._update_left._WL, _target_attention._update_left._gradWL, "_target_attention._update_left._WL", iter);
    checkgrad(examples, _target_attention._update_left._WR, _target_attention._update_left._gradWR, "_target_attention._update_left._WR", iter);
    checkgrad(examples, _target_attention._update_left._b, _target_attention._update_left._gradb, "_target_attention._update_left._b", iter);

    checkgrad(examples, _target_attention._update_right._WL, _target_attention._update_right._gradWL, "_target_attention._update_right._WL", iter);
    checkgrad(examples, _target_attention._update_right._WR, _target_attention._update_right._gradWR, "_target_attention._update_right._WR", iter);
    checkgrad(examples, _target_attention._update_right._b, _target_attention._update_right._gradb, "_target_attention._update_right._b", iter);

    checkgrad(examples, _target_attention._update_tilde._WL, _target_attention._update_tilde._gradWL, "_target_attention._update_tilde._WL", iter);
    checkgrad(examples, _target_attention._update_tilde._WR, _target_attention._update_tilde._gradWR, "_target_attention._update_tilde._WR", iter);
    checkgrad(examples, _target_attention._update_tilde._b, _target_attention._update_tilde._gradb, "_target_attention._update_tilde._b", iter);

    checkgrad(examples, _target_attention._recursive_tilde._WL, _target_attention._recursive_tilde._gradWL, "_target_attention._recursive_tilde._WL", iter);
    checkgrad(examples, _target_attention._recursive_tilde._WR, _target_attention._recursive_tilde._gradWR, "_target_attention._recursive_tilde._WR", iter);
    checkgrad(examples, _target_attention._recursive_tilde._b, _target_attention._recursive_tilde._gradb, "_target_attention._recursive_tilde._b", iter);

    for (int idx = 0; idx < 3; idx++) {
      stringstream ssposition;
      ssposition << "[" << idx << "]";

      checkgrad(examples, _represent_transform[idx]._W, _represent_transform[idx]._gradW, "_represent_transform" + ssposition.str() + "._W", iter);
      checkgrad(examples, _represent_transform[idx]._b, _represent_transform[idx]._gradb, "_represent_transform" + ssposition.str() + "._b", iter);
    }

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

#endif /* SRC_PoolExGRNNClassifier_H_ */
