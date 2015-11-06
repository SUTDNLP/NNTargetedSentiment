#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;


class Options
{
public:
  /**
   * Refuse to train on words which have a corpus frequency less than
   * this number.
   */
  int wordCutOff;

  /**
   * Refuse to train on features which have a corpus frequency less than
   * this number.
   */
  int featCutOff;

  /**
   * Refuse to train on chars which have a corpus frequency less than
   * this number.
   */
  int charCutOff;

  /**
   * Model weights will be initialized to random values within the
   * range {@code [-initRange, initRange]}.
   */
  double initRange;

  /**
   * Maximum number of iterations for training
   */
  int maxIter;

  /**
   * Size of mini-batch for training. A random subset of training
   * examples of this size will be used to train the classifier on each
   * iteration.
   */
  int batchSize;

  /**
   * An epsilon value added to the denominator of the AdaGrad
   * expression for numerical stability
   */
  double adaEps;

  /**
   * Initial global learning rate for AdaGrad training
   */
  double adaAlpha;

  /**
   * NN Regularization parameter. All weight updates are scaled by this
   * single parameter.
   */
  double regParameter;

  /**
   * Dropout probability. For each training example we randomly choose
   * some amount of units to disable in the neural network classifier.
   * This probability controls the proportion of units "dropped out."
   */
  double dropProb;

  /**
   * Size of the neural network hidden layer.
   */
  int wordHiddenSize;
  int charHiddenSize;
  int rnnHiddenSize;
  int hiddenSize;

  int wordEmbSize;
  int wordcontext;

  int charcontext;
  int charEmbSize;


  bool wordEmbFineTune;
  bool charEmbFineTune;

  int verboseIter;
  bool saveIntermediate;
  bool train;
  int maxInstance;

  vector<string> testFiles;

  string outBest;

  int rnnFunc;

  int lstmFunc;

  int cnnFunc;

  int relu; //0, no relu; 1, max(0,x); 2, leaky; 3, soft

  int hislinear;

  int removePool;

  int removeCharPool;

  bool seg;


  Options()
  {
    wordCutOff = 0;
    featCutOff = 0;
    charCutOff = 0;

    initRange = 0.01;
    maxIter = 1000;
    batchSize = 1;

    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.0;
    wordHiddenSize = 200;
    charHiddenSize = 100;
    rnnHiddenSize = 100;
    hiddenSize = 100;

    wordcontext = 1;
    wordEmbSize = 50;
    charcontext = 1;
    charEmbSize = 10;
    wordEmbFineTune = false;
    charEmbFineTune = true;
    verboseIter = 100;
    saveIntermediate = true;
    train = false;
    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    rnnFunc = 0;
    lstmFunc = 0;
    cnnFunc = 0;
    relu = 0;
    hislinear = 1;
    removePool = 0;
    removeCharPool = 0;
    seg = true;
  }

  virtual ~Options()
  {

  }



  void setOptions(const vector<string> &vecOption)
  {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff") wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "featCutOff") featCutOff = atoi(pr.second.c_str());
      if (pr.first == "charCutOff") charCutOff = atoi(pr.second.c_str());

      if (pr.first == "initRange") initRange = atof(pr.second.c_str());
      if (pr.first == "maxIter") maxIter = atoi(pr.second.c_str());

      if (pr.first == "batchSize") batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps") adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha") adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter") regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb") dropProb = atof(pr.second.c_str());


      if (pr.first == "wordHiddenSize") wordHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "charHiddenSize") charHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "rnnHiddenSize") rnnHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "hiddenSize") hiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordcontext") wordcontext = atoi(pr.second.c_str());
      if (pr.first == "charcontext") charcontext = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize") wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "charEmbSize") charEmbSize = atoi(pr.second.c_str());


      if (pr.first == "wordEmbFineTune")
      {
        if(pr.second == "true") wordEmbFineTune = true;
        else wordEmbFineTune = false;
      }

      if (pr.first == "charEmbFineTune")
      {
        if(pr.second == "true") charEmbFineTune = true;
        else charEmbFineTune = false;
      }

      if (pr.first == "verboseIter") verboseIter = atoi(pr.second.c_str());

      if (pr.first == "train")
      {
        if(pr.second == "true") train = true;
        else train = false;
      }
      if (pr.first == "saveIntermediate")
      {
        if(pr.second == "true") saveIntermediate = true;
        else saveIntermediate = false;
      }

      if (pr.first == "maxInstance") maxInstance = atoi(pr.second.c_str());

      if (pr.first == "testFile") testFiles.push_back(pr.second);


      if (pr.first == "outBest") outBest = pr.second;

      if (pr.first == "rnnFunc") rnnFunc = atoi(pr.second.c_str());

      if (pr.first == "lstmFunc") lstmFunc = atoi(pr.second.c_str());

      if (pr.first == "cnnFunc") cnnFunc = atoi(pr.second.c_str());

      if (pr.first == "relu") relu = atoi(pr.second.c_str());
      if (pr.first == "hislinear") hislinear = atoi(pr.second.c_str());
      if (pr.first == "removePool") removePool = atoi(pr.second.c_str());
      if (pr.first == "removeCharPool") removeCharPool = atoi(pr.second.c_str());

      if (pr.first == "seg")
      {
        if(pr.second == "true") seg = true;
        else seg = false;
      }
    }
  }


  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "featCutOff = " << featCutOff << std::endl;
    std::cout << "charCutOff = " << charCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;
    std::cout << "wordHiddenSize = " << wordHiddenSize << std::endl;
    std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
    std::cout << "rnnHiddenSize = " << rnnHiddenSize << std::endl;
    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "wordcontext = " << wordcontext << std::endl;
    std::cout << "charEmbSize = " << charEmbSize << std::endl;
    std::cout << "charcontext = " << charcontext << std::endl;
    std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    std::cout << "charEmbFineTune = " << charEmbFineTune << std::endl;
    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveItermediate = " << saveIntermediate << std::endl;
    std::cout << "train = " << train << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;
    for(int idx = 0; idx < testFiles.size(); idx++)
    {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;
    std::cout << "rnnFunc = " << rnnFunc << std::endl;
    std::cout << "lstmFunc = " << lstmFunc << std::endl;
    std::cout << "cnnFunc = " << cnnFunc << std::endl;
    std::cout << "relu = " << relu << std::endl;
    std::cout << "hislinear = " << hislinear << std::endl;
    std::cout << "removePool = " << removePool << std::endl;
    std::cout << "removeCharPool = " << removeCharPool << std::endl;
    std::cout << "seg = " << seg << std::endl;
  }

  void load(const std::string& infile)
  {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1)
    {
        string strLine;
        if (!my_getline(inf, strLine)) {
              break;
        }
        if (strLine.empty()) continue;
        vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif

