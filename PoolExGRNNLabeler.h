/*
 * Labeler.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNCRF_H_
#define SRC_NNCRF_H_


#include "N3L.h"
#include "basic/PoolExGRNNClassifier.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Labeler {

public:
  std::string nullkey;
  std::string unknownkey;
  std::string seperateKey;

public:
  Alphabet m_featAlphabet;
  Alphabet m_labelAlphabet;

  Alphabet m_wordAlphabet;
  Alphabet m_charAlphabet;

public:
  Options m_options;
  hash_set<string> m_sentiwords;
  Pipe m_pipe;

#if USE_CUDA==1
  PoolExGRNNClassifier<gpu> m_classifier;
#else
  PoolExGRNNClassifier<cpu> m_classifier;
#endif

public:
  void readWordEmbeddings(const string& inFile, NRMat<double>& wordEmb);
  void readSentimentWords(const string& inFile);
  //void dumpFeatures(string outFile, vector<Example> goldInsts);

public:
  Labeler();
  virtual ~Labeler();

public:

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);
  int addTestCharAlpha(const vector<Instance>& vecInsts);

  void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
  void extractFeature(Feature& feat, const Instance* pInstance, int idx);

  void convert2Example(const Instance* pInstance, Example& exam);
  void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& wordEmb1File, const string& wordEmb2File, const string& charEmbFile, const string& sentiFile);
  int predict(const vector<Feature>& features, const int& left, const int& right, vector<string>& outputs, const vector<string>& words);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCRF_H_ */
