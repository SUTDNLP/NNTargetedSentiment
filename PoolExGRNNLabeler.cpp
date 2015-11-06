/*
 * Labeler.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#include "PoolExGRNNLabeler.h"

#include "Argument_helper.h"
#include <iomanip>

Labeler::Labeler() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  seperateKey = "#";

}

Labeler::~Labeler() {
  // TODO Auto-generated destructor stub
  m_classifier.release();
}

int Labeler::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> feature_stat;
  hash_map<string, int> word_stat;
  hash_map<string, int> char_stat;
  m_labelAlphabet.clear();

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &words = pInstance->words;
    const vector<string> &labels = pInstance->labels;
    const vector<vector<string> > &sparsefeatures = pInstance->sparsefeatures;
    const vector<vector<string> > &charfeatures = pInstance->charfeatures;

    vector<string> features;
    int curInstSize = labels.size();
    int labelId;
    for (int i = 0; i < curInstSize; ++i) {
      if (labels[i].length() > 2) {
        labelId = m_labelAlphabet.from_string(labels[i].substr(2));
      }

      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
      for (int j = 0; j < charfeatures[i].size(); j++)
        char_stat[charfeatures[i][j]]++;
      for (int j = 0; j < sparsefeatures[i].size(); j++)
        feature_stat[sparsefeatures[i][j]]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
  cout << "Label num: " << m_labelAlphabet.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total feature num: " << feature_stat.size() << endl;

  m_featAlphabet.clear();
  m_wordAlphabet.clear();
  m_wordAlphabet.from_string(nullkey);
  m_wordAlphabet.from_string(unknownkey);
  m_charAlphabet.clear();
  m_charAlphabet.from_string(nullkey);
  m_charAlphabet.from_string(unknownkey);

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = feature_stat.begin(); feat_iter != feature_stat.end(); feat_iter++) {
    if (feat_iter->second > m_options.featCutOff) {
      m_featAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  cout << "Remain feature num: " << m_featAlphabet.size() << endl;
  cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_charAlphabet.size() << endl;

  m_labelAlphabet.set_fixed_flag(true);
  m_featAlphabet.set_fixed_flag(true);
  m_wordAlphabet.set_fixed_flag(true);
  m_charAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding word Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> word_stat;
  m_wordAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<string> &words = pInstance->words;

    int curInstSize = words.size();
    for (int i = 0; i < curInstSize; ++i) {
      string curword = normalize_to_lowerwithdigit(words[i]);
      word_stat[curword]++;
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
    if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
      m_wordAlphabet.from_string(feat_iter->first);
    }
  }

  m_wordAlphabet.set_fixed_flag(true);

  return 0;
}

int Labeler::addTestCharAlpha(const vector<Instance>& vecInsts) {
  cout << "Adding char Alphabet..." << endl;

  int numInstance;
  hash_map<string, int> char_stat;
  m_charAlphabet.set_fixed_flag(false);

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];

    const vector<vector<string> > &charfeatures = pInstance->charfeatures;

    int curInstSize = charfeatures.size();
    for (int i = 0; i < curInstSize; ++i) {
      for (int j = 1; j < charfeatures[i].size(); j++)
        char_stat[charfeatures[i][j]]++;
    }
    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  hash_map<string, int>::iterator feat_iter;
  for (feat_iter = char_stat.begin(); feat_iter != char_stat.end(); feat_iter++) {
    if (!m_options.charEmbFineTune || feat_iter->second > m_options.charCutOff) {
      m_charAlphabet.from_string(feat_iter->first);
    }
  }

  m_charAlphabet.set_fixed_flag(true);

  return 0;
}

void Labeler::extractFeature(Feature& feat, const Instance* pInstance, int idx) {
  feat.clear();

  const vector<string>& words = pInstance->words;
  int sentsize = words.size();
  string curWord = idx >= 0 && idx < sentsize ? normalize_to_lowerwithdigit(words[idx]) : nullkey;

  // word features
  int unknownId = m_wordAlphabet.from_string(unknownkey);

  int curWordId = m_wordAlphabet.from_string(curWord);
  if (curWordId >= 0) {
    feat.words.push_back(curWordId);
    if(idx >= 0 && idx < sentsize && m_sentiwords.find(words[idx]) != m_sentiwords.end())
    {
      feat.words.push_back(curWordId);
    }
  }
  else {
    feat.words.push_back(unknownId);
  }



  // char features
  unknownId = m_charAlphabet.from_string(unknownkey);

  const vector<vector<string> > &charfeatures = pInstance->charfeatures;

  const vector<string>& cur_chars = charfeatures[idx];
  int cur_char_size = cur_chars.size();

  // actually we support a max window of m_options.charcontext = 2
  for (int i = 0; i < cur_char_size; i++) {
    string curChar = cur_chars[i];

    int curCharId = m_charAlphabet.from_string(curChar);
    if (curCharId >= 0)
      feat.chars.push_back(curCharId);
    else
      feat.chars.push_back(unknownId);
  }

  int nullkeyId = m_charAlphabet.from_string(nullkey);
  if (feat.chars.empty()) {
    feat.chars.push_back(nullkeyId);
  }

  const vector<string>& linear_features = pInstance->sparsefeatures[idx];
  for (int i = 0; i < linear_features.size(); i++) {
    int curFeatId = m_featAlphabet.from_string(linear_features[i]);
    if (curFeatId >= 0)
      feat.linear_features.push_back(curFeatId);
  }

}

void Labeler::convert2Example(const Instance* pInstance, Example& exam) {
  exam.clear();
  const vector<string> &labels = pInstance->labels;
  int curInstSize = labels.size();
  int left, right;
  left = -1; right = -1;
  for (int i = 0; i < curInstSize; ++i) {
    if(labels[i].length() > 2)
    {
      right = i;
      if(left == -1) {
	      left = i;
	      string orcale = labels[i].substr(2);
	      int numLabels = m_labelAlphabet.size();
	      for (int j = 0; j < numLabels; ++j) {
	        string str = m_labelAlphabet.from_id(j);
	        if (str.compare(orcale) == 0)
	        {
	          exam.m_labels.push_back(1);
	          exam.goldLabel = j;
	        }
	        else
	          exam.m_labels.push_back(0);
	      }
      }
    }

    Feature feat;
    extractFeature(feat, pInstance, i);
    exam.m_features.push_back(feat);
  }
  if(left == -1 || right == -1)
    std::cout << "error" << std::endl;
  exam.left = left; exam.right = right;

}

void Labeler::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
  int numInstance;
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance *pInstance = &vecInsts[numInstance];
    Example curExam;
    convert2Example(pInstance, curExam);
    vecExams.push_back(curExam);

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  cout << numInstance << " " << endl;
}

void Labeler::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmb1File, const string& wordEmb2File, const string& charEmbFile, const string& sentiFile) {
  if (optionFile != "")
    m_options.load(optionFile);
  m_options.showOptions();
  vector<Instance> trainInsts, devInsts, testInsts;
  static vector<Instance> decodeInstResults;
  static Instance curDecodeInst;
  bool bCurIterBetter = false;

  m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  //Ensure that each file in m_options.testFiles exists!
  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
  }

  //std::cout << "Training example number: " << trainInsts.size() << std::endl;
  //std::cout << "Dev example number: " << trainInsts.size() << std::endl;
  //std::cout << "Test example number: " << trainInsts.size() << std::endl;

  createAlphabet(trainInsts);
  readSentimentWords(sentiFile);

  if (!m_options.wordEmbFineTune) {
    addTestWordAlpha(devInsts);
    addTestWordAlpha(testInsts);
    for (int idx = 0; idx < otherInsts.size(); idx++) {
      addTestWordAlpha(otherInsts[idx]);
    }
    cout << "Remain words num: " << m_wordAlphabet.size() << endl;
  }

  NRMat<double> wordEmb1;
  if (wordEmb1File != "") {
    readWordEmbeddings(wordEmb1File, wordEmb1);
  }
  
  NRMat<double> wordEmb2;
  if (wordEmb2File != "") {
    readWordEmbeddings(wordEmb2File, wordEmb2);
  } 

  NRMat<double> wordEmb;  
  if(wordEmb1File != "" && wordEmb2File != "" )
  {
    if(wordEmb1.nrows() != wordEmb2.nrows() || wordEmb1.nrows() != m_wordAlphabet.size()) 
      std::cout << "word emb rows do not match" << std::endl;
    int mergedRows = wordEmb1.nrows();
    int mergedCols = wordEmb1.ncols() + wordEmb2.ncols();
    std::cout << "merged word embedding dimension: " << mergedCols << std::endl;
		wordEmb.resize(mergedRows, mergedCols);
		for(int i = 0; i < mergedRows; i++)
		{
			for(int j = 0; j < wordEmb1.ncols(); j++)
			{
			  wordEmb[i][j] = wordEmb1[i][j];
			}
			for(int j = 0; j < wordEmb2.ncols(); j++)
			{
			  wordEmb[i][wordEmb1.ncols()+j] = wordEmb2[i][j];
			}
		}
  }
  else if(wordEmb1File != "")
  {
    wordEmb = wordEmb1;
  }
  else if(wordEmb1File != "")
  {
    wordEmb = wordEmb2;
  }
  else
  {
    wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
    wordEmb.randu(1000);
  }

  NRMat<double> charEmb;
  if (charEmbFile != "") {
    readWordEmbeddings(charEmbFile, charEmb);
  } else {
    charEmb.resize(m_charAlphabet.size(), m_options.wordEmbSize);
    charEmb.randu(1001);
  }

  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);
  m_classifier.init(wordEmb, m_options.wordcontext, wordEmb.ncols(), wordEmb.ncols(), m_labelAlphabet.size());
  m_classifier.resetRemove(m_options.removePool);
  m_classifier.setDropValue(m_options.dropProb);

  vector<Example> trainExamples, devExamples, testExamples;
  initialExamples(trainInsts, trainExamples);
  initialExamples(devInsts, devExamples);
  initialExamples(testInsts, testExamples);

  vector<int> otherInstNums(otherInsts.size());
  vector<vector<Example> > otherExamples(otherInsts.size());
  for (int idx = 0; idx < otherInsts.size(); idx++) {
    initialExamples(otherInsts[idx], otherExamples[idx]);
    otherInstNums[idx] = otherExamples[idx].size();
  }

  double bestDIS = 0;

  int inputSize = trainExamples.size();

  int batchBlock = inputSize / m_options.batchSize;
  if (inputSize % m_options.batchSize != 0)
    batchBlock++;

  srand(0);
  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;
  static vector<Example> subExamples;
  int devNum = devExamples.size(), testNum = testExamples.size();
  for (int iter = 0; iter < m_options.maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;

    random_shuffle(indexes.begin(), indexes.end());
    eval.reset();
    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
      subExamples.clear();
      int start_pos = updateIter * m_options.batchSize;
      int end_pos = (updateIter + 1) * m_options.batchSize;
      if (end_pos > inputSize)
        end_pos = inputSize;

      for (int idy = start_pos; idy < end_pos; idy++) {
        subExamples.push_back(trainExamples[indexes[idy]]);
      }

      int curUpdateIter = iter * batchBlock + updateIter;
      double cost = m_classifier.process(subExamples, curUpdateIter);

      eval.overall_label_count += m_classifier._eval.overall_label_count;
      eval.correct_label_count += m_classifier._eval.correct_label_count;

      if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
        //m_classifier.checkgrads(subExamples, curUpdateIter+1);
        std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
        std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
      }
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);

    }

    if (devNum > 0) {
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devExamples.size(); idx++) {
        vector<string> result_labels;
        predict(devExamples[idx].m_features, devExamples[idx].left, devExamples[idx].right, result_labels, devInsts[idx].words);

        if (m_options.seg)
          devInsts[idx].SegEvaluate(result_labels, metric_dev);
        else
          devInsts[idx].Evaluate(result_labels, metric_dev);

        if (!m_options.outBest.empty()) {
          curDecodeInst.copyValuesFrom(devInsts[idx]);
          curDecodeInst.assignLabel(result_labels);
          decodeInstResults.push_back(curDecodeInst);
        }
      }

      metric_dev.print();

      if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testExamples.size(); idx++) {
          vector<string> result_labels;
          predict(testExamples[idx].m_features, testExamples[idx].left, testExamples[idx].right, result_labels, testInsts[idx].words);

          if (m_options.seg)
            testInsts[idx].SegEvaluate(result_labels, metric_test);
          else
            testInsts[idx].Evaluate(result_labels, metric_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(testInsts[idx]);
            curDecodeInst.assignLabel(result_labels);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }

      for (int idx = 0; idx < otherExamples.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
          vector<string> result_labels;
          predict(otherExamples[idx][idy].m_features, otherExamples[idx][idy].left, otherExamples[idx][idy].right, result_labels, otherInsts[idx][idy].words);

          if (m_options.seg)
            otherInsts[idx][idy].SegEvaluate(result_labels, metric_test);
          else
            otherInsts[idx][idy].Evaluate(result_labels, metric_test);

          if (bCurIterBetter && !m_options.outBest.empty()) {
            curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
            curDecodeInst.assignLabel(result_labels);
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }

      if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
        std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
        bestDIS = metric_dev.getAccuracy();
        writeModelFile(modelFile);
      }

    }
    // Clear gradients
  }

}

int Labeler::predict(const vector<Feature>& features, const int& left, const int& right, vector<string>& outputs, const vector<string>& words) {
  assert(features.size() == words.size());
  vector<double> labelprobs;
  int labelId = m_classifier.predict(features, left, right, labelprobs);

  outputs.clear();

  string label = m_labelAlphabet.from_id(labelId);

  for (int idx = 0; idx < words.size(); idx++) {
    if(idx < left || idx > right) outputs.push_back("o");
    else if(idx == left) outputs.push_back("b-" +label);
    else outputs.push_back("i-" +label);
  }

  return 0;
}

void Labeler::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts);

  vector<Example> testExamples;
  initialExamples(testInsts, testExamples);

  int testNum = testExamples.size();
  vector<Instance> testInstResults;
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testExamples.size(); idx++) {
    vector<string> result_labels;
    predict(testExamples[idx].m_features, testExamples[idx].left, testExamples[idx].right, result_labels, testInsts[idx].words);
    testInsts[idx].SegEvaluate(result_labels, metric_test);
    Instance curResultInst;
    curResultInst.copyValuesFrom(testInsts[idx]);
    testInstResults.push_back(curResultInst);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  m_pipe.outputAllInstances(outputFile, testInstResults);

}

void Labeler::readWordEmbeddings(const string& inFile, NRMat<double>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  double sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      double curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        double curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] += curValue;
        wordEmb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Labeler::readSentimentWords(const string& inFile){
  m_sentiwords.clear();
  if(inFile == "") return;
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    chomp(strLine);
    if (strLine.empty())
      continue;
    m_sentiwords.insert(strLine);
  }
  std::cout << "sentiment word number: " << m_sentiwords.size() << std::endl;
}

/*
void Labeler::dumpFeatures(string outfile, vector<Example> goldInsts){
  NRVec<double> poolFeatures;
  ofstream outf;
  outf.open(outfile.c_str());
  if (!outf) {
    cout << "Writer::startWriting() open file err: " << outfile << endl;
  }
  for (int idx = 0; idx < goldInsts.size(); idx++) {
    m_classifier.dumpfeatures(goldInsts[idx].m_features, goldInsts[idx].left, goldInsts[idx].right, poolFeatures);
    outf << m_labelAlphabet.from_id(goldInsts[idx].goldLabel);
    for(int idd = 0; idd < poolFeatures.size(); idd++)
      outf << " " << idd+1 << ":" << std::setprecision(10) << poolFeatures[idd];
    outf << std::endl;
  }

  outf.close();
}
*/

void Labeler::loadModelFile(const string& inputModelFile) {

}

void Labeler::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif

  std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
  std::string wordEmb1File = "",  wordEmb2File = "", charEmbFile = "", optionFile = "";
  std::string outputFile = "";
  std::string sentiFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word1", "wordEmb1File", "named_string", "pretrained word embedding file 1 to train a model, optional when training", wordEmb1File);
  ah.new_named_string("word2", "wordEmb2File", "named_string", "pretrained word embedding file 2 to train a model, optional when training", wordEmb2File);  
  ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
  ah.new_named_string("sen", "sentiFile", "named_string", "sentiment word file to train a model, optional when training", sentiFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Labeler tagger;
  if (bTrain) {
    tagger.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmb1File, wordEmb2File, charEmbFile, sentiFile);
  } else {
    tagger.test(testFile, outputFile, modelFile);
  }

  //test(argv);
  //ah.write_values(std::cout);
#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif
}
