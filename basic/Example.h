/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "Feature.h"

using namespace std;

class Example {

public:
  vector<int> m_labels;
  vector<Feature> m_features;
  int left, right;
  int goldLabel;

public:
  Example()
  {
    clear();
  }
  virtual ~Example()
  {

  }

  void clear()
  {
    m_labels.clear();
    m_features.clear();
    left = -1;
    right = -1;
    goldLabel = -1;
  }


};

#endif /* SRC_EXAMPLE_H_ */
