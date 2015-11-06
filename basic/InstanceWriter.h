#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;
/*
	this class writes conll-format result (no srl-info).
*/
class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
	  if (!m_outf.is_open()) return -1;

	  const vector<string> &labels = pInstance->labels;

	  for (int i = 0; i < labels.size(); ++i) {
	    m_outf << pInstance->words[i] << " ";
	    /*
	    for(int j = 0; j < pInstance->sparsefeatures[i].size(); j++)
	    {
	      m_outf << pInstance->sparsefeatures[i][j] << " ";
	    }
	    for(int j = 0; j < pInstance->charfeatures[i].size(); j++)
	    {
	      m_outf << pInstance->charfeatures[i][j] << " ";
	    }*/
	    m_outf << labels[i] << endl;
	  }
	  m_outf << endl;
	  return 0;
	}
};

#endif

