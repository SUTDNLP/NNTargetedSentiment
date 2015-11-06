NNTargetedSentiment
======
NNTargetedSentiment is a package for targeted sentiment analysis using neural networks based on package [LibN3L](https://github.com/SUTDNLP/LibN3L).  
Targeted sentiment analysis aims to detect the sentiment polarity of a given span in a sentence. The span can be one entity or one expression.  
Please refer to the following papers for more detailed description:
[Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](http://www.aclweb.org/anthology/P/P14/P14-2009.pdf)  
[Target-dependent Twitter Sentiment Classification with Rich Automatic Features](http://people.sutd.edu.sg/~yue_zhang/pub/ijcai15.tin.pdf)  
[SemEval-2013 Task 2: Sentiment Analysis in Twitter](http://www.aclweb.org/anthology/S/S13/S13-2052.pdf)  


Compile
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and compile it. 
* Open [CMakeLists.txt](CMakeLists.txt) and change "../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.  

`cmake .`  
`make`  

Input data format 
======
conll format, sentences are seperated by one empty line, each line contains one word and its labels.  
The label is bio format: b-xx denotes the begining word of a span with polarity xx; i-xx denotes the non-starting words of a span with polarity xx; o denotes that it is not a target.  
3 o  
by o  
britney b-positive  
spears i-positive  
is o  
an o  
amazing o  
song o  
  
paris b-neutral  
hilton i-neutral  
denies o  
racism o  
allegations o  
\- o  
digital o  
spy o  



