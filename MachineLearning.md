# Machine Learning

- Decision Tree
   - opencv code
   ```
   #include "opencv2/core/core.hpp"
   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/imgproc/imgproc.hpp"
   #include "opencv2/ml/ml.hpp"

   #include <iostream>
   using namespace cv;
   using namespace std;

   int main( int argc, char** argv )
   { 
      //10个训练样本
      float trainingDat[10][4]={{1,0,0,0},{1,1,0,1},{0,0,1,0},
      {0,1,1,1},{1,0,0,0},{1,0,1,1},{1,1,1,1},{0,1,1,0},{1,1,0,1},{0,0,0,1}};
      Mat trainingDataMat(10,4,CV_32FC1,trainingDat);

      //样本的分类结果，作为标签供训练决策树分类器
      float responses[10]={1,1,0,0,0,0,1,1,1,0}; // 1代表考取，0代表不考
      Mat responsesMat(10,1,CV_32FC1,responses);

      float priors[4]={1,1,1,1};  //先验概率，这里每个特征的概率都是一样的

      //定义决策树参数
      CvDTreeParams params(15,  //决策树的最大深度
         1,   //决策树叶子节点的最小样本数
         0,   //回归精度，本例中忽略
         false,  //不使用替代分叉属性
         25,  //最大的类数量
         0,   //不需要交叉验证
         false,  //不需要使用1SE规则
         false,   //不对分支进行修剪
         priors   //先验概率
         );
      //掩码
      Mat varTypeMat(5,1,CV_8U,Scalar::all(1));

      CvDTree tree;
      tree.train(trainingDataMat,  //训练样本
         CV_ROW_SAMPLE,  //样本矩阵的行表示样本，列表示特征
         responsesMat,   //样本的响应值矩阵
         Mat(),
         Mat(),
         varTypeMat,   //类形式的掩码
         Mat(),   //没有属性确实
         params   //决策树参数
         );
      CvMat varImportance;
      varImportance=tree.getVarImportance();
      cout<<"各项所占的权重分别为：\n\n";
      string item;
      for(int i=0;i<varImportance.cols*varImportance.rows;i++)
      {
         switch (i)
         {
         case 0:
            item="马化腾深大毕业：";
            break;
         case 1:
            item="深大妹子多：";
            break;
         case 2:
            item="深圳台风多：";
            break;
         case 3:
            item="深圳房价高：";
            break;
         default:
            break;
         }
         float value =varImportance.data.db[i];
         cout<<item<<value<<endl<<endl;
      }
      float myData[4]={0,1,1,0};  //测试样本
      Mat myDataMat(4,1,CV_32FC1,myData);
      double r=tree.predict(myDataMat,Mat(),false)->value; //获得预测结果

      if(r==(double)1.0)
      {
         cout<<endl<<"预测结果是： 考取深圳大学"<<endl<<endl;
      }
      else
      {
         cout<<endl<<"预测结果是： 不考取深圳大学"<<endl<<endl;

      }	
      system("pause");
      return 0;
   }
   ```
- Random Forest
   - opencv
   ```
   #include "opencv2/core/core.hpp"  
   #include "opencv2/highgui/highgui.hpp"  
   #include "opencv2/imgproc/imgproc.hpp"  
   #include "opencv2/ml/ml.hpp"  

   #include <iostream>  
   using namespace cv;  
   using namespace std;  

   int main( int argc, char** argv )  
   {     
       double trainingData[28][2]={{210.4, 3}, {240.0, 3}, {300.0, 4}, {153.4, 3}, {138.0, 3},  
                                   {194.0,4}, {189.0, 3}, {126.8, 3}, {132.0, 2}, {260.9, 4},  
                                   {176.7,3}, {160.4, 3}, {389.0, 3}, {145.8, 3}, {160.0, 3},  
                                   {141.6,2}, {198.5, 4}, {142.7, 3}, {149.4, 3}, {200.0, 3},  
                                   {447.8,5}, {230.0, 4}, {123.6, 3}, {303.1, 4}, {188.8, 2},  
                                   {196.2,4}, {110.0, 3}, {252.6, 3} };  
       CvMat trainingDataCvMat = cvMat( 28, 2, CV_32FC1, trainingData );  

       float responses[28] = { 399900, 369000, 539900, 314900, 212000, 239999, 329999,  
                           259900, 299900, 499998, 252900, 242900, 573900, 464500,  
                           329900, 232000, 299900, 198999, 242500, 347000, 699900,   
                           449900, 199900, 599000, 255000, 259900, 249900, 469000};  
       CvMat responsesCvMat = cvMat( 28, 1, CV_32FC1, responses );  

       CvRTParams params= CvRTParams(10, 2, 0, false,16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER );  

       CvERTrees etrees;  
       etrees.train(&trainingDataCvMat, CV_ROW_SAMPLE, &responsesCvMat,   
                                     NULL, NULL, NULL, NULL,params);  

       double sampleData[2]={201.5, 3};  
       Mat sampleMat(2, 1, CV_32FC1, sampleData);  
       float r = etrees.predict(sampleMat);  
       cout<<endl<<"result:  "<<r<<endl;  

       return 0;  
   } 
   ```
   
- Bayes Classifier
   - opencv
   ```
   //openCV中贝叶斯分类器的API函数用法举例
   #include "stdafx.h"
   #include "cv.h"
   #include "highgui.h"
   #include "cxcore.h"
   #include"opencv.hpp"
   #include "iostream"
   using namespace cv;
   using namespace std;

   //10个样本特征向量维数为12的训练样本集，第一列为该样本的类别标签
   double inputArr[10][13] = 
   {
      1,0.708333,1,1,-0.320755,-0.105023,-1,1,-0.419847,-1,-0.225806,0,1, 
      -1,0.583333,-1,0.333333,-0.603774,1,-1,1,0.358779,-1,-0.483871,0,-1,
      1,0.166667,1,-0.333333,-0.433962,-0.383562,-1,-1,0.0687023,-1,-0.903226,-1,-1,
      -1,0.458333,1,1,-0.358491,-0.374429,-1,-1,-0.480916,1,-0.935484,0,-0.333333,
      -1,0.875,-1,-0.333333,-0.509434,-0.347032,-1,1,-0.236641,1,-0.935484,-1,-0.333333,
      -1,0.5,1,1,-0.509434,-0.767123,-1,-1,0.0534351,-1,-0.870968,-1,-1,
      1,0.125,1,0.333333,-0.320755,-0.406393,1,1,0.0839695,1,-0.806452,0,-0.333333,
      1,0.25,1,1,-0.698113,-0.484018,-1,1,0.0839695,1,-0.612903,0,-0.333333,
      1,0.291667,1,1,-0.132075,-0.237443,-1,1,0.51145,-1,-0.612903,0,0.333333,
      1,0.416667,-1,1,0.0566038,0.283105,-1,1,0.267176,-1,0.290323,0,1
   };

   //一个测试样本的特征向量
   double testArr[]=
   {
      0.25,1,1,-0.226415,-0.506849,-1,-1,0.374046,-1,-0.83871,0,-1
   };


   int _tmain(int argc, _TCHAR* argv[])
   {
      Mat trainData(10, 12, CV_32FC1);//构建训练样本的特征向量
      for (int i=0; i<10; i++)
      {
         for (int j=0; j<12; j++)
         {
            trainData.at<float>(i, j) = inputArr[i][j+1];
         }
      }

      Mat trainResponse(10, 1, CV_32FC1);//构建训练样本的类别标签
      for (int i=0; i<10; i++)
      {
         trainResponse.at<float>(i, 0) = inputArr[i][0];
      }

      CvNormalBayesClassifier nbc;
      bool trainFlag = nbc.train(trainData, trainResponse);//进行贝叶斯分类器训练
      if (trainFlag)
      {
         cout<<"train over..."<<endl;
         nbc.save("D:/normalBayes.txt");
      }
      else
      {
         cout<<"train error..."<<endl;
         system("pause");
         exit(-1);
      }


      CvNormalBayesClassifier testNbc;
      testNbc.load("D:/normalBayes.txt");

      Mat testSample(1, 12, CV_32FC1);//构建测试样本
      for (int i=0; i<12; i++)
      {
         testSample.at<float>(0, i) = testArr[i];
      }

      float flag = testNbc.predict(testSample);//进行测试
      cout<<"flag = "<<flag<<endl;

      system("pause");
      return 0;
   }
   ```
- PReLU
   - [[2015][iccv]Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
   - f(x) = alpha * x for x < 0,  f(x) = x for x>=0
- ELU
   - [[2016][ICLR]FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)](https://arxiv.org/pdf/1511.07289v1.pdf)
   - f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x>=0

- ThresholdedReLU
   - [[2015][ICLR]ZERO-BIAS AUTOENCODERS AND THE BENEFITS OF CO-ADAPTING FEATURES](https://arxiv.org/pdf/1402.3337.pdf)
   - f(x) = x for x > theta,f(x) = 0 otherwise

- Dropout
   - [[2014]Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
   
