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
   
