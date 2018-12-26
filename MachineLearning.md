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
- SVM
   - opencv
   ```
   #include <iostream>
   #include <opencv2/core/core.hpp>
   #include <opencv2/highgui/highgui.hpp>
   #include <opencv2/ml/ml.hpp>
   #include <CTYPE.H>  

   #define	NTRAINING_SAMPLES	100			// Number of training samples per class
   #define FRAC_LINEAR_SEP		0.9f	    // Fraction of samples which compose the linear separable part

   using namespace cv;
   using namespace std;

   int main(int argc, char* argv[])  
   {  
      int size = 400; // height and widht of image  
      const int s = 1000; // number of data  
      int i, j,sv_num;  
      IplImage* img;  

      CvSVM svm ;  
      CvSVMParams param;  
      CvTermCriteria criteria; // 停止迭代标准  
      CvRNG rng = cvRNG();  
      CvPoint pts[s]; // 定义1000个点  
      float data[s*2]; // 点的坐标  
      int res[s]; // 点的类别  

      CvMat data_mat, res_mat;  
      CvScalar rcolor;  

      const float* support;  

      // 图像区域的初始化  
      img = cvCreateImage(cvSize(size,size),IPL_DEPTH_8U,3);  
      cvZero(img);  

      // 学习数据的生成  
      for (i=0; i<s;++i)  
      {  
         pts[i].x = cvRandInt(&rng)%size;  
         pts[i].y = cvRandInt(&rng)%size;  

         if (pts[i].y>50*cos(pts[i].x*CV_PI/100)+200)  
         {  
            cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(255,0,0));  
            cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(255,0,0));  
            res[i]=1;  
         }  
         else  
         {  
            if (pts[i].x>200)  
            {  
               cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(0,255,0));  
               cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(0,255,0));  
               res[i]=2;  
            }  
            else  
            {  
               cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(0,0,255));  
               cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(0,0,255));  
               res[i]=3;  
            }  
         }  
      }  

      // 学习数据的现实  
      cvNamedWindow("SVM训练样本空间及分类",CV_WINDOW_AUTOSIZE);  
      cvShowImage("SVM训练样本空间及分类",img);  
      cvWaitKey(0);  

      // 学习参数的生成  
      for (i=0;i<s;++i)  
      {  
         data[i*2] = float(pts[i].x)/size;  
         data[i*2+1] = float(pts[i].y)/size;  
      }  

      cvInitMatHeader(&data_mat,s,2,CV_32FC1,data);  
      cvInitMatHeader(&res_mat,s,1,CV_32SC1,res);  
      criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,FLT_EPSILON);  
      param = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,10.0,8.0,1.0,10.0,0.5,0.1,NULL,criteria);  

      svm.train(&data_mat,&res_mat,NULL,NULL,param);  

      // 学习结果绘图  
      for (i=0;i<size;i++)  
      {  
         for (j=0;j<size;j++)  
         {  
            CvMat m;  
            float ret = 0.0;  
            float a[] = {float(j)/size,float(i)/size};  
            cvInitMatHeader(&m,1,2,CV_32FC1,a);  
            ret = svm.predict(&m);  

            switch((int)ret)  
            {  
            case 1:  
               rcolor = CV_RGB(100,0,0);  
               break;  
            case 2:  
               rcolor = CV_RGB(0,100,0);  
               break;  
            case 3:  
               rcolor = CV_RGB(0,0,100);  
               break;  
            }  
            cvSet2D(img,i,j,rcolor);  
         }  
      }  


      // 为了显示学习结果，通过对输入图像区域的所有像素（特征向量）进行分类，然后对输入的像素用所属颜色等级的颜色绘图  
      for(i=0;i<s;++i)  
      {  
         CvScalar rcolor;  
         switch(res[i])  
         {  
         case 1:  
            rcolor = CV_RGB(255,0,0);  
            break;  
         case 2:  
            rcolor = CV_RGB(0,255,0);  
            break;  
         case 3:  
            rcolor = CV_RGB(0,0,255);  
            break;  
         }  
         cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),rcolor);  
         cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),rcolor);             
      }  

      // 支持向量的绘制  
      sv_num = svm.get_support_vector_count();  
      for (i=0; i<sv_num;++i)  
      {  
         support = svm.get_support_vector(i);  
         cvCircle(img,cvPoint((int)(support[0]*size),(int)(support[i]*size)),5,CV_RGB(200,200,200));  
      }  

      cvNamedWindow("SVM",CV_WINDOW_AUTOSIZE);  
      cvShowImage("SVM分类结果及支持向量",img);  
      cvWaitKey(0);  
      cvDestroyWindow("SVM");  
      cvReleaseImage(&img);  
      return 0;  
   }
   ```

- 逻辑回归Logistic Regression
   - opencv例子
   ```
   #include "opencv.hpp"
   #include <string>
   #include <vector>
   #include <memory>
   #include <opencv2/opencv.hpp>
   #include <opencv2/ml.hpp>
   #include "common.hpp"


   ////////////////////////////////// Logistic Regression ///////////////////////////////
   static void show_image(const cv::Mat& data, int columns, const std::string& name)
   {
      cv::Mat big_image;
      for (int i = 0; i < data.rows; ++i) {
         big_image.push_back(data.row(i).reshape(0, columns));
      }

      cv::imshow(name, big_image);
      cv::waitKey(0);
   }

   static float calculate_accuracy_percent(const cv::Mat& original, const cv::Mat& predicted)
   {
      return 100 * (float)cv::countNonZero(original == predicted) / predicted.rows;
   }

   int test_opencv_logistic_regression_train()
   {
      const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
      cv::Mat data, labels, result;

      for (int i = 1; i < 11; ++i) {
         const std::vector<std::string> label{ "0_", "1_" };

         for (const auto& value : label) {
            std::string name = std::to_string(i);
            name = image_path + value + name + ".jpg";

            cv::Mat image = cv::imread(name, 0);
            if (image.empty()) {
               fprintf(stderr, "read image fail: %s\n", name.c_str());
               return -1;
            }

            data.push_back(image.reshape(0, 1));
         }
      }

      data.convertTo(data, CV_32F);
      //show_image(data, 28, "train data");

      std::unique_ptr<float[]> tmp(new float[20]);
      for (int i = 0; i < 20; ++i) {
         if (i % 2 == 0) tmp[i] = 0.f;
         else tmp[i] = 1.f;
      }
      labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

      cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::create();
      lr->setLearningRate(0.00001);
      lr->setIterations(100);
      lr->setRegularization(cv::ml::LogisticRegression::REG_DISABLE);
      lr->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
      lr->setMiniBatchSize(1);

      CHECK(lr->train(data, cv::ml::ROW_SAMPLE, labels));

      const std::string save_file{ "E:/GitCode/NN_Test/data/logistic_regression_model.xml" }; // .xml, .yaml, .jsons
      lr->save(save_file);

      return 0;
   }

   int test_opencv_logistic_regression_predict()
   {
      const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
      cv::Mat data, labels, result;

      for (int i = 11; i < 21; ++i) {
         const std::vector<std::string> label{ "0_", "1_" };

         for (const auto& value : label) {
            std::string name = std::to_string(i);
            name = image_path + value + name + ".jpg";

            cv::Mat image = cv::imread(name, 0);
            if (image.empty()) {
               fprintf(stderr, "read image fail: %s\n", name.c_str());
               return -1;
            }

            data.push_back(image.reshape(0, 1));
         }
      }

      data.convertTo(data, CV_32F);
      //show_image(data, 28, "test data");

      std::unique_ptr<int[]> tmp(new int[20]);
      for (int i = 0; i < 20; ++i) {
         if (i % 2 == 0) tmp[i] = 0;
         else tmp[i] = 1;
      }
      labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

      const std::string model_file{ "E:/GitCode/NN_Test/data/logistic_regression_model.xml" };
      cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::load(model_file);

      lr->predict(data, result);

      fprintf(stdout, "predict result: \n");
      std::cout << "actual: " << labels.t() << std::endl;
      std::cout << "target: " << result.t() << std::endl;
      fprintf(stdout, "accuracy: %.2f%%\n", calculate_accuracy_percent(labels, result));

      return 0;
   }

   ```
   
- 判别分析
   - 比如LDA
   - 降维的一种， 假设数据服从正太分布
   - opencv例子
   ```
   #include <iostream>
   #include <contrib\contrib.hpp>
   #include <cxcore.hpp>
   using namespace cv;
   using namespace std;

   int main(void)
   {
      //sampledata
      double sampledata[6][2]={{0,1},{0,2},{2,4},{8,0},{8,2},{9,4}};
      Mat mat=Mat(6,2,CV_64FC1,sampledata);
      //labels
      vector<int>labels;

      for(int i=0;i<mat.rows;i++)
      {
         if(i<mat.rows/2)
         {
            labels.push_back(0);
         }
         else
         {
            labels.push_back(1);
         }
      }

      //do LDA
      //初始化并计算，构造函数中带有计算
      LDA lda=LDA(mat,labels,1);
      //get the eigenvector
      //获得特征向量
      Mat eivector=lda.eigenvectors().clone();

      cout<<"特征向量（double）类型:"<<endl;
      for(int i=0;i<eivector.rows;i++)
      {
         for(int j=0;j<eivector.cols;j++)
         {
            cout<<eivector.ptr<double>(i)[j]<<" ";
         }
         cout<<endl;
      }


      //------------------------------计算两个类心------------
      //针对两类分类问题，计算两个数据集的中心
      int classNum=2;
      vector<Mat> classmean(classNum);
      vector<int> setNum(classNum);

      for(int i=0;i<classNum;i++)
      {
         classmean[i]=Mat::zeros(1,mat.cols,mat.type());  //初始化类中均值为0
         setNum[i]=0;  //每一类中的条目数
      }

      Mat instance;
      for(int i=0;i<mat.rows;i++)
      {
         instance=mat.row(i);//获取第i行
         if(labels[i]==0)  //如果标签为0
         {	
            add(classmean[0], instance, classmean[0]);  //矩阵相加
            setNum[0]++;  //数量相加
         }
         else if(labels[i]==1)  //对于第1类的处理
         {
            add(classmean[1], instance, classmean[1]);
            setNum[1]++;
         }
         else
         {}
      }
      for(int i=0;i<classNum;i++)   //计算每一类的均值
      {
         classmean[i].convertTo(classmean[i],CV_64FC1,1.0/static_cast<double>(setNum[i]));
      }
      //----------------------------------END计算类心-------------------------


      vector<Mat> cluster(classNum);  //一共2类


      cout<<"特征向量："<<endl;
      cout<<eivector<<endl;   //此时的特征向量是一个列向量


      cout<<endl<<endl;
      cout<<"第一种方式(手动计算)："<<endl;
       //1.投影的第一种方式：Y=X*W
      //有的教程写成Y=W^T*X,（此时的X是列向量看待的所以需要将w转置）
      for(int i=0;i<classNum;i++)
      {
         cluster[i]=Mat::zeros(1,1,mat.type()); //初始化0
         //特征向量的转置同类均值相乘)
         cluster[i]=classmean[i]*eivector;
      }

      cout<<"The project cluster center is:"<<endl;  //计算均值的投影
      for(int i=0;i<classNum;i++) //输出两类中心的投影值
      {
         cout<<cluster[i].at<double>(0)<<endl;
      }

      //2.第二种方式使用内置函数计算
      //第一个中心
      cout<<endl<<"第二种方式:";
      cout<<endl<<"第一个类均值的投影:"<<endl;
      cout<<lda.project(classmean[0]).at<double>(0);
      cout<<endl<<"第二个类均值的投影"<<endl;
      cout<<lda.project(classmean[1]).at<double>(0);


      system("pause");
      return 0;
   }
   ```
- 提升算法Boosting
   - 比如AdaBoost
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
       //训练样本  
       float trainingData[42][2]={ {40, 55},{35, 35},{55, 15},{45, 25},{10, 10},{15, 15},{40, 10},  
                               {30, 15},{30, 50},{100, 20},{45, 65},{20, 35},{80, 20},{90, 5},  
                               {95, 35},{80, 65},{15, 55},{25, 65},{85, 35},{85, 55},{95, 70},  
                               {105, 50},{115, 65},{110, 25},{120, 45},{15, 45},  
                               {55, 30},{60, 65},{95, 60},{25, 40},{75, 45},{105, 35},{65, 10},  
                               {50, 50},{40, 35},{70, 55},{80, 30},{95, 45},{60, 20},{70, 30},  
                               {65, 45},{85, 40}   };  
       Mat trainingDataMat(42, 2, CV_32FC1, trainingData);   
       //训练样本的响应值  
       float responses[42] = {'R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R',  
                               'R','R','R','R','R','R','R','R','R','R',  
                           'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B' };  
       Mat responsesMat(42, 1, CV_32FC1, responses);  

       float priors[2] = {1, 1};    //先验概率  

       CvBoostParams params( CvBoost::REAL, // boost_type    
                             10, // weak_count    
                             0.95, // weight_trim_rate    
                             15, // max_depth    
                             false, // use_surrogates    
                             priors // priors   
                             );    

       CvBoost boost;  
       boost.train (   trainingDataMat,   
                       CV_ROW_SAMPLE,   
                       responsesMat,  
                       Mat(),    
                       Mat(),  
                       Mat(),  
                       Mat(),    
                       params  
                       );    
       //预测样本  
       float myData[2] = {55, 25};  
       Mat myDataMat(2, 1, CV_32FC1, myData);  
       double r = boost.predict( myDataMat );  

       cout<<endl<<"result:  "<<(char)r<<endl;  

       return 0;  
    }
   ```

- 装袋算法Bagging

- 多专家模型
   - 合并多神经网络的结果
   
- 最大熵模型

- EM
   - [例子](https://www.tuicool.com/articles/Av6NVzy)
   

- 隐马尔可夫模型HMM

- 条件随机场CRF

   
   
   
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
   
