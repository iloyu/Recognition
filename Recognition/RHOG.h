//*****************************************************************
// 文件名 :						RHOG.h
// 版本	 :						1.0
// 目的及主要功能 :				用于RHOG目标检索算法
// 创建日期 :					2016.4.10
// 修改日期 :					
// 作者 :						王征
// 修改者 :						
// 联系方式 :					fiki@seu.edu.cn
// 注：							
//*****************************************************************/
#pragma once


/*****************************************************************
Library Files Included
*****************************************************************/
#include <opencv2/opencv.hpp>
using namespace cv;
using cv::Mat;

#include <vector>
#include "ListImage.h"
#include<iostream>
#include<fstream>
#include<string>

#include <stdlib.h>
#include <stdio.h>

#include "Markup.h"		//用于输出xml文件

#ifdef linux
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif

using namespace std;

//extern "C" void MedianFilter(unsigned char * src,unsigned char *dst,int  width,int height);
//extern "C" void countFeatures(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,uchar *p_ANG,uchar *p_Mag,int Imagewidth,int ImageHeight);
extern "C" void countFeaturesfloat(float *out,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask,int *histo_mask,int off_x,int off_y);

/*****************************************************************
Defines   
*****************************************************************/
#define ANG 18		//分多少个角度方向，必须是偶数

#define CellNumb 7		//每个角度方向分多少个cell

#define CellPerBlob 3	//每三个cell一个blob

#define BIN 8		//每个cell分多少个梯度方向

#define ImageWidth 128		//图像大小

#define SearchStep 12		//搜索步长

//#define PI 3.1415926535897f		//pi
#define PI2 6.2831853071794f	//2 * pi

#define Adaboost 1			//1:
#define Rtree 2
#define SVMC 3

#define Sym false			//是否需要对称特征
#define RSC false			//是否对Cell进行角度间的平滑
#define DSC false			//是否对Cell进行角度间的平滑

#define FilterSize 7		//默认中值滤波的模板大小为7*7

#define MatchTime 6			//计算结果时模板重叠的次数


//error define
#define NotEnoughSampleForTrainPos  -50001			//训练分类器的样本不足
#define NotEnoughSampleForTrainNeg	-50002			//输入图像异常（未打开或者尺寸不对）
#define InputImageError				-50003			//训练分类器的样本不足
#define ClassifierNotExist			-50004			//分类器不存在，无法保存（保存时弹出）
#define ClassifierFileNotExist		-50005			//分类器文件不存在，无法读入（load时弹出）
#define ClassifierSaveFailed		-50006			//分类器保存失败
#define WrongClassifierType			-50007			//错误的分类器类型
#define ParIllegal					-50008			//参数非法（取值不合理）
#define InputImageNotSupport		-50009			//不支持非8位图像

/*****************************************************************
Structures
*****************************************************************/
/* NONE */


/*****************************************************************
					RHOGPar类
			参数类，用于读出、写入参数
*****************************************************************/
class RHOGPar
{
public:
	RHOGPar()
	{
		m_nANG = ANG;		//分多少个角度方向，必须是偶数

		m_nCellNumb = CellNumb;			//每个角度方向分多少个cell

		m_nCellPerBlob = CellPerBlob;		//每三个cell一个blob

		m_nBIN = BIN;			//每个cell分多少个梯度方向

		m_nImageWidth = ImageWidth;			//图像大小

		m_nSearchStep = SearchStep;			//搜索步长

		m_nClassType = SVMC;		//分类器

		m_bSym = Sym;				//是否需要对称特征
		m_bRSC = RSC;				//是否对Cell进行角度间的平滑
		m_bDSC = DSC;				//是否对Cell进行角度间的平滑

		m_nFilterSize = FilterSize;	//默认中值滤波器模板大小

		m_nMatchTime = MatchTime;			//计算结果时模板重叠的次数

		m_bSavePosPatch = false;		//是否保留检测中所截取的正例子图
		m_bSaveNegPatch = false;		//是否保留检测中所截取的反例子图

		m_sPosPatchPath = "D:\\Test\\Pos\\";	//检测中所截取的正例子图保存路径
		m_sNegPatchPath = "D:\\Test\\Neg\\";	//检测中所截取的正例子图保存路径
	}

	~RHOGPar(){};

public:
	int m_nANG;		//分多少个角度方向，必须是偶数

	int m_nCellNumb;		//每个角度方向分多少个cell

	int m_nCellPerBlob;	//每三个cell一个blob

	int m_nBIN;		//每个cell分多少个梯度方向

	int m_nImageWidth;		//图像大小

	int m_nSearchStep;		//搜索步长

	int m_nClassType;	//分类器

	bool m_bSym;			//是否需要对称特征
	bool m_bRSC;			//是否对Cell进行角度间的平滑
	bool m_bDSC;			//是否对Cell进行角度间的平滑

	int m_nFilterSize;		//图像预处理中值滤波器模板大小

	int m_nMatchTime;		//计算结果时模板重叠的次数

	bool m_bSavePosPatch;		//是否保留检测中所截取的正例子图
	bool m_bSaveNegPatch;		//是否保留检测中所截取的反例子图

	CString m_sPosPatchPath;	//检测中所截取的正例子图保存路径
	CString m_sNegPatchPath;	//检测中所截取的正例子图保存路径
};


/*****************************************************************
					RHOG类
*****************************************************************/ 
class RHOG
{
/*****************************************************************
Routine Definitions
*****************************************************************/
public:
	RHOG(void);
	~RHOG(void);

public:
	/*****************************************************************
	Name:			Training
	Inputs:
		string t_sPosPath - 正样本的路径
		string t_sNegPath - 负样本的路径
	Return Value:
		int - <0 错误代码
			  1  正确返回
	Description:	训练分类器，训练结果需SaveClassifier函数保存
	*****************************************************************/
	int Training( string t_sPosPath, string t_sNegPath );		

	/*****************************************************************
	Name:			Test
	Inputs:
		string t_sPosPath - 正样本路径
		string t_sNegPath - 负样本路径
		float &t_fPosRate - 正样本正确率
		float &t_fNegRate - 负样本正确率
	Return Value:
		int - <0 错误代码
			  1  正确返回
	Description:	测试分类器正确率
	*****************************************************************/
	int Test( string t_sPosPath, string t_sNegPath, float &t_fPosRate, float &t_fNegRate );	


	/*****************************************************************
	Name:			SearchTarget
	Inputs:
		ListImage *SrcImg - 待搜索的图像
		iRect *& t_pRect - 目标队列
		float t_fStartRat - 开始匹配时的缩放比例，初始化为0.5
		float t_fStepSize - 每次搜索时图像缩放的比例改变量，初始化为0.1
		int t_nResizeStep - 匹配时图像缩放的次数
		int t_nMatchTime - 匹配时的重叠阈值
		int t_nSearchStep - 匹配时滑动窗的滑动步长
	Return Value:
		int - >0 目标数量
			  <0 对应错误代码
	Description:	在图像中检索目标
	*****************************************************************/
	int SearchTarget( ListImage *SrcImg, iRect *& t_pRect, 
					  float t_fStartRate = 0.5f, float t_fStepSize = 0.1f, int t_nResizeStep = 5,
					  int t_nMatchTime = -1,
					  int t_nSearchStep = -1 );	

	
	/*****************************************************************
	Name:			GetPar
	Inputs:
		RHOGPar &t_Par - 返回读出的参数
	Return Value:
		none
	Description:	读出当前系统参数
	*****************************************************************/
	void GetPar( RHOGPar &t_Par );		


	/*****************************************************************
	Name:			SetPar
	Inputs:
		RHOGPar t_Par - 待设置的分析参数
	Return Value:
		1 - 保存成功
		<0 - 保存错误
	Description:	设置参数，清空当前开辟空间，并根据参数开辟空间
	*****************************************************************/
	int SetPar( RHOGPar t_Par );		


	/*****************************************************************
	Name:			SaveClassifier
	Inputs:
		string t_sClassFilePath - 保存的分类器路径
	Return Value:
		1 - 保存成功
		<0 - 保存错误
	Description:	保存当前分类器至文件
	*****************************************************************/
	int SaveClassifier( string t_sClassFilePath );


	/*****************************************************************
	Name:			LoadClassifier
	Inputs:
		string t_sClassFilePath - 读取的分类器路径
	Return Value:
		1 - 读取成功
		<0 - 读取错误
	Description:	读取的分类器，会根据读取的分类器参数，重新开辟空间
	*****************************************************************/
	int LoadClassifier( string t_sClassFilePath );		

		
	/*****************************************************************
	Name:			SetSavePatchImage
	Inputs:
		bool t_bPosSave - 是否保存正样本子图
		bool t_bNegSave - 是否保存负样本子图
		string t_sPosPath - 正样本子图保存路径
		string t_sNegPath - 负样本子图保存路径
	Return Value:
		none
	Description:	设置是否保存正/负测试分割子图（用于分析和二次训练）
	*****************************************************************/
	void SetSavePatchImage( bool t_bPosSave, bool t_bNegSave, string t_sPosPath = "", string t_sNegPath = "" );


private:
	/*****************************************************************
	Name:			SearchTargetPerImg
	Inputs:
		Mat t_Image - 待分析图像
		float t_fAugRate - 图像放大比例
		vector <CvRect> &t_vTarget - 返回的目标结果队列
	Return Value:
		int - >0 目标数量
			  <0 对应错误代码
	Description:	在单张图像中检索目标
	*****************************************************************/
	int SearchTargetPerImg( Mat t_Image, float t_fAugRate, vector <CvRect> &t_vTarget );	//从图像中检索出目标

private:
	int CountFeatureFromImg( Mat t_Image, float *t_pFeatures );						//以输入图像计算特征

	int CountFeature( int t_nX, int t_nY, int t_nWidth, int t_nHeight, float *t_pFeatures );//计算图像中某一区域的特征

	void Clear( void );				//清空当前数据
	void InitFeatures( void );		//初始化开辟特征空间

	void PreProcessImage( Mat t_Image, Mat & t_TarImage );		//预处理图像
	void CountGrad( Mat t_pImage );									//对给定图像计算梯度信息
	void CountCell( int t_nX, int t_nY, int t_nWidth, int t_nHeight );		//计算Cell
	void SmoothCell( void );		//对Cell进行平滑(m_pCellFeatures)
	void RSmoothCell( void );		//对Cell进行角度平滑(m_pCellFeatures)
	void DSmoothCell( void );		//对Cell进行角度平滑(m_pCellFeatures)
	
	void Countm_nBIN( void );		//计算m_nBIN
	void Normalm_nBIN( void );		//归一化m_nBIN
	void CountSym( void );			//计算对称性特征

	int RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime );	//重新归并目标队列，采用聚类法

	int GetImageList( string t_sPath, vector <string> &t_vFileName );						//获取原始图像列表(辅助)

	int cvtList2Mat( ListImage *SrcImg, Mat &t_pImage );								//将listimage图像转为Mat图像

/*****************************************************************
Variables
*****************************************************************/
private:
	int m_nANG;			//分多少个角度方向，必须是偶数

	int m_nCellNumb;	//每个角度方向分多少个cell
	int m_nBlobNumb;	//( m_nCellNumb - m_nCellPerBlob + 1 )		//每个角度方向多少个m_nBlobNumb

	int m_nCellPerBlob;	//每三个cell一个blob

	int m_nBIN;			//每个cell分多少个梯度方向

	int m_nImageWidth;	//图像大小

	int m_nSearchStep;	//搜索步长

	int m_nClassType;	//分类器

	bool m_bSym;			//是否需要对称特征
	bool m_bRSC;			//是否对Cell进行角度间的平滑
	bool m_bDSC;			//是否对Cell进行角度间的平滑

	bool m_bSavePosPatch;	//是否保留检测中所截取的正例子图
	bool m_bSaveNegPatch;	//是否保留检测中所截取的反例子图

	string m_sPosPatchPath;//检测中所截取的正例子图保存路径
	string m_sNegPatchPath;//检测中所截取的正例子图保存路径

private:
	float *m_pfFeature;		//用于存放特征的向量
	float *m_pCellFeatures;	//cell特征

	int m_nFeatureNumber;	//特征数量

	int m_nCellWidth;		//每个cell的宽度
	int m_nANGWidth;		//每个方向的特征数

	Mat m_Image;			//待分析图像，与模板相同大小
	Mat m_MagImage;		//模图像，与模板相同大小
	Mat m_ANGImage;		//相角图像，与模板相同大小

	Mat m_TestImage;		//测试图象

	float * m_fNormalMat;	//用于预先计算模板中每一点到中点的角度
	float * m_fMagMat;		//用于预先计算模板中每一点到中点的距离
	int * m_nANGle;			//用于确定所属角度
	int * m_nMag;           //用于确定扇区编号
	int * mask;              //用于判断是否在圆内
	int * histo_mask;         //用于确定bin值

	int m_nFilterSize;		//图像预处理中值滤波器模板大小

	int m_nMatchTime;		//计算结果时模板重叠的次数

	void * m_pClassifier;	//分类器指针
};


//目标区域类
class TargetArea
{
public:
	TargetArea()
	{
		m_nDupeNumber = 0;
		m_nCenterX = 0;
		m_nCenterY = 0;
		m_nWidth = 0;
		m_nHeight = 0;
	}

	~TargetArea(){};

public:
	int m_nDupeNumber;
	int m_nCenterX;
	int m_nCenterY;

	int m_nWidth;
	int m_nHeight;
};


