//*****************************************************************
// �ļ��� :						RHOG.h
// �汾	 :						1.0
// Ŀ�ļ���Ҫ���� :				����RHOGĿ������㷨
// �������� :					2016.4.10
// �޸����� :					
// ���� :						����
// �޸��� :						
// ��ϵ��ʽ :					fiki@seu.edu.cn
// ע��							
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

#include "Markup.h"		//�������xml�ļ�

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
#define ANG 18		//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

#define CellNumb 7		//ÿ���Ƕȷ���ֶ��ٸ�cell

#define CellPerBlob 3	//ÿ����cellһ��blob

#define BIN 8		//ÿ��cell�ֶ��ٸ��ݶȷ���

#define ImageWidth 128		//ͼ���С

#define SearchStep 12		//��������

//#define PI 3.1415926535897f		//pi
#define PI2 6.2831853071794f	//2 * pi

#define Adaboost 1			//1:
#define Rtree 2
#define SVMC 3

#define Sym false			//�Ƿ���Ҫ�Գ�����
#define RSC false			//�Ƿ��Cell���нǶȼ��ƽ��
#define DSC false			//�Ƿ��Cell���нǶȼ��ƽ��

#define FilterSize 7		//Ĭ����ֵ�˲���ģ���СΪ7*7

#define MatchTime 6			//������ʱģ���ص��Ĵ���


//error define
#define NotEnoughSampleForTrainPos  -50001			//ѵ������������������
#define NotEnoughSampleForTrainNeg	-50002			//����ͼ���쳣��δ�򿪻��߳ߴ粻�ԣ�
#define InputImageError				-50003			//ѵ������������������
#define ClassifierNotExist			-50004			//�����������ڣ��޷����棨����ʱ������
#define ClassifierFileNotExist		-50005			//�������ļ������ڣ��޷����루loadʱ������
#define ClassifierSaveFailed		-50006			//����������ʧ��
#define WrongClassifierType			-50007			//����ķ���������
#define ParIllegal					-50008			//�����Ƿ���ȡֵ������
#define InputImageNotSupport		-50009			//��֧�ַ�8λͼ��

/*****************************************************************
Structures
*****************************************************************/
/* NONE */


/*****************************************************************
					RHOGPar��
			�����࣬���ڶ�����д�����
*****************************************************************/
class RHOGPar
{
public:
	RHOGPar()
	{
		m_nANG = ANG;		//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

		m_nCellNumb = CellNumb;			//ÿ���Ƕȷ���ֶ��ٸ�cell

		m_nCellPerBlob = CellPerBlob;		//ÿ����cellһ��blob

		m_nBIN = BIN;			//ÿ��cell�ֶ��ٸ��ݶȷ���

		m_nImageWidth = ImageWidth;			//ͼ���С

		m_nSearchStep = SearchStep;			//��������

		m_nClassType = SVMC;		//������

		m_bSym = Sym;				//�Ƿ���Ҫ�Գ�����
		m_bRSC = RSC;				//�Ƿ��Cell���нǶȼ��ƽ��
		m_bDSC = DSC;				//�Ƿ��Cell���нǶȼ��ƽ��

		m_nFilterSize = FilterSize;	//Ĭ����ֵ�˲���ģ���С

		m_nMatchTime = MatchTime;			//������ʱģ���ص��Ĵ���

		m_bSavePosPatch = false;		//�Ƿ������������ȡ��������ͼ
		m_bSaveNegPatch = false;		//�Ƿ������������ȡ�ķ�����ͼ

		m_sPosPatchPath = "D:\\Test\\Pos\\";	//���������ȡ��������ͼ����·��
		m_sNegPatchPath = "D:\\Test\\Neg\\";	//���������ȡ��������ͼ����·��
	}

	~RHOGPar(){};

public:
	int m_nANG;		//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

	int m_nCellNumb;		//ÿ���Ƕȷ���ֶ��ٸ�cell

	int m_nCellPerBlob;	//ÿ����cellһ��blob

	int m_nBIN;		//ÿ��cell�ֶ��ٸ��ݶȷ���

	int m_nImageWidth;		//ͼ���С

	int m_nSearchStep;		//��������

	int m_nClassType;	//������

	bool m_bSym;			//�Ƿ���Ҫ�Գ�����
	bool m_bRSC;			//�Ƿ��Cell���нǶȼ��ƽ��
	bool m_bDSC;			//�Ƿ��Cell���нǶȼ��ƽ��

	int m_nFilterSize;		//ͼ��Ԥ������ֵ�˲���ģ���С

	int m_nMatchTime;		//������ʱģ���ص��Ĵ���

	bool m_bSavePosPatch;		//�Ƿ������������ȡ��������ͼ
	bool m_bSaveNegPatch;		//�Ƿ������������ȡ�ķ�����ͼ

	CString m_sPosPatchPath;	//���������ȡ��������ͼ����·��
	CString m_sNegPatchPath;	//���������ȡ��������ͼ����·��
};


/*****************************************************************
					RHOG��
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
		string t_sPosPath - ��������·��
		string t_sNegPath - ��������·��
	Return Value:
		int - <0 �������
			  1  ��ȷ����
	Description:	ѵ����������ѵ�������SaveClassifier��������
	*****************************************************************/
	int Training( string t_sPosPath, string t_sNegPath );		

	/*****************************************************************
	Name:			Test
	Inputs:
		string t_sPosPath - ������·��
		string t_sNegPath - ������·��
		float &t_fPosRate - ��������ȷ��
		float &t_fNegRate - ��������ȷ��
	Return Value:
		int - <0 �������
			  1  ��ȷ����
	Description:	���Է�������ȷ��
	*****************************************************************/
	int Test( string t_sPosPath, string t_sNegPath, float &t_fPosRate, float &t_fNegRate );	


	/*****************************************************************
	Name:			SearchTarget
	Inputs:
		ListImage *SrcImg - ��������ͼ��
		iRect *& t_pRect - Ŀ�����
		float t_fStartRat - ��ʼƥ��ʱ�����ű�������ʼ��Ϊ0.5
		float t_fStepSize - ÿ������ʱͼ�����ŵı����ı�������ʼ��Ϊ0.1
		int t_nResizeStep - ƥ��ʱͼ�����ŵĴ���
		int t_nMatchTime - ƥ��ʱ���ص���ֵ
		int t_nSearchStep - ƥ��ʱ�������Ļ�������
	Return Value:
		int - >0 Ŀ������
			  <0 ��Ӧ�������
	Description:	��ͼ���м���Ŀ��
	*****************************************************************/
	int SearchTarget( ListImage *SrcImg, iRect *& t_pRect, 
					  float t_fStartRate = 0.5f, float t_fStepSize = 0.1f, int t_nResizeStep = 5,
					  int t_nMatchTime = -1,
					  int t_nSearchStep = -1 );	

	
	/*****************************************************************
	Name:			GetPar
	Inputs:
		RHOGPar &t_Par - ���ض����Ĳ���
	Return Value:
		none
	Description:	������ǰϵͳ����
	*****************************************************************/
	void GetPar( RHOGPar &t_Par );		


	/*****************************************************************
	Name:			SetPar
	Inputs:
		RHOGPar t_Par - �����õķ�������
	Return Value:
		1 - ����ɹ�
		<0 - �������
	Description:	���ò�������յ�ǰ���ٿռ䣬�����ݲ������ٿռ�
	*****************************************************************/
	int SetPar( RHOGPar t_Par );		


	/*****************************************************************
	Name:			SaveClassifier
	Inputs:
		string t_sClassFilePath - ����ķ�����·��
	Return Value:
		1 - ����ɹ�
		<0 - �������
	Description:	���浱ǰ���������ļ�
	*****************************************************************/
	int SaveClassifier( string t_sClassFilePath );


	/*****************************************************************
	Name:			LoadClassifier
	Inputs:
		string t_sClassFilePath - ��ȡ�ķ�����·��
	Return Value:
		1 - ��ȡ�ɹ�
		<0 - ��ȡ����
	Description:	��ȡ�ķ�����������ݶ�ȡ�ķ��������������¿��ٿռ�
	*****************************************************************/
	int LoadClassifier( string t_sClassFilePath );		

		
	/*****************************************************************
	Name:			SetSavePatchImage
	Inputs:
		bool t_bPosSave - �Ƿ񱣴���������ͼ
		bool t_bNegSave - �Ƿ񱣴渺������ͼ
		string t_sPosPath - ��������ͼ����·��
		string t_sNegPath - ��������ͼ����·��
	Return Value:
		none
	Description:	�����Ƿ񱣴���/�����Էָ���ͼ�����ڷ����Ͷ���ѵ����
	*****************************************************************/
	void SetSavePatchImage( bool t_bPosSave, bool t_bNegSave, string t_sPosPath = "", string t_sNegPath = "" );


private:
	/*****************************************************************
	Name:			SearchTargetPerImg
	Inputs:
		Mat t_Image - ������ͼ��
		float t_fAugRate - ͼ��Ŵ����
		vector <CvRect> &t_vTarget - ���ص�Ŀ��������
	Return Value:
		int - >0 Ŀ������
			  <0 ��Ӧ�������
	Description:	�ڵ���ͼ���м���Ŀ��
	*****************************************************************/
	int SearchTargetPerImg( Mat t_Image, float t_fAugRate, vector <CvRect> &t_vTarget );	//��ͼ���м�����Ŀ��

private:
	int CountFeatureFromImg( Mat t_Image, float *t_pFeatures );						//������ͼ���������

	int CountFeature( int t_nX, int t_nY, int t_nWidth, int t_nHeight, float *t_pFeatures );//����ͼ����ĳһ���������

	void Clear( void );				//��յ�ǰ����
	void InitFeatures( void );		//��ʼ�����������ռ�

	void PreProcessImage( Mat t_Image, Mat & t_TarImage );		//Ԥ����ͼ��
	void CountGrad( Mat t_pImage );									//�Ը���ͼ������ݶ���Ϣ
	void CountCell( int t_nX, int t_nY, int t_nWidth, int t_nHeight );		//����Cell
	void SmoothCell( void );		//��Cell����ƽ��(m_pCellFeatures)
	void RSmoothCell( void );		//��Cell���нǶ�ƽ��(m_pCellFeatures)
	void DSmoothCell( void );		//��Cell���нǶ�ƽ��(m_pCellFeatures)
	
	void Countm_nBIN( void );		//����m_nBIN
	void Normalm_nBIN( void );		//��һ��m_nBIN
	void CountSym( void );			//����Գ�������

	int RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime );	//���¹鲢Ŀ����У����þ��෨

	int GetImageList( string t_sPath, vector <string> &t_vFileName );						//��ȡԭʼͼ���б�(����)

	int cvtList2Mat( ListImage *SrcImg, Mat &t_pImage );								//��listimageͼ��תΪMatͼ��

/*****************************************************************
Variables
*****************************************************************/
private:
	int m_nANG;			//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

	int m_nCellNumb;	//ÿ���Ƕȷ���ֶ��ٸ�cell
	int m_nBlobNumb;	//( m_nCellNumb - m_nCellPerBlob + 1 )		//ÿ���Ƕȷ�����ٸ�m_nBlobNumb

	int m_nCellPerBlob;	//ÿ����cellһ��blob

	int m_nBIN;			//ÿ��cell�ֶ��ٸ��ݶȷ���

	int m_nImageWidth;	//ͼ���С

	int m_nSearchStep;	//��������

	int m_nClassType;	//������

	bool m_bSym;			//�Ƿ���Ҫ�Գ�����
	bool m_bRSC;			//�Ƿ��Cell���нǶȼ��ƽ��
	bool m_bDSC;			//�Ƿ��Cell���нǶȼ��ƽ��

	bool m_bSavePosPatch;	//�Ƿ������������ȡ��������ͼ
	bool m_bSaveNegPatch;	//�Ƿ������������ȡ�ķ�����ͼ

	string m_sPosPatchPath;//���������ȡ��������ͼ����·��
	string m_sNegPatchPath;//���������ȡ��������ͼ����·��

private:
	float *m_pfFeature;		//���ڴ������������
	float *m_pCellFeatures;	//cell����

	int m_nFeatureNumber;	//��������

	int m_nCellWidth;		//ÿ��cell�Ŀ��
	int m_nANGWidth;		//ÿ�������������

	Mat m_Image;			//������ͼ����ģ����ͬ��С
	Mat m_MagImage;		//ģͼ����ģ����ͬ��С
	Mat m_ANGImage;		//���ͼ����ģ����ͬ��С

	Mat m_TestImage;		//����ͼ��

	float * m_fNormalMat;	//����Ԥ�ȼ���ģ����ÿһ�㵽�е�ĽǶ�
	float * m_fMagMat;		//����Ԥ�ȼ���ģ����ÿһ�㵽�е�ľ���
	int * m_nANGle;			//����ȷ�������Ƕ�
	int * m_nMag;           //����ȷ���������
	int * mask;              //�����ж��Ƿ���Բ��
	int * histo_mask;         //����ȷ��binֵ

	int m_nFilterSize;		//ͼ��Ԥ������ֵ�˲���ģ���С

	int m_nMatchTime;		//������ʱģ���ص��Ĵ���

	void * m_pClassifier;	//������ָ��
};


//Ŀ��������
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


