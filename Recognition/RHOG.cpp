//*****************************************************************
// �ļ��� :						HOGDetector.cpp
// �汾	 :						1.0
// Ŀ�ļ���Ҫ���� :				����CHOG�������㼰ͼ��ƥ��
// �������� :					2016.1.28
// �޸����� :					
// ���� :						����
// �޸��� :						
// ��ϵ��ʽ :					fiki@seu.edu.cn
//*****************************************************************/

///////////////////////////////////////////////////////
////////////////////////include////////////////////////
///////////////////////////////////////////////////////
#include "StdAfx.h"
#include "RHOG.h"
#include "Markup.h"		//�������xml�ļ�
#include "time.h"

#include <windows.h> 
/*****************************************************************
Defines
*****************************************************************/
//none


/*****************************************************************
Global Variables
*****************************************************************/
int g_nPosImageNumber = 1;
int g_nNegImageNumber = 1;


/*****************************************************************
Global Function
*****************************************************************/


/*****************************************************************
							RHOG�ඨ��
*****************************************************************/ 
/*****************************************************************
Name:			RHOG
Inputs:
	none.
Return Value:
	none.
Description:	Ĭ�Ϲ��캯��
*****************************************************************/
RHOG::RHOG(void)
{
	//��ʼ������
	m_nANG = ANG;				//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

	m_nCellNumb = CellNumb;		//ÿ���Ƕȷ���ֶ��ٸ�cell

	m_nCellPerBlob = CellPerBlob;		//ÿ����cellһ��blob

	m_nBlobNumb = m_nCellNumb - m_nCellPerBlob;		//ÿ���Ƕȷ�����ٸ�m_nBlobNumb

	m_nBIN = 10;				//ÿ��cell�ֶ��ٸ��ݶȷ���

	m_nImageWidth = ImageWidth;	//ͼ���С

	m_nSearchStep = SearchStep;	//��������

	m_nClassType = SVMC;		//������

	m_bSym = Sym;				//�Ƿ���Ҫ�Գ�����
	m_bRSC = RSC;				//�Ƿ��Cell���нǶȼ��ƽ��
	m_bDSC = DSC;				//�Ƿ��Cell���нǶȼ��ƽ��

	m_nFilterSize = FilterSize;	//Ĭ����ֵ�˲���ģ���С

	m_nMatchTime = MatchTime;	//������ʱģ���ص��Ĵ���

	m_bSavePosPatch = false;	//�Ƿ������������ȡ��������ͼ
	m_bSaveNegPatch = false;	//�Ƿ������������ȡ�ķ�����ͼ

	m_sPosPatchPath = "E:\\Train\\Pos\\";	//���������ȡ��������ͼ����·��
	m_sNegPatchPath = "E:\\Train\\Neg\\";	//���������ȡ��������ͼ����·��


	//��ʼ���ռ�ָ�뼰��ز���
	m_pfFeature = NULL;			//��������
	m_pCellFeatures = NULL;		//Cell��������

	m_pClassifier=NULL;			//��������ʼ��Ϊ��

	if ( m_bSym )
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN + m_nANG / 2 * m_nCellNumb;		//��������
	}
	else
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//�������� 18*4*3*10
	}

	m_nANGWidth = m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//ÿ������������� 4*3*10


	//���ٳ�ʼ���ռ�
	m_fNormalMat = new float[m_nImageWidth * m_nImageWidth];
	m_fMagMat = new float[m_nImageWidth * m_nImageWidth];
	m_nANGle = new int[m_nImageWidth * m_nImageWidth];
	m_nMag=new int[m_nImageWidth * m_nImageWidth];

	//Ԥ��ģ�����
	float t_fCenterX;
	t_fCenterX = m_nImageWidth / 2.0f;		//�����е�����
	float t_fCenterY;
	t_fCenterY = m_nImageWidth / 2.0f;
	int i, j;
	for ( j = 0; j < m_nImageWidth; ++j )
	{
		for ( i = 0; i < m_nImageWidth; ++i )
		{
			float t_fDeltaX;
			float t_fDeltaY;
			t_fDeltaX = i - t_fCenterX;
			t_fDeltaY = t_fCenterY - j;

			m_fNormalMat[j * m_nImageWidth + i] = atan2( 0, 1.0f );	


			m_fNormalMat[j * m_nImageWidth + i] = atan2( ( t_fDeltaY ), ( t_fDeltaX ) );		// + PI2;
			m_fMagMat[j * m_nImageWidth + i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );

			if ( m_fNormalMat[j * m_nImageWidth + i] < 0 )
			{
				m_fNormalMat[j * m_nImageWidth + i] = m_fNormalMat[j * m_nImageWidth + i] + PI2;
			}

			m_nANGle[j * m_nImageWidth + i] = (int)( m_fNormalMat[j * m_nImageWidth + i] * m_nANG / PI2 );
			m_nMag[j*m_nImageWidth+i]=(int)(m_fMagMat[j*m_nImageWidth+i]/10);
		}
	}
}//RHOG


/*****************************************************************
Name:			~HOGDetector
Inputs:
	none.
Return Value:
	none.
Description:	��������
*****************************************************************/
RHOG::~RHOG(void)
{
	Clear();	//�ͷſռ�

	if ( m_fNormalMat != NULL )
	{
		delete [] m_fNormalMat;
		m_fNormalMat = NULL;
	}

	if ( m_fMagMat != NULL )
	{
		delete [] m_fMagMat;
		m_fMagMat = NULL;
	}

	if ( m_nANGle != NULL )
	{
		delete [] m_nANGle;
		m_nANGle = NULL;
	}

	//�رշ�����
	if (m_pClassifier!=NULL)
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;

		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}
}//~RHOG(void)


/*****************************************************************
Name:			GetPar
Inputs:
	RHOGPar &t_Par - ���ض����Ĳ���
Return Value:
	none
Description:	������ǰϵͳ����
*****************************************************************/
void RHOG::GetPar( RHOGPar &t_Par )
{
	t_Par.m_nANG = m_nANG;		//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

	t_Par.m_nCellNumb = m_nCellNumb;		//ÿ���Ƕȷ���ֶ��ٸ�cell

	t_Par.m_nCellPerBlob = m_nCellPerBlob;	//ÿ����cellһ��blob

	t_Par.m_nBIN = m_nBIN;		//ÿ��cell�ֶ��ٸ��ݶȷ���

	t_Par.m_nImageWidth = m_nImageWidth;//ͼ���С

	t_Par.m_nClassType = m_nClassType;	//������

	t_Par.m_bSym = m_bSym;		//�Ƿ���Ҫ�Գ�����
	t_Par.m_bRSC = m_bRSC;		//�Ƿ��Cell���нǶȼ��ƽ��
	t_Par.m_bDSC = m_bDSC;		//�Ƿ��Cell���нǶȼ��ƽ��

	t_Par.m_nFilterSize = m_nFilterSize;	//ͼ��Ԥ������ֵ�˲���ģ���С������������
}//GetPar


/*****************************************************************
Name:			SetPar
Inputs:
	RHOGPar t_Par - �����õķ�������
Return Value:
	1 - ����ɹ�
	<0 - �������
Description:	���ò�������յ�ǰ���ٿռ䣬�����ݲ������ٿռ�
*****************************************************************/
int RHOG::SetPar( RHOGPar t_Par )
{
	//�жϺϷ���
	if ( t_Par.m_nANG < 4 
		|| t_Par.m_nCellNumb < 5
		|| t_Par.m_nCellNumb > 20
		|| t_Par.m_nCellPerBlob >=  t_Par.m_nCellNumb - 2
		|| t_Par.m_nBIN < 5
		|| t_Par.m_nBIN > 18
		|| t_Par.m_nImageWidth > 256
		|| t_Par.m_nImageWidth < 32
		|| t_Par.m_nClassType > SVMC
		|| t_Par.m_nClassType < Adaboost
		|| t_Par.m_nFilterSize % 2 != 1
		|| t_Par.m_nFilterSize > 21 )
	{
		return ParIllegal;
	}

	//���Ʋ���
	m_nANG = t_Par.m_nANG;		//�ֶ��ٸ��Ƕȷ��򣬱�����ż��

	m_nCellNumb = t_Par.m_nCellNumb;		//ÿ���Ƕȷ���ֶ��ٸ�cell

	m_nCellPerBlob = t_Par.m_nCellPerBlob;	//ÿ����cellһ��blob

	m_nBlobNumb = m_nCellNumb - m_nCellPerBlob;		//ÿ���Ƕȷ�����ٸ�m_nBlobNumb

	m_nBIN = t_Par.m_nBIN;		//ÿ��cell�ֶ��ٸ��ݶȷ���

	m_nImageWidth = t_Par.m_nImageWidth;//ͼ���С

	m_nClassType = t_Par.m_nClassType;	//������

	m_bSym = t_Par.m_bSym;		//�Ƿ���Ҫ�Գ�����
	m_bRSC = t_Par.m_bRSC;		//�Ƿ��Cell���нǶȼ��ƽ��
	m_bDSC = t_Par.m_bDSC;		//�Ƿ��Cell���нǶȼ��ƽ��

	m_nFilterSize = t_Par.m_nFilterSize;//ͼ��Ԥ������ֵ�˲���ģ���С


	//�������пռ�
	Clear();

	if ( m_fNormalMat != NULL )
	{
		delete [] m_fNormalMat;
		m_fNormalMat = NULL;
	}

	if ( m_fMagMat != NULL )
	{
		delete [] m_fMagMat;
		m_fMagMat = NULL;
	}

	if ( m_nANGle != NULL )
	{
		delete [] m_nANGle;
		m_nANGle = NULL;
	}

	//�رշ�����
	if (m_pClassifier!=NULL)
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;

		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}

	//���ٿռ�
	if ( m_bSym )
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN + m_nANG / 2 * m_nCellNumb;		//��������
	}
	else
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//��������
	}

	m_nANGWidth = m_nBlobNumb * m_nCellPerBlob * m_nBIN;					//ÿ�������������

	m_fNormalMat = new float[m_nImageWidth * m_nImageWidth];	
	m_fMagMat = new float[m_nImageWidth * m_nImageWidth];
	m_nANGle = new int[m_nImageWidth * m_nImageWidth];

	//Ԥ��ģ�����
	float t_fCenterX;
	t_fCenterX = m_nImageWidth / 2.0f;		//�����е�����
	float t_fCenterY;
	t_fCenterY = m_nImageWidth / 2.0f;
	int i, j;
	for ( j = 0; j < m_nImageWidth; ++j )
	{
		for ( i = 0; i < m_nImageWidth; ++i )
		{
			float t_fDeltaX;
			float t_fDeltaY;
			t_fDeltaX = i - t_fCenterX;
			t_fDeltaY = t_fCenterY - j;

			m_fNormalMat[j * m_nImageWidth + i] = atan2( 0, 1.0f );	


			m_fNormalMat[j * m_nImageWidth + i] = atan2( ( t_fDeltaY ), ( t_fDeltaX ) );		// + PI2;
			m_fMagMat[j * m_nImageWidth + i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );

			if ( m_fNormalMat[j * m_nImageWidth + i] < 0 )
			{
				m_fNormalMat[j * m_nImageWidth + i] = m_fNormalMat[j * m_nImageWidth + i] + PI2;
			}

			m_nANGle[j * m_nImageWidth + i] = (int)( m_fNormalMat[j * m_nImageWidth + i] * m_nANG / PI2 );
		}
	}

	return 1;
}//SetPar


/*****************************************************************
Name:			SaveClassifier
Inputs:
	string t_sClassFilePath - ����ķ�����·�����ļ���
Return Value:
	1 - ����ɹ�
	<0 - �������
Description:	���浱ǰ���������ļ����ļ���Ӧ��xml��β��
*****************************************************************/
int RHOG::SaveClassifier( string t_sClassFilePath )
{
	//д�������
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}

	switch ( m_nClassType )
	{
	case Adaboost:	((CvBoost *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	case Rtree:		((CvRTrees *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	case SVMC:		((CvSVM *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	}

	locale loc = locale::global(locale(""));
		
	ifstream ifs( t_sClassFilePath.c_str());
	if( !ifs )
	{
		locale::global(locale("C"));//��ԭȫ�������趨

		return ClassifierSaveFailed;
	}

	//д�������Ϣ
	string t_sKey;
	string t_sOut;

	CMarkup t_XML;  
	t_XML.Load( t_sClassFilePath.c_str() );   

	t_XML.AddElem( "ClassifierPar" );

	t_XML.IntoElem();

	t_XML.AddElem( "Par" );

	char p[32]={0,};//��ʼ����ʱ�ַ���

	t_sKey = "ANG";
	sprintf( p,"%d", m_nANG );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "CellNumb";
	sprintf( p,"%d", m_nCellNumb );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "CellPerBlob";
	sprintf( p,"%d", m_nCellPerBlob );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "BIN";
	sprintf( p,"%d", m_nBIN );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "ImageWidth";
	sprintf( p,"%d", m_nImageWidth );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "ClassType";
	sprintf( p,"%d", m_nClassType );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "Sym";
	sprintf( p,"%d", m_bSym );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "RSC";
	sprintf( p,"%d", m_bRSC );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "DSC";
	sprintf( p,"%d", m_bDSC );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_sKey = "FilterSize";
	sprintf( p,"%d", m_nFilterSize );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );

	t_XML.OutOfElem();

	t_XML.Save( t_sClassFilePath.c_str() );

	locale::global(locale("C"));//��ԭȫ�������趨

	return 1;
}//SaveClassifier


/*****************************************************************
Name:			LoadClassifier
Inputs:
	string t_sClassFilePath - ��ȡ�ķ�����·��
Return Value:
	1 - ��ȡ�ɹ�
	<0 - ��ȡ����
Description:	��ȡ�ķ���������������󣬻���ݶ�ȡ��
				���������������¿��ٿռ�
*****************************************************************/
int RHOG::LoadClassifier( string t_sClassFilePath )
{
	locale loc = locale::global(locale(""));

	ifstream ifs( t_sClassFilePath.c_str() );
	if( !ifs )
	{
		locale::global(locale("C"));//��ԭȫ�������趨
		return ClassifierSaveFailed;
	}

	//��ȡXML��Ϣ
	RHOGPar t_RHOGPar;
	{
		bool t_bFind;
		CMarkup t_XML;  
		t_bFind = t_XML.Load( t_sClassFilePath );  

		if ( !t_bFind )
		{
			locale::global(locale("C"));//��ԭȫ�������趨

			return ClassifierFileNotExist;
		}

		string t_sKey;
		string t_sIn;

		t_XML.ResetMainPos();
		t_XML.FindElem( "ClassifierPar" );    //UserInfo
		while( t_XML.FindChildElem("Par") )
		{
			t_sIn = t_XML.GetChildAttrib( "ANG" );
			t_RHOGPar.m_nANG = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "CellNumb" );
			t_RHOGPar.m_nCellNumb = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "CellPerBlob" );
			t_RHOGPar.m_nCellPerBlob = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "BIN" );
			t_RHOGPar.m_nBIN = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "ImageWidth" );
			t_RHOGPar.m_nImageWidth = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "ClassType" );
			t_RHOGPar.m_nClassType = atoi( t_sIn.c_str() );

			t_sIn = t_XML.GetChildAttrib( "Sym" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bSym = false;
			}
			else
			{
				t_RHOGPar.m_bSym = true;
			}

			t_sIn = t_XML.GetChildAttrib( "RSC" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bRSC = false;
			}
			else
			{
				t_RHOGPar.m_bRSC = true;
			}

			t_sIn = t_XML.GetChildAttrib( "DSC" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bDSC = false;
			}
			else
			{
				t_RHOGPar.m_bDSC = true;
			}

			t_sIn = t_XML.GetChildAttrib( "FilterSize" );
			t_RHOGPar.m_nFilterSize = atoi( t_sIn.c_str() );
		}

		t_XML.FindElem( "ClassifierPar" );    //UserInfo
		t_XML.RemoveElem();
		t_XML.Save( t_sClassFilePath );
	}

	SetPar( t_RHOGPar );		//�������ò���

	//��ȡ������
	switch( m_nClassType )
	{
	case Adaboost:	{ 
		CvBoost * t_pBoost = new CvBoost;
		t_pBoost->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pBoost;
		break;
					}

	case Rtree:		{ 
		CvRTrees * t_pRTrees = new CvRTrees;
		t_pRTrees->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pRTrees;
		break;
					}

	case SVMC:		{ 
		CvSVM * t_pSVM = new CvSVM;
		t_pSVM->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pSVM;
		break;
					}
	default: return WrongClassifierType;
		break;
	}


	//����д����Ϣ
	//д�������Ϣ
	{
		string t_sKey;
		string t_sOut;

		CMarkup t_XML;  
		t_XML.Load( t_sClassFilePath );   

		t_XML.AddElem( "ClassifierPar" );

		t_XML.IntoElem();

		t_XML.AddElem( "Par" );

		char p[32]={0,};//��ʼ����ʱ�ַ���

		t_sKey = "ANG";
		sprintf( p,"%d", m_nANG );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "CellNumb";
		sprintf( p,"%d", m_nCellNumb );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "CellPerBlob";
		sprintf( p,"%d", m_nCellPerBlob );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "BIN";
		sprintf( p,"%d", m_nBIN );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "ImageWidth";
		sprintf( p,"%d", m_nImageWidth );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "ClassType";
		sprintf( p,"%d", m_nClassType );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "Sym";
		sprintf( p,"%d", m_bSym );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "RSC";
		sprintf( p,"%d", m_bRSC );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "DSC";
		sprintf( p,"%d", m_bDSC );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_sKey = "FilterSize";
		sprintf( p,"%d", m_nFilterSize );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );

		t_XML.OutOfElem();

		t_XML.Save( t_sClassFilePath );
	}

	locale::global(locale("C"));//��ԭȫ�������趨

	return 1;
}//LoadClassifier


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
void RHOG::SetSavePatchImage( bool t_bPosSave, bool t_bNegSave, string t_sPosPath, string t_sNegPath )
{
	m_bSavePosPatch = t_bPosSave;
	m_bSaveNegPatch = t_bNegSave;

	m_sPosPatchPath = t_sPosPath;
	m_sNegPatchPath = t_sNegPath;
}//SetSavePatchImage


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
int RHOG::Training( string t_sPosPath, string t_sNegPath )
{
	//��ȡ����ͼ���ļ�������
	int t_nPosNumber;			//����������
	vector <string> t_vPosFileList;	//���ڱ���ͼ�����ƶ���
	t_nPosNumber = GetImageList( t_sPosPath, t_vPosFileList );

	if ( t_nPosNumber < 20 )
	{
		return NotEnoughSampleForTrainPos;
	}

	//��ȡ����ͼ���ļ�������
	int t_nNegNumber;			//����������
	vector <string> t_vNegFileList;	//���ڱ���ͼ�����ƶ���
	t_nNegNumber = GetImageList( t_sNegPath, t_vNegFileList );
	if ( t_nNegNumber < 20 )
	{
		return NotEnoughSampleForTrainNeg;
	}

	//���������洢�ռ�ͱ�ǩ�洢�ռ�
	int t_nSampleNumber;		//����������
	t_nSampleNumber = t_nPosNumber + t_nNegNumber * ( m_nANG / 2 );

	Clear();

	InitFeatures();			//���������ռ�

	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( t_nSampleNumber, m_nFeatureNumber, CV_32FC1 );

	CvMat * t_ResponseMat;
	t_ResponseMat = cvCreateMat( t_nSampleNumber, 1, CV_32FC1 );

	//�ر��Ѵ򿪵ķ�����
	if ( m_pClassifier != NULL )
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;

		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}

	//����������������
	float * t_pFeature;
	t_pFeature = new float[m_nFeatureNumber];

	int i,j;
	for ( i = 0; i < t_nPosNumber; ++i )
	{
		//��������
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vPosFileList[i].c_str() );
		t_Img.ConvertToGreyImg();

		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );

		CountFeatureFromImg( t_Image, t_pFeature );	//��������

		float * t_pPos;
		t_pPos = &t_FeatureMat->data.fl[t_FeatureMat->width * i];
		memcpy( t_pPos, t_pFeature, m_nFeatureNumber * sizeof( float ) );
		t_ResponseMat->data.fl[i] = 1;
	}

	//���㷴����������
	for ( i = 0; i < t_nNegNumber; ++i )
	{
		//��������
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vNegFileList[i].c_str() );

		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );

		CountFeatureFromImg( t_Image, t_pFeature );

		//������תʹ��
		for ( j = 0; j < ( m_nANG / 2 ); j++ )
		{
			float * t_pPos;
			t_pPos = &t_FeatureMat->data.fl[t_FeatureMat->width * ( t_nPosNumber + ( i * ( m_nANG / 2 ) + j ) )];

			memcpy( t_pPos, 
					&t_pFeature[j * 2 * m_nANGWidth], 
					(m_nANG - j * 2) * m_nANGWidth * sizeof ( float ) );

			if ( j > 0 )
			{
				memcpy( &t_pPos[(m_nANG - j * 2) * m_nANGWidth], 
						t_pFeature, 
						j * 2 * m_nANGWidth * sizeof ( float ) );
			}

			if ( m_bSym )
			{
				memcpy( &t_pPos[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
						&t_pFeature[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
						m_nANG * m_nCellNumb * sizeof ( float ) / 2 );
			}

			t_ResponseMat->data.fl[t_nPosNumber + i * ( m_nANG / 2 ) + j] = 0;
		}
	}

	delete [] t_pFeature;


	//���÷���������
	if ( m_nClassType == Rtree )
	{
		float priors[] = {1,1};  // weights of each classification for classes
		CvRTParams params = CvRTParams(20, // max depth
										50, // min sample count
										0, // regression accuracy: N/A here
										false, // compute surrogate split, no missing data
										15, // max number of categories (use sub-optimal algorithm for larger numbers)
										priors, // the array of priors
										false,  // calculate variable importance
										50,       // number of variables randomly selected at node and used to find the best split(s).
										100,     // max number of trees in the forest
										0.01f,                // forest accuracy
										CV_TERMCRIT_ITER |    CV_TERMCRIT_EPS // termination cirteria
										);
		CvRTrees *t_pRTree;
		t_pRTree = new CvRTrees;

		//��ʼѵ��
		t_pRTree->train( t_FeatureMat, CV_ROW_SAMPLE, t_ResponseMat,
			0, 0, 0, 0, params );

		//������
		m_pClassifier = (void *)t_pRTree;
	}
	else if ( m_nClassType == Adaboost )
	{
		CvBoost *t_pBooster;
		t_pBooster = new CvBoost;
		CvBoostParams t_BoostParams( CvBoost::REAL, 150, 0, 1, false, 0 );

		//��ʼѵ��
		t_pBooster->train( t_FeatureMat, CV_ROW_SAMPLE, t_ResponseMat, 0, 0, 0, 0, t_BoostParams, false );

		//������
		m_pClassifier = (void *)t_pBooster;
	}
	else
	{
		//����֧���������Ĳ���  
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;//SVM���ͣ�ʹ��C֧��������
		params.kernel_type = CvSVM::LINEAR;//�˺������ͣ�����
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//��ֹ׼�����������������ﵽ���ֵʱ��ֹ

		//ѵ��SVM
		//����һ��SVM���ʵ��
		CvSVM *t_pSVM;
		t_pSVM = new CvSVM;
		//ѵ��ģ�ͣ�����Ϊ���������ݡ���Ӧ��XX��XX��������ǰ�����ù���
		t_pSVM->train( t_FeatureMat, t_ResponseMat, 0, 0, params );  

		//������
		m_pClassifier = (void *)t_pSVM;
	}

	cvReleaseMat( &t_FeatureMat );
	cvReleaseMat( &t_ResponseMat );

	return 1;
}//Training


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
Description:	���Է�������ȷ�ʣ�����ͼΪ����ͼƬ��
				��ͬ��������Ŀ���⡣
*****************************************************************/
int RHOG::Test( string t_sPosPath, string t_sNegPath, float &t_fPosRate, float &t_fNegRate )
{
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}

	//��ȡ����ͼ���ļ�������
	int t_nPosNumber;			//����������
	vector <string> t_vPosFileList;	//���ڱ���ͼ�����ƶ���

	t_nPosNumber = GetImageList( t_sPosPath, t_vPosFileList );


	//��ȡ����ͼ���ļ�������
	int t_nNegNumber;			//����������
	vector <string> t_vNegFileList;	//���ڱ���ͼ�����ƶ���
	t_nNegNumber = GetImageList( t_sNegPath, t_vNegFileList );

	//���Դ��룬���ڲ���Ч��
	float t_fSumPos;
	t_fSumPos = 0;

	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( m_nFeatureNumber, 1, CV_32FC1 );

	float * t_pFeature;
	t_pFeature = new float[m_nFeatureNumber];

	int i;
	for ( i = 0; i < t_nPosNumber; ++i )
	{
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vPosFileList[i].c_str() );

		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );

		CountFeatureFromImg( t_Image, t_pFeature );
		memcpy( t_FeatureMat->data.fl, t_pFeature, m_nFeatureNumber * sizeof ( float ) );

		float t_nS;
		if ( m_nClassType == Rtree )
		{
			t_nS = ((CvRTrees*)m_pClassifier)->predict( t_FeatureMat );
		}
		else if ( m_nClassType == Adaboost )
		{
			t_nS = ((CvBoost*)m_pClassifier)->predict( t_FeatureMat );
		}
		else
		{
			t_nS = ((CvSVM*)m_pClassifier)->predict( t_FeatureMat );
		}

		t_fSumPos += t_nS;
	}
	t_fPosRate = t_fSumPos / t_nPosNumber;		//��������ȷ��

	float t_fSumNeg;
	t_fSumNeg = (float)t_nNegNumber;
	for ( i = 0; i < t_nNegNumber; ++i )
	{
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vNegFileList[i].c_str() );

		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );

		CountFeatureFromImg( t_Image, t_pFeature );
		memcpy( t_FeatureMat->data.fl, t_pFeature, m_nFeatureNumber * sizeof ( float ) );

		float t_nS;
		if ( m_nClassType == Rtree )
		{
			t_nS = ((CvRTrees*)m_pClassifier)->predict( t_FeatureMat );
		}
		else if ( m_nClassType == Adaboost )
		{
			t_nS = ((CvBoost*)m_pClassifier)->predict( t_FeatureMat );
		}
		else
		{
			t_nS = ((CvSVM*)m_pClassifier)->predict( t_FeatureMat );
		}

		t_fSumNeg -= t_nS;
	}
	t_fNegRate = t_fSumNeg / t_nNegNumber;		//��������ȷ��

	delete [] t_pFeature;
	cvReleaseMat( &t_FeatureMat );	//�ͷ���Դ

	return 1;
}//Test


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
int RHOG::SearchTarget( ListImage *SrcImg, iRect *& t_pRect, 
						float t_fStartRate, float t_fStepSize, int t_nResizeStep,
						int t_nMatchTime,
						int t_nSearchStep )
{
	//�ж�����
	if ( SrcImg->GetImgWidth() < 70 || SrcImg->GetImgHeight() < 70 )
	{
		return InputImageError;
	}

	if ( SrcImg->GetImgDataType() != uint_8 )
	{
		return InputImageNotSupport;
	}

	if ( t_nSearchStep > 0 )
	{
		m_nSearchStep = t_nSearchStep;
	}

	if ( t_nMatchTime <= 0 )
	{
		t_nMatchTime = m_nMatchTime;
	}

	if ( t_fStartRate <= 0.1f || t_fStartRate > 3.0f || fabs( t_fStepSize ) < 0.01f 
		|| t_fStartRate + t_fStepSize * t_nResizeStep > 3.0f
		|| t_fStartRate + t_fStepSize * t_nResizeStep <= 0.1f )
	{
		return ParIllegal;
	}

	//�жϷ������Ƿ����
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	clock_t Teststart,End1,End2,end3,end4,start2,start3,start4,start5;
	CString out;
	
	Mat t_Image;
	cvtList2Mat( SrcImg, t_Image );
	Teststart=clock();
	//��ʼ��
	Clear();
	End1=clock();
	out.Format("clear():  %lf ms",double(End1-Teststart));
	//AfxMessageBox(out);
	start4=clock();
	InitFeatures();			//���������ռ�
	End2=clock();
	out.Format("InitFeatures():  %lf ms",double(End2-start4));
	//AfxMessageBox(out);
	//ȷ�����Ų���
	int t_nRectNumber;
	t_nRectNumber = 0;

	//ͼ��Ԥ����
	Mat t_SrcImage;		//���ڴ��32λͼ��
	start5=clock();
	PreProcessImage( t_Image, t_SrcImage );
	end3=clock();
	out.Format("PreProcessImage:  %lf ms",double(end3-start5));
	//AfxMessageBox(out);
	//��ͬ�߶�������
	vector <CvRect> t_vTarget;
	int i;
	for ( i = 0; i < t_nResizeStep; ++i )
	{
		int t_nNewWidth;
		int t_nNewHeight;
		t_nNewWidth = (int)( t_Image.cols * t_fStartRate );
		t_nNewHeight = (int)( t_Image.rows * t_fStartRate );

		Mat t_NewImage( t_nNewHeight, t_nNewWidth, CV_32FC1 );		//��ͬ�߶�ͼ��
		resize( t_SrcImage, t_NewImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );

		//����ѵ��ͼƬ�������
		if ( t_Image.channels() == 3 )
		{
			m_TestImage.create( t_nNewHeight, t_nNewWidth, CV_8UC3 );
		}
		else
		{
			m_TestImage.create( t_nNewHeight, t_nNewWidth, CV_8UC1 );
		}

		resize( t_Image, m_TestImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		//����ѵ��ͼƬ�������
		start2=clock();
		CountGrad( t_NewImage );		//Ԥ�����ݶ�
		end3=clock();
	out.Format("countGradImage:  %lf ms",double(end3-start2));
	//AfxMessageBox(out);
	start3=clock();
	SearchTargetPerImg( t_NewImage, t_fStartRate, t_vTarget );
	end4=clock();
	out.Format("SearchTargetPerImage:  %lf ms",double(end4-start3));

	//AfxMessageBox(out);
		t_fStartRate += t_fStepSize;		//���·Ŵ���
	}
	
	return RefineTargetSeq( t_vTarget, t_pRect, t_nMatchTime );
}//SearchTarget


/*****************************************************************
Name:			SearchTargetPerImg
Inputs:
	Mat t_Image - ������ͼ��
	float t_fAugRate - ͼ��Ŵ����
	vector <CvRect> &t_vTarget - ���صĽ������
Return Value:
	int - 1 ��������
		  <0 ��Ӧ�������
Description:	�ڵ���ͼ���м���Ŀ��
*****************************************************************/
int RHOG::SearchTargetPerImg( Mat t_Image, float t_fAugRate, vector <CvRect> &t_vTarget )
{
	//�жϷ������Ƿ����
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	double count=0;
	//����ɨ��
	vector <CvRect> t_vCurRect;

	CvRect t_Rect;
	t_Rect.x = 0;
	t_Rect.y = 0;
	t_Rect.width = m_nImageWidth;
	t_Rect.height = m_nImageWidth;

	float * t_pFeature,*t_oFeature;
	t_pFeature = new float[m_nFeatureNumber];//2160
	
	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( m_nFeatureNumber, 1, CV_32FC1 );

	clock_t start1,end1,start2,end2,start3,end3,start4,end4;CString out;
	start2=clock();
	int width=t_Image.cols;
	int height=t_Image.rows;
	int h_windowx=t_Image.cols/t_Rect.width;
	int h_windowy=t_Image.rows/t_Rect.height;
	CountGrad(t_Image);
	uchar *in_image,*in_m_ANG,*in_m_Mag;
	in_image=t_Image.data;
	in_m_ANG=m_ANGImage.data;
	in_m_Mag=m_MagImage.data;

	//t_oFeature=(float*)malloc(sizeof(float)*m_nFeatureNumber*h_windowx*h_windowy);
	//t_oFeature=(float*)malloc();
	//float *out_image,*out_m_ANG,*out_m_Mag;
	//out_image=(float *)malloc(sizeof(float)*width*height);
	//out_m_ANG=(float *)malloc(sizeof(float)*width*height);
	//out_m_Mag=(float *)malloc(sizeof(float)*width*height);
	//	for(int j=0;j<height;j++)
	//		for(int i=0;i<width;i++)
	//			{
	//				
	//				out_image[i+j*width]=(float)in_image[i+j*width];
	//				out_m_ANG[i+j*width]=(float)in_m_ANG[i+j*width];
	//				out_m_Mag[i+j*width]=(float)in_m_Mag[i+j*width];
	//				float val=(float)in_image[i+j*width];
	//				//printf(" m_ANG:%c ",val);
	//	}
			countFeatures(t_Image.data,t_oFeature,m_nANGle,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage.data,m_MagImage.data,width,height);
			
			float temp=t_oFeature[1260];
			//countFeaturesfloat(out_image,t_oFeature,m_nANGle,m_nMag,m_fNormalMat,m_fMagMat,out_m_ANG,out_m_Mag,width,height);
	for (int i=0;i<h_windowx;i++)
		{
			for(int j=0;j<h_windowy;j++)
		{
			float t_nRes = 0;
			t_pFeature=t_oFeature+(i+h_windowx*j)*2160;
			for ( int k = 0; k < m_nANG; ++k )
			{
				memcpy( t_FeatureMat->data.fl, 
						&t_pFeature[k * m_nANGWidth], 
						(m_nANG - k) * m_nANGWidth * sizeof ( float ) );

				if ( k > 0 )
				{
					memcpy( &t_FeatureMat->data.fl[(m_nANG - k) * m_nANGWidth], 
							t_pFeature, 
							k * m_nANGWidth * sizeof ( float ) );
				}

				if ( m_bSym )
				{
					memcpy( &t_FeatureMat->data.fl[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
							&t_pFeature[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
							m_nANG * m_nCellNumb * sizeof ( float ) / 2 );
				}
				switch( m_nClassType )
				{
				case Adaboost:	t_nRes = (( CvBoost * )m_pClassifier)->predict( t_FeatureMat );break;
				case Rtree:		t_nRes = (( CvRTrees * )m_pClassifier)->predict( t_FeatureMat );break;
				case SVMC:		t_nRes = (( CvSVM * )m_pClassifier)->predict( t_FeatureMat );  break;
				}	
				if ( t_nRes > 0.5f )
				{
					break;
				}
			}

			if( t_nRes > 0.5f )
			{
				CvRect t_AddRect;
				t_AddRect.x = (int)( ( t_Rect.x + t_Rect.width * 0.5f ) / t_fAugRate );
				t_AddRect.y = (int)( ( t_Rect.y + t_Rect.height * 0.5f ) / t_fAugRate );

				t_AddRect.width = (int)( m_nImageWidth / t_fAugRate ) - 3;
				t_AddRect.height = t_AddRect.width;

				t_AddRect.width /= 2;
				t_AddRect.height /= 2;

				t_vTarget.push_back( t_AddRect );

			}

			//������ͼ���ģ��
			if ( m_bSavePosPatch )
			{	
				if( t_nRes >= 0.5f )
				{
					string t_sSave;
					char p[32] = { 0, };	//��ʼ����ʱ�ַ���
					sprintf( p,"%d", g_nPosImageNumber );
					t_sSave = p;
					t_sSave = m_sPosPatchPath + t_sSave;

					g_nPosImageNumber++;

					Mat t_SaveImage;
					t_SaveImage = m_TestImage( t_Rect ).clone();

					ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
					memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );

					t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
				}
			}

			if ( m_bSaveNegPatch )
			{
				if( t_nRes <= 0.5f )
				{
					string t_sSave;
					char p[32] = { 0, };	//��ʼ����ʱ�ַ���
					sprintf( p,"%d", g_nNegImageNumber );
					t_sSave = p;
					t_sSave = m_sNegPatchPath + t_sSave;

					g_nNegImageNumber++;

					Mat t_SaveImage;
					t_SaveImage = m_TestImage( t_Rect ).clone();

					ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
					memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );


					t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
				}
			}
		}
		
		}
		delete [] t_pFeature;
		//delete [] t_oFeature;
	/*	free(out_image);
		free(out_m_ANG);
		free(out_m_Mag);*/
	cvReleaseMat( &t_FeatureMat );
	//while ( t_Rect.x + m_nImageWidth < t_Image.cols )
	//{
	//	t_Rect.y = 0;

	//	while ( t_Rect.y + m_nImageWidth < t_Image.rows )
	//	{
	//		start1=clock();
	//		CountFeature( t_Rect.x, t_Rect.y, t_Rect.width, t_Rect.height, t_pFeature );	//��������
	//	
	//		end1=clock();
	//		out.Format("CountFeature:  %lf ms",double(end1-start1));
	//		//AfxMessageBox(out);
	//		int i;
	//		float t_nRes = 0;
	//		for ( i = 0; i < m_nANG; ++i )
	//		{
	//			memcpy( t_FeatureMat->data.fl, 
	//					&t_pFeature[i * m_nANGWidth], 
	//					(m_nANG - i) * m_nANGWidth * sizeof ( float ) );

	//			if ( i > 0 )
	//			{
	//				memcpy( &t_FeatureMat->data.fl[(m_nANG - i) * m_nANGWidth], 
	//						t_pFeature, 
	//						i * m_nANGWidth * sizeof ( float ) );
	//			}

	//			if ( m_bSym )
	//			{
	//				memcpy( &t_FeatureMat->data.fl[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
	//						&t_pFeature[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
	//						m_nANG * m_nCellNumb * sizeof ( float ) / 2 );
	//			}
	//			switch( m_nClassType )
	//			{
	//			case Adaboost:	t_nRes = (( CvBoost * )m_pClassifier)->predict( t_FeatureMat );break;
	//			case Rtree:		t_nRes = (( CvRTrees * )m_pClassifier)->predict( t_FeatureMat );break;
	//			case SVMC:		t_nRes = (( CvSVM * )m_pClassifier)->predict( t_FeatureMat );  break;
	//			}	
	//			if ( t_nRes > 0.5f )
	//			{
	//				break;
	//			}
	//		}

	//		if( t_nRes > 0.5f )
	//		{
	//			CvRect t_AddRect;
	//			t_AddRect.x = (int)( ( t_Rect.x + t_Rect.width * 0.5f ) / t_fAugRate );
	//			t_AddRect.y = (int)( ( t_Rect.y + t_Rect.height * 0.5f ) / t_fAugRate );

	//			t_AddRect.width = (int)( m_nImageWidth / t_fAugRate ) - 3;
	//			t_AddRect.height = t_AddRect.width;

	//			t_AddRect.width /= 2;
	//			t_AddRect.height /= 2;

	//			t_vTarget.push_back( t_AddRect );

	//		}

	//		//������ͼ���ģ��
	//		if ( m_bSavePosPatch )
	//		{	
	//			if( t_nRes >= 0.5f )
	//			{
	//				string t_sSave;
	//				char p[32] = { 0, };	//��ʼ����ʱ�ַ���
	//				sprintf( p,"%d", g_nPosImageNumber );
	//				t_sSave = p;
	//				t_sSave = m_sPosPatchPath + t_sSave;

	//				g_nPosImageNumber++;

	//				Mat t_SaveImage;
	//				t_SaveImage = m_TestImage( t_Rect ).clone();

	//				ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
	//				memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );

	//				t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
	//			}
	//		}

	//		if ( m_bSaveNegPatch )
	//		{
	//			if( t_nRes <= 0.5f )
	//			{
	//				string t_sSave;
	//				char p[32] = { 0, };	//��ʼ����ʱ�ַ���
	//				sprintf( p,"%d", g_nNegImageNumber );
	//				t_sSave = p;
	//				t_sSave = m_sNegPatchPath + t_sSave;

	//				g_nNegImageNumber++;

	//				Mat t_SaveImage;
	//				t_SaveImage = m_TestImage( t_Rect ).clone();

	//				ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
	//				memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );


	//				t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
	//			}
	//		}
	//		count++;
	//		t_Rect.y += m_nSearchStep;
	//	}

	//	t_Rect.x += m_nSearchStep;
	//}
	//		out.Format("ѭ��:  %lf ��",count);
			//AfxMessageBox(out);

	

	return 1;
}//SearchTargetPerImg


/*****************************************************************
Name:			CountFeatureFromImg
Inputs:
	Mat t_Image - ������������ͼ��
	float *t_pFeatures - ���ص���������
Return Value:
	int - 1 ��������
		  <0 ��Ӧ�������
Description:	���㵥��ͼƬ������
*****************************************************************/
int RHOG::CountFeatureFromImg( Mat t_Image, float *t_pFeatures )
{
	//�����������
	Clear();	

	//��ʼ�������ռ�
	InitFeatures();			//���������ռ�

	//��ʼ��ͼ��
	Mat t_GrayImage;
	PreProcessImage( t_Image, t_GrayImage );

	m_Image.create( m_nImageWidth, m_nImageWidth, CV_32SC1 );
	resize( t_GrayImage, m_Image, Size( m_nImageWidth, m_nImageWidth ), 0, 0, CV_INTER_LINEAR );

	//�����ݶ�
	CountGrad( m_Image );

	//����cell��������
	CountCell( 0, 0, m_nImageWidth, m_nImageWidth );

	//ƽ��Cell��������
	SmoothCell();

	//��Cell���нǶȼ��ƽ��
	if ( m_bRSC )
	{
		RSmoothCell();
	}

	//��Cell���о����ƽ��
	if ( m_bDSC )
	{
		DSmoothCell();
	}

	//����m_nBIN��������
	Countm_nBIN();

	//��һ��m_nBIN��������
	Normalm_nBIN();

	//����Գ�����
	if ( m_bSym )
	{
		CountSym();
	}

	memcpy( t_pFeatures, m_pfFeature, m_nFeatureNumber * sizeof( float ) );

	return 1;
}//CountFeatureFromImg


/*****************************************************************
Name:			CountFeature
Inputs:
	int t_nX - ��ȡ��������
	int t_nY 
	int t_nWidth
	int t_nHeight
	float *&t_pFeatures - ���ص���������
Return Value:
	int - 1 ��������
		  <0 ��Ӧ�������
Description:	����ָ�������������������ͼ�еľֲ��ӿ飩
*****************************************************************/
int RHOG::CountFeature( int t_nX, int t_nY, int t_nWidth, int t_nHeight, float *t_pFeatures )
{
	//clock_t start1,end1,start2,end2,start3,end3,start4,end4;
	CString out;
//	 cudaEvent_t start,stop,start1,stop1,start2,stop2,start3,stop3;
	 float time_elapsed=0;
//	cudaEventCreate(&start);
//cudaEventCreate(&stop);
//cudaEventRecord(start, 0);
//	
	//_LARGE_INTEGER time_start,s1,s2,s3;  //��ʼʱ��  
	//_LARGE_INTEGER time_over,e1,e2,e3;   //����ʱ��  
	//double dqFreq;      //��ʱ��Ƶ��  
	//LARGE_INTEGER f;    //��ʱ��Ƶ��  
	//QueryPerformanceFrequency(&f);  
	//dqFreq=(double)f.QuadPart;  
	//QueryPerformanceCounter(&time_start);
	//����cell��������
	CountCell( t_nX, t_nY, t_nWidth, t_nHeight );
	// QueryPerformanceCounter(&time_over);    //��ʱ����  
	// time_elapsed=1000000*(time_over.QuadPart-time_start.QuadPart)/dqFreq;  
	////����1000000�ѵ�λ���뻯Ϊ΢�룬����Ϊ1000 000/��cpu��Ƶ��΢��  
	//out.Format("Countcell:  %lf us",time_elapsed);
			//AfxMessageBox(out);
			 //QueryPerformanceCounter(&s1);
	//ƽ��Cell��������
	SmoothCell();
	 //QueryPerformanceCounter(&e1);    //��ʱ����  
	 //time_elapsed=1000000*(e1.QuadPart-s1.QuadPart)/dqFreq;  
	 //out.Format("smoothcell:  %lf us",time_elapsed);
			//AfxMessageBox(out);
	//��Cell���нǶȼ��ƽ��
	if ( m_bRSC )
	{
		RSmoothCell();
	}

	//��Cell���о����ƽ��
	if ( m_bDSC )
	{
		DSmoothCell();
	}
	 //QueryPerformanceCounter(&s2);
	
	//����m_nBIN��������
	Countm_nBIN();
	
	 //QueryPerformanceCounter(&e2);    //��ʱ����  
	 //time_elapsed=1000000*(e2.QuadPart-s2.QuadPart)/dqFreq;  
	 //out.Format("Countm_nBIN():  %lf us",time_elapsed);
			//AfxMessageBox(out);
			//QueryPerformanceCounter(&s3);
	//��һ��m_nBIN��������
	Normalm_nBIN();
	 //QueryPerformanceCounter(&e3);    //��ʱ����  
	 //time_elapsed=1000000*(e3.QuadPart-s3.QuadPart)/dqFreq;  
	 //out.Format("Normalm_nBIN():  %lf us",time_elapsed);
			//AfxMessageBox(out);
	//����Գ�����
	if ( m_bSym )
	{
		CountSym();
	}

	//���ƴ����ص�����ֵ
	memcpy( t_pFeatures, m_pfFeature, m_nFeatureNumber * sizeof( float ) );

	return 1;
}//CountFeature


/*****************************************************************s
Name:			Clear
Inputs:
	none.
Return Value:
	none.
Description:	ע���ռ�
*****************************************************************/
void RHOG::Clear(void)
{
	//�����������
	if ( m_pfFeature != NULL )
	{
		delete [] m_pfFeature;
		m_pfFeature = NULL;
	}

	if ( m_pCellFeatures != NULL )
	{
		delete [] m_pCellFeatures;
		m_pCellFeatures = NULL;
	}
}//Clear


/*****************************************************************
Name:			InitFeatures
Inputs:
	none.
Return Value:
	none.
Description:	��ʼ�������ռ�
*****************************************************************/
void RHOG::InitFeatures(void)
{
	//����
	//�����������
	if ( m_pfFeature != NULL )
	{
		delete [] m_pfFeature;
		m_pfFeature = NULL;
	}

	if ( m_pCellFeatures != NULL )
	{
		delete [] m_pCellFeatures;
		m_pCellFeatures = NULL;
	}

	m_pfFeature = new float [m_nFeatureNumber];					//���ٿռ�
	m_pCellFeatures = new float [m_nANG * m_nCellNumb * m_nBIN];//���ٿռ�

	m_nCellWidth = ( m_nImageWidth / 2 ) / m_nCellNumb + 1;		//ÿ��cell�Ŀ��

}//InitFeatures


/*****************************************************************
Name:			PreProcessImage
Inputs:
	Mat t_Image - ����ͼ��
	 Mat &t_TarImage - Ԥ�����ͼ��
Return Value:
	none.
Description:	Ԥ����ͼ��
*****************************************************************/

void RHOG::PreProcessImage( Mat t_Image, Mat &t_TarImage )
{
	//ת�Ҷ�ͼ�񣬲�������ͳһ��С
	Mat t_GrayImage;

	if ( t_Image.channels() == 3 )
	{
		cvtColor( t_Image, t_GrayImage, CV_BGR2GRAY );
	}
	else
	{
		t_GrayImage = t_Image.clone();
	}
	clock_t start1,end1,start2,end2,start3,end3,start4,end4;
	CString out;

	/*unsigned char *src=t_GrayImage.data;
	unsigned char *dst=t_GrayImage.data;*/
	int width;
	width=t_GrayImage.cols;
	int height;
	height=t_GrayImage.rows;
	//�˲�
		start4=clock();
	medianBlur( t_GrayImage, t_GrayImage, m_nFilterSize );
	end4=clock();
	out.Format("��ֵ�˲�:  %lf ms",double(end4-start4));
			//AfxMessageBox(out);
		start1=clock();
		//MedianFilter(t_GrayImage.data,t_GrayImage.data ,width,height);
		end1=clock();
	out.Format("��ֵ�˲�:  %lf ms",double(end1-start1));
			//AfxMessageBox(out);
	//imshow("meidan",t_GrayImage);
	//waitKey(0);
	GaussianBlur( t_GrayImage, t_GrayImage, cvSize( 3, 3 ), 1 );

	//��ͼ��תΪfloat�ͣ�����ͼ����й�һ������
			start2=clock();
	t_GrayImage.convertTo( t_TarImage, CV_32F );
	end2=clock();
	out.Format("ת�Ҷ�ͼ:  %lf ms",double(end2-start2));
			//AfxMessageBox(out);
	////�Ҷ�gamma����
	//��ʱȥ��
			start3=clock();
	int i, j;
	for ( j = 0; j < t_Image.rows; ++j )
	{
		float *t_pData; 
		t_pData = t_TarImage.ptr<float>(j);

		for ( i = 0; i < t_Image.cols; ++i )
		{
			t_pData[i] = sqrt( t_pData[i] );
		}
	}
	end3=clock();
	out.Format("gamma��һ:  %lf ms",double(end3-start3));
			//AfxMessageBox(out);
}//PreProcessImage


/*****************************************************************
Name:			CountGrad
Inputs:
	Mat t_Image - ����ͼ��
Return Value:
	none.
Description:	�����ݶȣ����洢��m_MagImage��m_ANGImageͼ����
*****************************************************************/
void RHOG::CountGrad( Mat t_Image )
{
	//��ʼ���ռ�
	m_MagImage.create( t_Image.rows, t_Image.cols, CV_32FC1 );

	m_ANGImage.create( t_Image.rows, t_Image.cols, CV_32FC1 );

	//��ʼ����
	Mat t_DeltaX;
	t_DeltaX.create( t_Image.rows, t_Image.cols, CV_32FC1 );

	Mat t_DeltaY;
	t_DeltaY.create( t_Image.rows, t_Image.cols, CV_32FC1 );

	Sobel( t_Image, t_DeltaX, CV_32FC1, 1, 0, 1 );
	Sobel( t_Image, t_DeltaY, CV_32FC1, 0, 1, 1 );

	int i, j;
	for ( j = 1; j < t_Image.rows - 1; ++j )
	{
		float *t_pPosDeltaX;		//Դ����ָ��
		t_pPosDeltaX = t_DeltaX.ptr<float>(j);

		float *t_pPosDeltaY;		//Դ����ָ��
		t_pPosDeltaY = t_DeltaY.ptr<float>(j);

		float *t_pPosMag;		//�ݶ�ģָ��
		t_pPosMag = m_MagImage.ptr<float>(j);

		float *t_pPosm_nANG;		//�ݶȽǶ�ָ��
		t_pPosm_nANG = m_ANGImage.ptr<float>(j);

		for ( i = 1; i < t_Image.cols - 1; ++i )
		{
			float t_fDeltaX;
			float t_fDeltaY;
			t_fDeltaX = t_pPosDeltaX[i];
			t_fDeltaY = t_pPosDeltaY[i];

			//t_pPosMag[i] = pow( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY, 0.125f );	//���滻���һ��
			t_pPosMag[i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );
			t_pPosm_nANG[i] = atan2( t_fDeltaX, t_fDeltaY );
		}
	}
}//CountGrad


/*****************************************************************
Name:			CountCell
Inputs:
	int t_nX - ��������
	int t_nY
	int t_nWidth
	int t_nHeight
Return Value:
	none.
Description:	����Cell
*****************************************************************/
void RHOG::CountCell( int t_nX, int t_nY, int t_nWidth, int t_nHeight )
{
	int t_nEndX;
	int t_nEndY;
	t_nEndX = t_nX + t_nWidth - 1;
	t_nEndY = t_nY + t_nHeight - 1;

	int t_nCellFeatureSize;
	t_nCellFeatureSize = m_nANG * m_nCellNumb * m_nBIN;

	int i, j;
	for ( i = 0; i < t_nCellFeatureSize; ++i )
	{
		m_pCellFeatures[i] = 0;
	}


	//����cellfeature
	int t_nLineWidth;		//ÿ������Ŀ��
	t_nLineWidth = m_nCellNumb * m_nBIN;

	for ( j = t_nY + 1; j < t_nEndY; ++j )
	{
		float *t_pMagData;
		float *t_pm_nANGData;
		t_pMagData = m_MagImage.ptr<float>(j);
		t_pm_nANGData = m_ANGImage.ptr<float>(j);

		for ( i = t_nX + 1; i < t_nEndX; ++i )
		{
			//�ж��Ƿ񳬳��뾶
			if ( m_fMagMat[( j - t_nY) * m_nImageWidth + i - t_nX] > m_nImageWidth / 2.0f )
			{
				continue;
			}

			//����m_nBIN
			float t_fm_nANGel;
			t_fm_nANGel = t_pm_nANGData[i] - m_fNormalMat[( j - t_nY) * m_nImageWidth + i - t_nX];

			while ( t_fm_nANGel < 0 )
			{
				t_fm_nANGel += (float)PI;
			}

			int t_nm_nBIN =  (int)( t_fm_nANGel * m_nBIN / PI );

			//�����������
			//int t_nCir;
			//t_nCir = (int)( m_fMagMat[( j - t_nY) * m_nImageWidth + i - t_nX] / m_nCellWidth);
			//m_pCellFeatures[t_nLineWidth * m_nANGle[( j - t_nY) * m_nImageWidth + i - t_nX] + t_nCir * m_nBIN + t_nm_nBIN ] += t_pMagData[i];
			m_pCellFeatures[t_nLineWidth * m_nANGle[( j - t_nY) * m_nImageWidth + i - t_nX] + m_nMag[( j - t_nY) * m_nImageWidth + i - t_nX] * m_nBIN + t_nm_nBIN ] += t_pMagData[i];
		}
	}
}//CountCell


/*****************************************************************
Name:			SmoothCell
	Inputs:
none.
Return Value:
	none.
Description:	��Cell����ƽ��(m_pCellFeatures)
*****************************************************************/
void RHOG::SmoothCell( void )
{
	int t_nLineWidth;		//ÿ������Ŀ��
	t_nLineWidth = m_nCellNumb * m_nBIN;//7*10

	int i, j, k;

	float * t_pTemp;		//��ʱ����
	t_pTemp = new float [m_nBIN];
	for ( k = 0; k < m_nANG; ++k )//18
	{
		for ( j = 0; j < m_nCellNumb; ++j )//7
		{
			for ( i = 0; i< m_nBIN; ++i )//10
			{
				int t_nLeft;
				int t_nRight;
				t_nLeft = ( i - 1 + m_nBIN ) % m_nBIN;
				t_nRight = ( i + 1 ) % m_nBIN;

				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.8f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nLeft] * 0.1f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nRight] * 0.1f;
			}

			for ( i = 0; i < m_nBIN; ++i )
			{
				m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//SmoothCell


/*****************************************************************
Name:			RSmoothCell
Inputs:
	none.
Return Value:
	none.
Description:	��Cell���нǶ�ƽ��(m_pCellFeatures)
*****************************************************************/
void RHOG::RSmoothCell( void )
{
	int t_nLineWidth;		//ÿ������Ŀ��
	t_nLineWidth = m_nCellNumb * m_nBIN;

	int i, j, k;

	float * t_pTemp;		//��ʱ�����м���
	t_pTemp = new float [m_nANG];
	for ( k = 0; k < m_nCellNumb; ++k )
	{
		for ( j = 0; j < m_nBIN; ++j )
		{


			for ( i = 0; i < m_nANG; ++i )
			{
				int t_nLeft;
				int t_nRight;
				t_nLeft = ( i - 1 + m_nANG ) % m_nANG;
				t_nRight = ( i + 1 ) % m_nANG;

				t_pTemp[i] = m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] * 0.8f 
					+ m_pCellFeatures[t_nLeft * t_nLineWidth + k * m_nBIN + j] * 0.1f 
					+ m_pCellFeatures[t_nRight * t_nLineWidth + k * m_nBIN + j] * 0.1f;
			}

			for ( i = 0; i < m_nANG; ++i )
			{
				m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//RSmoothCell



/*****************************************************************
Name:			DSmoothCell
Inputs:
	none.
Return Value:
	none.
Description:	��Cell���о���ƽ��(m_pCellFeatures)
*****************************************************************/
void RHOG::DSmoothCell( void )
{
	int t_nLineWidth;		//ÿ������Ŀ��
	t_nLineWidth = m_nCellNumb * m_nBIN;

	int i, j, k;

	float * t_pTemp;		//��ʱ�����м���
	t_pTemp = new float[m_nCellNumb];

	for ( k = 0; k < m_nANG; ++k )
	{
		for ( j = 0; j < m_nBIN; ++j )
		{

			for ( i = 1; i < m_nCellNumb - 1; ++i )
			{
				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.5f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i + 1] * 0.25f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i - 1] * 0.25f;
			}

			for ( i = 1; i < m_nCellNumb - 1; ++i )
			{
				m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] = t_pTemp[i];
			}
		}
	}

	delete [] t_pTemp;
}//DSmoothCell


/*****************************************************************
Name:			Countm_nBIN
Inputs:
	none.
Return Value:
	none.
Description:	����m_nBIN
*****************************************************************/
void RHOG::Countm_nBIN( void )
{
	int t_nLineWidthBlob;		//ÿ������Ŀ��
	t_nLineWidthBlob = m_nBlobNumb * m_nCellPerBlob * m_nBIN;//4*3*10
	int t_nBlobWidth;
	t_nBlobWidth = m_nCellPerBlob * m_nBIN;//3*10

	int t_nLineWidthCell;		//cellÿ������Ŀ��
	t_nLineWidthCell = m_nCellNumb * m_nBIN;//7*10

	int i, j, k;

	for ( k = 0; k < m_nANG; ++k )
	{
		for ( j = 0; j < m_nBlobNumb; ++j )
		{
			for ( i = 0; i< m_nCellPerBlob; ++i )
			{
				memcpy( &m_pfFeature[k * t_nLineWidthBlob + j * t_nBlobWidth + i * m_nBIN], 
					&m_pCellFeatures[k * t_nLineWidthCell + (i + j) * m_nBIN], 
					m_nBIN * sizeof( float ) );
			}
		}
	}
}//Countm_nBIN


/*****************************************************************
Name:			Normalm_nBIN
Inputs:
	none.
Return Value:
	none.
Description:	��һ��m_nBIN
*****************************************************************/
void RHOG::Normalm_nBIN( void )
{
	int i, j;
	int t_nDataSize;		//�����������ȣ�m_nBIN��
	t_nDataSize = m_nCellPerBlob * m_nBIN;//30

	int t_nm_nBlobNumber;		//blob������
	t_nm_nBlobNumber = m_nANG * m_nBlobNumb;//18*4=72

	for ( j = 0; j < t_nm_nBlobNumber; ++j )
	{
		//ͳ��L2-normal��ĸ
		float * t_fPos;		//���ݷ���ָ��
		t_fPos = &m_pfFeature[j * t_nDataSize];

		float t_fAddUp;
		t_fAddUp = 0;
		for ( i = 0; i < t_nDataSize; ++i )
		{
			//if ( t_fAddUp < t_fPos[i] )	//���滻����ֵ��һ��
			//{
			//	t_fAddUp = t_fPos[i];
			//}
			t_fAddUp += t_fPos[i] * t_fPos[i];
		}

		t_fAddUp = sqrt( t_fAddUp + 1.0f );
		//t_fAddUp += 0.1f;		//���滻����ֵ��һ��

		for ( i = 0; i < t_nDataSize; ++i )
		{
			t_fPos[i] = t_fPos[i] / t_fAddUp;
		}
	}
}//Normalm_nBIN


/*****************************************************************
Name:			CountSym
Inputs:
	none.
Return Value:
	none.
Description:	����Գ�������
*****************************************************************/
void RHOG::CountSym( void )
{
	float * t_pPos;		//ָ��Գ�������ʼ��
	t_pPos = &m_pfFeature[ m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN ];

	int i,j,k;
	for ( i = 0; i < m_nANG / 2; ++i )
	{
		for ( j = 0; j < m_nCellNumb; ++j )
		{
			float t_fSum;		//��¼�ۼ�
			t_fSum = 0;

			float t_fAddUpA;	//�����ֵ
			float t_fAddUpB;
			t_fAddUpA = 0;
			t_fAddUpB = 0;

			for ( k = 0; k < m_nBIN; ++k )
			{
				float t_fA;
				float t_fB;

				t_fA = m_pCellFeatures[ i * m_nCellNumb * m_nBIN + j * m_nBIN + k ];
				t_fB = m_pCellFeatures[ ( m_nANG - i - 1 ) * m_nCellNumb * m_nBIN + j * m_nBIN + ( m_nBIN - k ) ];
				t_fSum += t_fA * t_fB;
				t_fAddUpA += t_fA;
				t_fAddUpB += t_fB;
			}

			t_fSum = t_fSum / ( t_fAddUpA * t_fAddUpB + 1 );

			*t_pPos = t_fSum;
			++t_pPos;
		}
	}
}//CountSym


/*****************************************************************
Name:			RefineTargetSeq
Inputs:
	vector <CvRect> t_vTarget - �����Ŀ�����
	iRect *& t_pRect - ���ص�Ŀ�����
	int t_nMatchTime  - ͳ��Ŀ��ʱ�ص��Ĵ���
Return Value:
	int - >0 Ŀ������
		  <0 ��Ӧ�������
Description:	���¹鲢Ŀ����У����þ��෨
*****************************************************************/
int RHOG::RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime )	
{
	int t_nRectNum;	
	t_nRectNum = (int)t_vTarget.size();
	clock_t start,stop;
	_LARGE_INTEGER time_start,s1,s2,s3;  //��ʼʱ��  
	_LARGE_INTEGER time_over,e1,e2,e3;   //����ʱ��  
	double dqFreq;      //��ʱ��Ƶ��  
	LARGE_INTEGER f;    //��ʱ��Ƶ��  
	QueryPerformanceFrequency(&f);  
	dqFreq=(double)f.QuadPart;  
	QueryPerformanceCounter(&time_start);
	//��������
	vector <TargetArea> t_vAreaSeq;
	int i, j;
	start=clock();
	for ( i = 0; i < t_nRectNum; ++i )
	{
		int t_nCenterX;
		int t_nCenterY;
		t_nCenterX = t_vTarget[i].x;	// + t_vTarget[i].width / 2;
		t_nCenterY = t_vTarget[i].y;	// + t_vTarget[i].height / 2;

		bool t_bFinded;		//�Ƿ��ҵ�ƥ��Ŀ��
		t_bFinded = false;

		for( j = 0; j < (int)t_vAreaSeq.size(); ++j )
		{
			if ( abs( t_nCenterX - t_vAreaSeq[j].m_nCenterX ) < t_vTarget[i].width / 3
				&& abs( t_nCenterY - t_vAreaSeq[j].m_nCenterY ) < t_vTarget[i].height / 3
				&& t_vAreaSeq[j].m_nWidth == t_vTarget[i].width )
			{
				t_vAreaSeq[j].m_nCenterX = ( t_vAreaSeq[j].m_nCenterX * t_vAreaSeq[j].m_nDupeNumber + t_nCenterX ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nCenterY = ( t_vAreaSeq[j].m_nCenterY * t_vAreaSeq[j].m_nDupeNumber + t_nCenterY ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nWidth = ( t_vAreaSeq[j].m_nWidth * t_vAreaSeq[j].m_nDupeNumber + t_vTarget[i].width ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nHeight = ( t_vAreaSeq[j].m_nHeight * t_vAreaSeq[j].m_nDupeNumber + t_vTarget[i].height ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nDupeNumber ++;

				t_bFinded = true;

				break;
			}
		}

		if ( !t_bFinded )
		{
			TargetArea t_TarAdd;
			t_TarAdd.m_nCenterX = t_nCenterX;
			t_TarAdd.m_nCenterY = t_nCenterY;
			t_TarAdd.m_nWidth = t_vTarget[i].width;
			t_TarAdd.m_nHeight = t_vTarget[i].height;
			t_TarAdd.m_nDupeNumber = 1;

			t_vAreaSeq.push_back( t_TarAdd );
		}
	}


	//ɾ����������
	int t_nTarNUmber;
	t_nTarNUmber = (int)t_vAreaSeq.size();
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber >= 2 )
		{
			t_vAreaSeq[i].m_nDupeNumber += 1;
		}
	}


	//�ϲ�����
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber <= 0 )
		{
			continue;
		}

		for ( j = i + 1; j < (int)t_vAreaSeq.size(); ++j )
		{
			if ( t_vAreaSeq[j].m_nDupeNumber <= 0 )
			{
				continue;
			}

			if ( abs( t_vAreaSeq[i].m_nCenterX - t_vAreaSeq[j].m_nCenterX ) < ( t_vAreaSeq[i].m_nWidth + t_vAreaSeq[j].m_nWidth ) / 3
				&& abs( t_vAreaSeq[i].m_nCenterY - t_vAreaSeq[j].m_nCenterY ) < ( t_vAreaSeq[i].m_nWidth + t_vAreaSeq[j].m_nWidth ) / 3 )
			{
				t_vAreaSeq[i].m_nCenterX = ( t_vAreaSeq[i].m_nCenterX * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nCenterX * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nCenterY = ( t_vAreaSeq[i].m_nCenterY * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nCenterY * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nWidth = ( t_vAreaSeq[i].m_nWidth * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nWidth * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nHeight = ( t_vAreaSeq[i].m_nHeight * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nHeight * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nDupeNumber = t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber;

				t_vAreaSeq[j].m_nDupeNumber = -1;

				t_nTarNUmber--;
			}
		}
	}


	//ɾ���ϵ͸�������
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber < t_nMatchTime && t_vAreaSeq[i].m_nDupeNumber > 0 )
		{
			t_vAreaSeq[i].m_nDupeNumber = -1;
			t_nTarNUmber--;
		}
	}


	//��������
	if ( t_nTarNUmber <= 0 )		//���û��Ŀ�ֱ꣬�ӷ���NULL
	{
		return 0;
	}

	t_pRect = new iRect[t_nTarNUmber];

	j = 0;
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber > 0 )
		{
			t_pRect[j].x = t_vAreaSeq[i].m_nCenterX - t_vAreaSeq[i].m_nWidth;
			t_pRect[j].y = t_vAreaSeq[i].m_nCenterY - t_vAreaSeq[i].m_nHeight;
			t_pRect[j].m_nWidth = t_vAreaSeq[i].m_nWidth * 2 - 2;
			t_pRect[j].m_nHeight = t_vAreaSeq[i].m_nHeight * 2 - 2;

			j++;
		}
	}
		CString out;
	 QueryPerformanceCounter(&time_over);    //��ʱ����  
	float  time_elapsed=1000000*(time_over.QuadPart-time_start.QuadPart)/dqFreq;  
	//����1000000�ѵ�λ���뻯Ϊ΢�룬����Ϊ1000 000/��cpu��Ƶ��΢��  
   
	out.Format("RefineTarget:  %lf us",time_elapsed);
			//AfxMessageBox(out);
	return t_nTarNUmber;
}//RefineTargetSeq


/*****************************************************************
Name:			GetImageList
Inputs:
	string t_sPath - ���ص�·��
	vector <string> t_vFileName - �ļ�������
Return Value:
	int - ͼ������.
Description:	��ȡԭʼͼ���б�
*****************************************************************/
int RHOG::GetImageList( string t_sPath, vector <string> &t_vFileName )
{
	//��ն���
	t_vFileName.clear();
	int t_nEnd = 0;


	//��ȡ��·���µ������ļ�  
	_finddata_t file;
long	long lf;

	string t_sTempPath = t_sPath + "*";

	//�����ļ���·��
	locale loc = locale::global(locale(""));

	lf = (long long )_findfirst( t_sTempPath.c_str(), &file );

	if ( lf == -1 ) 
	{
		locale::global(locale("C"));//��ԭȫ�������趨

		return 0;
	} 
	else  
	{
		while( _findnext( lf, &file ) == 0 ) 
		{
			//����ļ���
			//cout<<file.name<<endl;
			if ( strcmp( file.name, "." ) == 0 || strcmp( file.name, ".." ) == 0 )
			{
				continue;
			}


			string m_strFileExt = strrchr( file.name, '.' );

			if ( m_strFileExt == ".jpg" || m_strFileExt == ".JPG" || m_strFileExt == ".Jpg"
				|| m_strFileExt == ".bmp" || m_strFileExt == ".BMP" || m_strFileExt == ".PNG"|| m_strFileExt == ".png")		//ֻ����jpg��ʽ�ļ������账��������ʽ�����������
			{
				m_strFileExt = t_sPath + file.name;	//��������·��+�ļ���

				t_vFileName.push_back( m_strFileExt );	//����ļ�
				t_nEnd++;
			}
		}
	}
	_findclose(lf);

	locale::global(locale("C"));//��ԭȫ�������趨

	return t_nEnd;
}//GetImageList


/*****************************************************************
Name:			cvtList2Mat
Inputs:
	ListImage *SrcImg - ����ͼ��
	Mat & t_Image - ���ͼ��
Return Value:
	int 1 - ��������
		<0 ��Ӧ������� 
Description:	��listimageͼ��תΪiplͼ��
*****************************************************************/
int RHOG::cvtList2Mat( ListImage *SrcImg, Mat & t_Image )
{
	if ( SrcImg->GetImgChannel() == 1 )
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC1 );

		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;
		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
			}
		}
	}
	else if ( SrcImg->GetImgChannel() == 3 )
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC3 );

		int a = t_Image.cols;
		int b = t_Image.rows;
		int c = t_Image.channels();
		CvSize d = t_Image.size();

		int e = SrcImg->GetImgDataSize();
	
		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;
		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
			}
		}
	}
	else
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC3 );
	
		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;

		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				
				t_pSrc++;
			}
		}
	}
	
	return 1;
}//cvtList2Mat