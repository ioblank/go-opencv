// Copyright 2011 <chaishushan@gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package opencv

//#include "opencv.h"
//#cgo linux  pkg-config: opencv
//#cgo darwin pkg-config: opencv
//#cgo freebsd pkg-config: opencv
//#cgo windows LDFLAGS: -lopencv_core242.dll -lopencv_imgproc242.dll -lopencv_photo242.dll -lopencv_highgui242.dll -lstdc++
import "C"
import (
	//"errors"
	"unsafe"
)

func init() {
}

const (
	CV_BGR2BGRA = C.CV_BGR2BGRA
	CV_RGB2RGBA = C.CV_RGB2RGBA

	CV_BGRA2BGR = C.CV_BGRA2BGR
	CV_RGBA2RGB = C.CV_RGBA2RGB

	CV_BGR2RGBA = C.CV_BGR2RGBA
	CV_RGB2BGRA = C.CV_RGB2BGRA

	CV_RGBA2BGR = C.CV_RGBA2BGR
	CV_BGRA2RGB = C.CV_BGRA2RGB

	CV_BGR2RGB = C.CV_BGR2RGB
	CV_RGB2BGR = C.CV_RGB2BGR

	CV_BGRA2RGBA = C.CV_BGRA2RGBA
	CV_RGBA2BGRA = C.CV_RGBA2BGRA

	CV_BGR2GRAY  = C.CV_BGR2GRAY
	CV_RGB2GRAY  = C.CV_RGB2GRAY
	CV_GRAY2BGR  = C.CV_GRAY2BGR
	CV_GRAY2RGB  = C.CV_GRAY2RGB
	CV_GRAY2BGRA = C.CV_GRAY2BGRA
	CV_GRAY2RGBA = C.CV_GRAY2RGBA
	CV_BGRA2GRAY = C.CV_BGRA2GRAY
	CV_RGBA2GRAY = C.CV_RGBA2GRAY

	CV_BGR2BGR565  = C.CV_BGR2BGR565
	CV_RGB2BGR565  = C.CV_RGB2BGR565
	CV_BGR5652BGR  = C.CV_BGR5652BGR
	CV_BGR5652RGB  = C.CV_BGR5652RGB
	CV_BGRA2BGR565 = C.CV_BGRA2BGR565
	CV_RGBA2BGR565 = C.CV_RGBA2BGR565
	CV_BGR5652BGRA = C.CV_BGR5652BGRA
	CV_BGR5652RGBA = C.CV_BGR5652RGBA

	CV_GRAY2BGR565 = C.CV_GRAY2BGR565
	CV_BGR5652GRAY = C.CV_BGR5652GRAY

	CV_BGR2BGR555  = C.CV_BGR2BGR555
	CV_RGB2BGR555  = C.CV_RGB2BGR555
	CV_BGR5552BGR  = C.CV_BGR5552BGR
	CV_BGR5552RGB  = C.CV_BGR5552RGB
	CV_BGRA2BGR555 = C.CV_BGRA2BGR555
	CV_RGBA2BGR555 = C.CV_RGBA2BGR555
	CV_BGR5552BGRA = C.CV_BGR5552BGRA
	CV_BGR5552RGBA = C.CV_BGR5552RGBA

	CV_GRAY2BGR555 = C.CV_GRAY2BGR555
	CV_BGR5552GRAY = C.CV_BGR5552GRAY

	CV_BGR2XYZ = C.CV_BGR2XYZ
	CV_RGB2XYZ = C.CV_RGB2XYZ
	CV_XYZ2BGR = C.CV_XYZ2BGR
	CV_XYZ2RGB = C.CV_XYZ2RGB

	CV_BGR2YCrCb = C.CV_BGR2YCrCb
	CV_RGB2YCrCb = C.CV_RGB2YCrCb
	CV_YCrCb2BGR = C.CV_YCrCb2BGR
	CV_YCrCb2RGB = C.CV_YCrCb2RGB

	CV_BGR2HSV = C.CV_BGR2HSV
	CV_RGB2HSV = C.CV_RGB2HSV

	CV_BGR2Lab = C.CV_BGR2Lab
	CV_RGB2Lab = C.CV_RGB2Lab

	CV_BayerBG2BGR = C.CV_BayerBG2BGR
	CV_BayerGB2BGR = C.CV_BayerGB2BGR
	CV_BayerRG2BGR = C.CV_BayerRG2BGR
	CV_BayerGR2BGR = C.CV_BayerGR2BGR

	CV_BayerBG2RGB = C.CV_BayerBG2RGB
	CV_BayerGB2RGB = C.CV_BayerGB2RGB
	CV_BayerRG2RGB = C.CV_BayerRG2RGB
	CV_BayerGR2RGB = C.CV_BayerGR2RGB

	CV_BGR2Luv = C.CV_BGR2Luv
	CV_RGB2Luv = C.CV_RGB2Luv
	CV_BGR2HLS = C.CV_BGR2HLS
	CV_RGB2HLS = C.CV_RGB2HLS

	CV_HSV2BGR = C.CV_HSV2BGR
	CV_HSV2RGB = C.CV_HSV2RGB

	CV_Lab2BGR = C.CV_Lab2BGR
	CV_Lab2RGB = C.CV_Lab2RGB
	CV_Luv2BGR = C.CV_Luv2BGR
	CV_Luv2RGB = C.CV_Luv2RGB
	CV_HLS2BGR = C.CV_HLS2BGR
	CV_HLS2RGB = C.CV_HLS2RGB

	CV_BayerBG2BGR_VNG = C.CV_BayerBG2BGR_VNG
	CV_BayerGB2BGR_VNG = C.CV_BayerGB2BGR_VNG
	CV_BayerRG2BGR_VNG = C.CV_BayerRG2BGR_VNG
	CV_BayerGR2BGR_VNG = C.CV_BayerGR2BGR_VNG

	CV_BayerBG2RGB_VNG = C.CV_BayerBG2RGB_VNG
	CV_BayerGB2RGB_VNG = C.CV_BayerGB2RGB_VNG
	CV_BayerRG2RGB_VNG = C.CV_BayerRG2RGB_VNG
	CV_BayerGR2RGB_VNG = C.CV_BayerGR2RGB_VNG

	CV_BGR2HSV_FULL = C.CV_BGR2HSV_FULL
	CV_RGB2HSV_FULL = C.CV_RGB2HSV_FULL
	CV_BGR2HLS_FULL = C.CV_BGR2HLS_FULL
	CV_RGB2HLS_FULL = C.CV_RGB2HLS_FULL

	CV_HSV2BGR_FULL = C.CV_HSV2BGR_FULL
	CV_HSV2RGB_FULL = C.CV_HSV2RGB_FULL
	CV_HLS2BGR_FULL = C.CV_HLS2BGR_FULL
	CV_HLS2RGB_FULL = C.CV_HLS2RGB_FULL

	CV_LBGR2Lab = C.CV_LBGR2Lab
	CV_LRGB2Lab = C.CV_LRGB2Lab
	CV_LBGR2Luv = C.CV_LBGR2Luv
	CV_LRGB2Luv = C.CV_LRGB2Luv

	CV_Lab2LBGR = C.CV_Lab2LBGR
	CV_Lab2LRGB = C.CV_Lab2LRGB
	CV_Luv2LBGR = C.CV_Luv2LBGR
	CV_Luv2LRGB = C.CV_Luv2LRGB

	CV_BGR2YUV = C.CV_BGR2YUV
	CV_RGB2YUV = C.CV_RGB2YUV
	CV_YUV2BGR = C.CV_YUV2BGR
	CV_YUV2RGB = C.CV_YUV2RGB

	CV_BayerBG2GRAY = C.CV_BayerBG2GRAY
	CV_BayerGB2GRAY = C.CV_BayerGB2GRAY
	CV_BayerRG2GRAY = C.CV_BayerRG2GRAY
	CV_BayerGR2GRAY = C.CV_BayerGR2GRAY

	//YUV 4:2:0 formats family
	CV_YUV2RGB_NV12 = C.CV_YUV2RGB_NV12
	CV_YUV2BGR_NV12 = C.CV_YUV2BGR_NV12
	CV_YUV2RGB_NV21 = C.CV_YUV2RGB_NV21
	CV_YUV2BGR_NV21 = C.CV_YUV2BGR_NV21
	CV_YUV420sp2RGB = C.CV_YUV420sp2RGB
	CV_YUV420sp2BGR = C.CV_YUV420sp2BGR

	CV_YUV2RGBA_NV12 = C.CV_YUV2RGBA_NV12
	CV_YUV2BGRA_NV12 = C.CV_YUV2BGRA_NV12
	CV_YUV2RGBA_NV21 = C.CV_YUV2RGBA_NV21
	CV_YUV2BGRA_NV21 = C.CV_YUV2BGRA_NV21
	CV_YUV420sp2RGBA = C.CV_YUV420sp2RGBA
	CV_YUV420sp2BGRA = C.CV_YUV420sp2BGRA

	CV_YUV2RGB_YV12 = C.CV_YUV2RGB_YV12
	CV_YUV2BGR_YV12 = C.CV_YUV2BGR_YV12
	CV_YUV2RGB_IYUV = C.CV_YUV2RGB_IYUV
	CV_YUV2BGR_IYUV = C.CV_YUV2BGR_IYUV
	CV_YUV2RGB_I420 = C.CV_YUV2RGB_I420
	CV_YUV2BGR_I420 = C.CV_YUV2BGR_I420
	CV_YUV420p2RGB  = C.CV_YUV420p2RGB
	CV_YUV420p2BGR  = C.CV_YUV420p2BGR

	CV_YUV2RGBA_YV12 = C.CV_YUV2RGBA_YV12
	CV_YUV2BGRA_YV12 = C.CV_YUV2BGRA_YV12
	CV_YUV2RGBA_IYUV = C.CV_YUV2RGBA_IYUV
	CV_YUV2BGRA_IYUV = C.CV_YUV2BGRA_IYUV
	CV_YUV2RGBA_I420 = C.CV_YUV2RGBA_I420
	CV_YUV2BGRA_I420 = C.CV_YUV2BGRA_I420
	CV_YUV420p2RGBA  = C.CV_YUV420p2RGBA
	CV_YUV420p2BGRA  = C.CV_YUV420p2BGRA

	CV_YUV2GRAY_420  = C.CV_YUV2GRAY_420
	CV_YUV2GRAY_NV21 = C.CV_YUV2GRAY_NV21
	CV_YUV2GRAY_NV12 = C.CV_YUV2GRAY_NV12
	CV_YUV2GRAY_YV12 = C.CV_YUV2GRAY_YV12
	CV_YUV2GRAY_IYUV = C.CV_YUV2GRAY_IYUV
	CV_YUV2GRAY_I420 = C.CV_YUV2GRAY_I420
	CV_YUV420sp2GRAY = C.CV_YUV420sp2GRAY
	CV_YUV420p2GRAY  = C.CV_YUV420p2GRAY

	//YUV 4:2:2 formats family
	CV_YUV2RGB_UYVY = C.CV_YUV2RGB_UYVY
	CV_YUV2BGR_UYVY = C.CV_YUV2BGR_UYVY
	//CV_YUV2RGB_VYUY = C.//CV_YUV2RGB_VYUY
	//CV_YUV2BGR_VYUY = C.//CV_YUV2BGR_VYUY
	CV_YUV2RGB_Y422 = C.CV_YUV2RGB_Y422
	CV_YUV2BGR_Y422 = C.CV_YUV2BGR_Y422
	CV_YUV2RGB_UYNV = C.CV_YUV2RGB_UYNV
	CV_YUV2BGR_UYNV = C.CV_YUV2BGR_UYNV

	CV_YUV2RGBA_UYVY = C.CV_YUV2RGBA_UYVY
	CV_YUV2BGRA_UYVY = C.CV_YUV2BGRA_UYVY
	//CV_YUV2RGBA_VYUY = C.//CV_YUV2RGBA_VYUY
	//CV_YUV2BGRA_VYUY = C.//CV_YUV2BGRA_VYUY
	CV_YUV2RGBA_Y422 = C.CV_YUV2RGBA_Y422
	CV_YUV2BGRA_Y422 = C.CV_YUV2BGRA_Y422
	CV_YUV2RGBA_UYNV = C.CV_YUV2RGBA_UYNV
	CV_YUV2BGRA_UYNV = C.CV_YUV2BGRA_UYNV

	CV_YUV2RGB_YUY2 = C.CV_YUV2RGB_YUY2
	CV_YUV2BGR_YUY2 = C.CV_YUV2BGR_YUY2
	CV_YUV2RGB_YVYU = C.CV_YUV2RGB_YVYU
	CV_YUV2BGR_YVYU = C.CV_YUV2BGR_YVYU
	CV_YUV2RGB_YUYV = C.CV_YUV2RGB_YUYV
	CV_YUV2BGR_YUYV = C.CV_YUV2BGR_YUYV
	CV_YUV2RGB_YUNV = C.CV_YUV2RGB_YUNV
	CV_YUV2BGR_YUNV = C.CV_YUV2BGR_YUNV

	CV_YUV2RGBA_YUY2 = C.CV_YUV2RGBA_YUY2
	CV_YUV2BGRA_YUY2 = C.CV_YUV2BGRA_YUY2
	CV_YUV2RGBA_YVYU = C.CV_YUV2RGBA_YVYU
	CV_YUV2BGRA_YVYU = C.CV_YUV2BGRA_YVYU
	CV_YUV2RGBA_YUYV = C.CV_YUV2RGBA_YUYV
	CV_YUV2BGRA_YUYV = C.CV_YUV2BGRA_YUYV
	CV_YUV2RGBA_YUNV = C.CV_YUV2RGBA_YUNV
	CV_YUV2BGRA_YUNV = C.CV_YUV2BGRA_YUNV

	CV_YUV2GRAY_UYVY = C.CV_YUV2GRAY_UYVY
	CV_YUV2GRAY_YUY2 = C.CV_YUV2GRAY_YUY2
	//CV_YUV2GRAY_VYUY = C.//CV_YUV2GRAY_VYUY
	CV_YUV2GRAY_Y422 = C.CV_YUV2GRAY_Y422
	CV_YUV2GRAY_UYNV = C.CV_YUV2GRAY_UYNV
	CV_YUV2GRAY_YVYU = C.CV_YUV2GRAY_YVYU
	CV_YUV2GRAY_YUYV = C.CV_YUV2GRAY_YUYV
	CV_YUV2GRAY_YUNV = C.CV_YUV2GRAY_YUNV

	// alpha premultiplication
	CV_RGBA2mRGBA = C.CV_RGBA2mRGBA
	CV_mRGBA2RGBA = C.CV_mRGBA2RGBA

	CV_RGB2YUV_I420 = C.CV_RGB2YUV_I420
	CV_BGR2YUV_I420 = C.CV_BGR2YUV_I420
	CV_RGB2YUV_IYUV = C.CV_RGB2YUV_IYUV
	CV_BGR2YUV_IYUV = C.CV_BGR2YUV_IYUV

	CV_RGBA2YUV_I420 = C.CV_RGBA2YUV_I420
	CV_BGRA2YUV_I420 = C.CV_BGRA2YUV_I420
	CV_RGBA2YUV_IYUV = C.CV_RGBA2YUV_IYUV
	CV_BGRA2YUV_IYUV = C.CV_BGRA2YUV_IYUV
	CV_RGB2YUV_YV12  = C.CV_RGB2YUV_YV12
	CV_BGR2YUV_YV12  = C.CV_BGR2YUV_YV12
	CV_RGBA2YUV_YV12 = C.CV_RGBA2YUV_YV12
	CV_BGRA2YUV_YV12 = C.CV_BGRA2YUV_YV12

	CV_COLORCVT_MAX = C.CV_COLORCVT_MAX
)

const (
	CV_BLUR_NO_SCALE = C.CV_BLUR_NO_SCALE
	CV_BLUR          = C.CV_BLUR
	CV_GAUSSIAN      = C.CV_GAUSSIAN
	CV_MEDIAN        = C.CV_MEDIAN
	CV_BILATERAL     = C.CV_BILATERAL

	CV_8U  = C.CV_8U
	CV_8S  = C.CV_8S
	CV_16U = C.CV_16U
	CV_16S = C.CV_16S
	CV_32S = C.CV_32S
	CV_32F = C.CV_32F
	CV_64F = C.CV_64F
)

/* Smoothes array (removes noise) */
func Smooth(src, dst *IplImage, smoothtype,
	param1, param2 int, param3, param4 float64) {
	C.cvSmooth(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(smoothtype),
		C.int(param1), C.int(param2), C.double(param3), C.double(param4),
	)
}

//CVAPI(void) cvSmooth( const CvArr* src, CvArr* dst,
//                      int smoothtype CV_DEFAULT(CV_GAUSSIAN),
//                      int param1 CV_DEFAULT(3),
//                      int param2 CV_DEFAULT(0),
//                      double param3 CV_DEFAULT(0),
//                      double param4 CV_DEFAULT(0));

/*
ConvertScale converts one image to another with optional linear transformation.
*/
func ConvertScale(a, b *IplImage, scale, shift float64) {
	C.cvConvertScale(unsafe.Pointer(a), unsafe.Pointer(b), C.double(scale), C.double(shift))
}

//CVAPI(void)  cvConvertScale( const CvArr* src,
//                             CvArr* dst,
//                             double scale CV_DEFAULT(1),
//                             double shift CV_DEFAULT(0) );

/* Converts input array pixels from one color space to another */
func CvtColor(src, dst *IplImage, code int) {
	C.cvCvtColor(unsafe.Pointer(src), unsafe.Pointer(dst), C.int(code))
}

//CVAPI(void)  cvCvtColor( const CvArr* src, CvArr* dst, int code );

/* Runs canny edge detector */
func Canny(image, edges *IplImage, threshold1, threshold2 float64, aperture_size int) {
	C.cvCanny(unsafe.Pointer(image), unsafe.Pointer(edges),
		C.double(threshold1), C.double(threshold2),
		C.int(aperture_size),
	)
}

//CVAPI(void)  cvCanny( const CvArr* image, CvArr* edges, double threshold1,
//                      double threshold2, int  aperture_size CV_DEFAULT(3) );

/* Calculates the first, second, third, or mixed image derivatives using an
* extended Sobel operator.  */
func Sobel(src, dst *IplImage, xorder, yorder, aperture_size int) {
	C.cvSobel(unsafe.Pointer(src), unsafe.Pointer(dst),
		C.int(xorder), C.int(yorder),
		C.int(aperture_size),
	)
}

// C: void cvSobel(const CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size=3 )

const (
	CV_INPAINT_NS    = C.CV_INPAINT_NS
	CV_INPAINT_TELEA = C.CV_INPAINT_TELEA
)

/* Inpaints the selected region in the image */
func Inpaint(src, inpaint_mask, dst *IplImage, inpaintRange float64, flags int) {
	C.cvInpaint(
		unsafe.Pointer(src),
		unsafe.Pointer(inpaint_mask),
		unsafe.Pointer(dst),
		C.double(inpaintRange),
		C.int(flags),
	)
}

//CVAPI(void) cvInpaint( const CvArr* src, const CvArr* inpaint_mask,
//                       CvArr* dst, double inpaintRange, int flags );

const (
	CV_THRESH_BINARY     = C.CV_THRESH_BINARY
	CV_THRESH_BINARY_INV = C.CV_THRESH_BINARY_INV
	CV_THRESH_TRUNC      = C.CV_THRESH_TRUNC
	CV_THRESH_TOZERO     = C.CV_THRESH_TOZERO
	CV_THRESH_TOZERO_INV = C.CV_THRESH_TOZERO_INV
	CV_THRESH_MASK       = C.CV_THRESH_MASK
	CV_THRESH_OTSU       = C.CV_THRESH_OTSU
)

/* Applies a fixed-level threshold to each array element. */
func Threshold(src, dst *IplImage, threshold, max_value float64, threshold_type int) {
	C.cvThreshold(
		unsafe.Pointer(src),
		unsafe.Pointer(dst),
		C.double(threshold),
		C.double(max_value),
		C.int(threshold_type),
	)
}

//CVAPI(double) cvThreshold( const CvArr* src, CvArr* dst, double threshold,
//                           double max_value, int threshold_type );

const (
	CV_ADAPTIVE_THRESH_MEAN_C     = C.CV_ADAPTIVE_THRESH_MEAN_C
	CV_ADAPTIVE_THRESH_GAUSSIAN_C = C.CV_ADAPTIVE_THRESH_GAUSSIAN_C
)

/* Applies an adaptive threshold to an array. */
func AdaptiveThreshold(src, dst *IplImage, max_value float64, adaptive_method,
	threshold_type, block_size int, thresh_C float64) {
	C.cvAdaptiveThreshold(
		unsafe.Pointer(src),
		unsafe.Pointer(dst),
		C.double(max_value),
		C.int(adaptive_method),
		C.int(threshold_type),
		C.int(block_size),
		C.double(thresh_C),
	)
}

//CVAPI(void) cvAdaptiveThreshold( const CvArr* src, CvArr* dst, double max_value,
//                                 int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C,
//                                 int threshold_type=CV_THRESH_BINARY,
//                                 int block_size=3, double param1=5 );

const (
	CV_MORPH_RECT    = 0
	CV_MORPH_ELLIPSE = 1
	CV_MORPH_CROSS   = 2
	//CV_SHAPE_CUSTOM  = 3 // TODO: currently we don't support a fully custom kernel
)

/* Returns a structuring element of the specified size and shape for morphological operations. */
func CreateStructuringElement(cols, rows, anchor_x, anchor_y, shape int) *IplConvKernel {
	return (*IplConvKernel)(C.cvCreateStructuringElementEx(
		C.int(cols),
		C.int(rows),
		C.int(anchor_x),
		C.int(anchor_y),
		C.int(shape),
		nil, // TODO: currently we don't support a fully custom kernel
	))
}

//CVAPI(IplConvKernel*) cvCreateStructuringElementEx( int cols, int rows, int anchor_x,
//                                                    int anchor_y, int shape, int* values=NULL )

/* Releases the structuring element */
func (k *IplConvKernel) ReleaseElement() {
	C.cvReleaseStructuringElement(
		(**C.IplConvKernel)(unsafe.Pointer(&k)),
	)
}

//CVAPI(void) cvReleaseStructuringElement( IplConvKernel** element );

/* Dilates an image by using a specific structuring element. */
func Dilate(src, dst *IplImage, element *IplConvKernel, iterations int) {
	C.cvDilate(
		unsafe.Pointer(src),
		unsafe.Pointer(dst),
		(*C.IplConvKernel)(unsafe.Pointer(element)),
		C.int(iterations),
	)
}

//CVAPI(void) cvDilate( const CvArr* src, CvArr* dst, IplConvKernel* element=NULL,
//                      int iterations=1 );

/* Erodes an image by using a specific structuring element. */
func Erode(src, dst *IplImage, element *IplConvKernel, iterations int) {
	C.cvErode(
		unsafe.Pointer(src),
		unsafe.Pointer(dst),
		(*C.IplConvKernel)(unsafe.Pointer(element)),
		C.int(iterations),
	)
}

//CVAPI(void) cvErode( const CvArr* src, CvArr* dst, IplConvKernel* element=NULL,
//                     int iterations=1 );

const (
	CV_MORPH_OPEN     = C.CV_MOP_OPEN
	CV_MORPH_CLOSE    = C.CV_MOP_CLOSE
	CV_MORPH_GRADIENT = C.CV_MOP_GRADIENT
	CV_MORPH_TOPHAT   = C.CV_MOP_TOPHAT
	CV_MORPH_BLACKHAT = C.CV_MOP_BLACKHAT
)

/* Performs advanced morphological transformations. */
func MorphologyEx(src, dst, temp *IplImage, element *IplConvKernel, operation int, iterations int) {
	C.cvMorphologyEx(
		unsafe.Pointer(src),
		unsafe.Pointer(dst),
		unsafe.Pointer(temp),
		(*C.IplConvKernel)(unsafe.Pointer(element)),
		C.int(operation),
		C.int(iterations),
	)
}

//CVAPI(void) cvMorphologyEx( const CvArr* src, CvArr* dst, CvArr* temp,
//                            IplConvKernel* element, int operation, int iterations=1 );
