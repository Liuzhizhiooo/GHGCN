# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@notice  : Raster data processing toolbox
"""


from osgeo import gdal, gdalconst, gdal_array

# 错误提示,便于查错
gdal.UseExceptions()
# gdal.DontUseExceptions()

def getTifDataset(srcfp, mode=gdalconst.GA_ReadOnly):
    """
    获取数据
    """
    ds = gdal.Open(srcfp, mode)
    if ds is None: raise ValueError(f"Open {srcfp} Failed !!")
    return ds


def readTif(fp, xoff=None, yoff=None, xwinsize=None, ywinsize=None):
    """
    读取fp文件的数组
    """
    ds = getTifDataset(fp)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    if xoff is None: xoff = 0
    if yoff is None: yoff = 0
    if xwinsize is None: xwinsize = xsize
    if ywinsize is None: ywinsize = ysize

    # LoadFile <=> getTifDataset + DatasetReadAsArray
    return gdal_array.LoadFile(fp, xoff=xoff, yoff=yoff, xsize=xwinsize, ysize=ywinsize)



def createSameFormatTif(array, dstfp, reffp=None):
    """
    create tif by array with reference to reffp
    """
    if gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype) is None:
        raise ValueError(f"the datatype of array {array.dtype} is not supported!")
    dstDs = gdal_array.SaveArray(array, dstfp, format="GTiff", prototype=reffp)
    dstDs = None


def setColorTable(fp, colorDict):
    """
    set the colortable for tif
    colorDict={(0, 100, 100): 0, (0, 200, 100): 1, ...}
    """
    ds = getTifDataset(fp, mode=gdalconst.GA_Update)
    band = ds.GetRasterBand(1)
    assert band.DataType in [gdalconst.GDT_Byte, gdalconst.GDT_UInt16], \
        f"SetColorTable() only supported for Byte or UInt16 bands in TIFF format."

    if ds.RasterCount > 1:
        raise ValueError(f"Multi-Spectral images cannot add color table!")

    colorTable = gdal.ColorTable()
    for color, colorIdx in colorDict.items():
        colorTable.SetColorEntry(colorIdx, color)
    band.SetColorTable(colorTable)
    ds = None
